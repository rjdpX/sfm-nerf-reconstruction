import os
import torch
from rendering.get_rays import get_rays
from rendering.volume_render import volume_render
from torch.utils.tensorboard import SummaryWriter
from rendering.sampler import stratified_sample, hierarchical_sample


def _render_rays(
    model,
    rays_o,
    rays_d,
    near,
    far,
    n_samples,
    n_importance=0,
    perturb=True,
):
    """Render a batch of rays using (optional) hierarchical sampling."""
    pts, z_vals = stratified_sample(rays_o, rays_d, near, far, n_samples, perturb)

    n_rays, n_samples = z_vals.shape
    dirs = rays_d[:, None, :].expand(n_rays, n_samples, 3)

    radiance = model(
        pts.reshape(-1, 3),
        dirs.reshape(-1, 3),
    ).reshape(n_rays, n_samples, 4)

    rgb_map, depth_map, acc_map, weights = volume_render(radiance, z_vals, rays_d)

    if n_importance > 0:
        pts_fine, z_vals_fine = hierarchical_sample(
            rays_o, rays_d, z_vals, weights, n_importance, perturb
        )
        n_rays, n_samples_fine = z_vals_fine.shape
        dirs_fine = rays_d[:, None, :].expand(n_rays, n_samples_fine, 3)
        radiance_fine = model(
            pts_fine.reshape(-1, 3),
            dirs_fine.reshape(-1, 3),
        ).reshape(n_rays, n_samples_fine, 4)
        rgb_map, depth_map, acc_map, weights = volume_render(
            radiance_fine, z_vals_fine, rays_d
        )

    return rgb_map, depth_map, acc_map, weights


def train_loop(model, dataloader, camera_info, args, device, loss_fn):
    """Main training loop."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)

    near = getattr(args, "near", 2.0)
    far = getattr(args, "far", 6.0)
    n_importance = getattr(args, "n_importance", 0)

    H = camera_info["H"]
    W = camera_info["W"]
    focal = camera_info["focal"]

    writer = SummaryWriter(log_dir=args.logs_path)

    ckpt_dir = args.checkpoint_path
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resume from checkpoint
    if getattr(args, "load_checkpoint", False):
        ckpt_path = os.path.join(ckpt_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optim"])
            start_step = ckpt.get("step", 0)
        else:
            start_step = 0
    else:
        start_step = 0
    # step = 0
    step = start_step
    for epoch in range(getattr(args, "num_epochs", 1)):
        for images, poses, cam in dataloader:
            # DataLoader may return a list of identical dicts for cam
            if isinstance(cam, (list, tuple)):
                cam = cam[0]

            images = images.to(device, non_blocking=True)
            poses = poses.to(device, non_blocking=True)

            # Build rays for each image in the batch
            rays_o_list = []
            rays_d_list = []
            rgb_list = []

            for i in range(images.shape[0]):
                rays_o, rays_d = get_rays(H, W, focal, poses[i])
                rays_o_list.append(rays_o)
                rays_d_list.append(rays_d)
                rgb_list.append(images[i].reshape(-1, 3))

            rays_o = torch.cat(rays_o_list, dim=0)
            rays_d = torch.cat(rays_d_list, dim=0)
            target_rgb = torch.cat(rgb_list, dim=0)

            # Random ray batch
            n_rays_batch = args.n_rays_batch
            idx = torch.randint(0, rays_o.shape[0], (n_rays_batch,), device=device)
            rays_o = rays_o[idx]
            rays_d = rays_d[idx]
            target_rgb = target_rgb[idx]

            pred_rgb, _, _, _ = _render_rays(
                model,
                rays_o,
                rays_d,
                near,
                far,
                args.n_sample,
                n_importance=n_importance,
                perturb=True,
            )

            loss = loss_fn(target_rgb, pred_rgb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % getattr(args, "log_every", 100) == 0:
                print(f"[train] step={step} loss={loss.item():.6f}")
                writer.add_scalar("loss/train", loss.item(), step)

            if step % args.save_ckpt_iter == 0 and step > 0:
                ckpt_path = os.path.join(ckpt_dir, "latest.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "step": step,
                    },
                    ckpt_path,
                )
            step += 1

            if step >= args.max_iters:
                ckpt_path = os.path.join(ckpt_dir, "latest.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "step": step,
                    },
                    ckpt_path,
                )
                writer.close()
                return model

    writer.close()
    return model
