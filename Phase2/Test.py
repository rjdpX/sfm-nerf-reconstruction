import os
import torch
import imageio
import numpy as np

from rendering.get_rays import get_rays
from rendering.sampler import stratified_sample, hierarchical_sample
from rendering.volume_render import volume_render


def _render_rays(
    model,
    rays_o,
    rays_d,
    near,
    far,
    n_samples,
    n_importance=0,
    perturb=False,
):
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


def test_loop(model, dataloader, camera_info, args, device, loss_fn=None):
    """Evaluation loop. Renders full images in chunks."""
    model.eval()

    near = getattr(args, "near", 2.0)
    far = getattr(args, "far", 6.0)
    n_importance = getattr(args, "n_importance", 0)
    chunk = getattr(args, "render_chunk", 0)
    if not chunk:
        chunk = args.n_rays_batch

    H = camera_info["H"]
    W = camera_info["W"]
    focal = camera_info["focal"]

    results = []
    save_dir = getattr(args, "images_path", None)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    prefix = getattr(args, "mode", "test")
    img_idx = 0

    with torch.no_grad():
        for images, poses, cam in dataloader:
            if isinstance(cam, (list, tuple)):
                cam = cam[0]

            images = images.to(device, non_blocking=True)
            poses = poses.to(device, non_blocking=True)

            for i in range(images.shape[0]):
                rays_o, rays_d = get_rays(H, W, focal, poses[i])
                rays_o = rays_o.to(device)
                rays_d = rays_d.to(device)

                # Chunked rendering to avoid OOM
                rgb_chunks = []
                for j in range(0, rays_o.shape[0], chunk):
                    ro = rays_o[j : j + chunk]
                    rd = rays_d[j : j + chunk]
                    rgb_chunk, _, _, _ = _render_rays(
                        model,
                        ro,
                        rd,
                        near,
                        far,
                        args.n_sample,
                        n_importance=n_importance,
                        perturb=False,
                    )
                    rgb_chunks.append(rgb_chunk)

                rgb = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3)

                result = {"rgb": rgb}
                if loss_fn is not None:
                    gt = images[i].reshape(-1, 3)
                    result["loss"] = loss_fn(gt, rgb.reshape(-1, 3)).item()

                results.append(result)

                if save_dir:
                    rgb_img = rgb.detach().cpu().clamp(0.0, 1.0).numpy()
                    rgb_img = (rgb_img * 255.0).astype(np.uint8)
                    out_path = os.path.join(
                        save_dir, f"{prefix}_render_{img_idx:04d}.png"
                    )
                    imageio.imwrite(out_path, rgb_img)
                    img_idx += 1

    return results
