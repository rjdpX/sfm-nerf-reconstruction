import torch


def volume_render(
    radiance_field: torch.Tensor, z_vals: torch.Tensor, rays_d: torch.Tensor
):
    """
    Volume rendering for NeRF.

    Args:
        radiance_field: [N, S, 4] where last dim is (rgb(3), sigma(1))
        z_vals: [N, S] sample depths
        rays_d: [N, 3] ray directions

    Returns:
        rgb_map: [N, 3]
        depth_map: [N]
        acc_map: [N]
        weights: [N, S]
    """
    if radiance_field.ndim != 3 or radiance_field.shape[-1] != 4:
        raise ValueError("radiance_field must be [N, S, 4]")

    rgb = radiance_field[..., :3]
    sigma = radiance_field[..., 3]

    # Distances between adjacent samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

    # Account for ray direction norm (if not unit length)
    dists = dists * torch.linalg.norm(rays_d, dim=-1, keepdim=True)

    alpha = 1.0 - torch.exp(-sigma * dists)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1]

    weights = alpha * transmittance

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map, weights
