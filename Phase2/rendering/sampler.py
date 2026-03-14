import torch


def stratified_sample(
    o: torch.Tensor, d: torch.Tensor, near, far, n_samples: int, perturb: bool = True
):
    """Stratified sampling along rays between near and far

    Args:
        o (torch.Tensor): ray origin
        d (torch.Tensor): ray direction
        near (_type_): float or [N] tensor
        far (_type_): float or [N] tensor
        n_samples (int): number of samples per ray
        perturb (bool, optional): if True, stratefied jittering within bins
    Return :
        pts: [N, n_samples, 3] sample points
        z_vals: [N, n_samples] depths along each ray
    """

    if o.ndim != 2 or d.ndim != 2:
        raise ValueError("Rays o and d must be [N, 3] tensors")

    device = o.device
    dtype = o.dtype
    n_rays = o.shape[0]

    if not torch.is_tensor(near):
        near = torch.tensor(near, device=device, dtype=dtype)
    if not torch.is_tensor(far):
        far = torch.tensor(far, device=device, dtype=dtype)

    if near.ndim == 0:
        near = near.expand(n_rays)
    if far.ndim == 0:
        far = far.expand(n_rays)

    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=device, dtype=dtype)
    z_vals = near[:, None] * (1.0 - t_vals[None, :]) + far[:, None] * t_vals[None, :]

    if perturb:
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=device, dtype=dtype)
        z_vals = lower + (upper - lower) * t_rand

    pts = o[:, None, :] + d[:, None, :] * z_vals[..., None]
    return pts, z_vals


def sample_pdf(
    bins: torch.Tensor, weights: torch.Tensor, n_samples: int, det: bool = False
):
    """PDF Sampling using inverse transform sampling

    Args:
        bins (torch.Tensor): bin centers (monotonic)
        weights (torch.Tensor): [N, n_bins] weights per bin
        n_samples (int): number of samples
        det (bool, optional): if True, deterministic sampling(linespace). Defaults to False.
    """
    device = bins.device
    dtype = bins.dtype

    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [N, n_bins+1]

    if det:
        u = torch.linspace(0.0, 1.0, steps=n_samples, device=device, dtype=dtype)
        u = u.expand(cdf.shape[0], n_samples)

    else:
        u = torch.rand(cdf.shape[0], n_samples, device=device, dtype=dtype)

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)

    cdf_below = torch.gather(cdf, 1, below)
    cdf_above = torch.gather(cdf, 1, above)

    # bins are centers, build bin edges by padding
    bins_pad = torch.cat([bins[:, :1], bins, bins[:, -1:]], dim=-1)
    bins_below = torch.gather(bins_pad, 1, below)
    bins_above = torch.gather(bins_pad, 1, above)

    denom = cdf_above - cdf_below
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom
    samples = bins_below + t * (bins_above - bins_below)
    return samples


def hierarchical_sample(
    o: torch.Tensor,
    d: torch.Tensor,
    z_vals: torch.Tensor,
    weights: torch.Tensor,
    n_importance: int,
    perturb: bool = True,
):
    """Hierarchical (importance) sampling using coarse weights

    Args:
        o (torch.Tensor): [N, 3]
        d (torch.Tensor): [N, 3]
        z_vals (torch.Tensor): [N, n_samples]
        weights (torch.Tensor): [N, n_samples] coarse weights
        n_importance (int): number of extra samples
        perturb (bool, optional): if True, stochastic sampling. Defaults to True.

    Returns:
        pts_fine: [N, n_samples+n_importance, 3]
        z_vals_fine: [N, n_samples+n_importance]
    """
    # Use midpoints between samples as bins
    z_mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
    weights = weights[:, 1:-1]  # skip endpoints for stability
    z_samples = sample_pdf(z_mids, weights, n_importance, det=not perturb)
    z_samples = z_samples.detach()

    z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
    pts_fine = o[:, None, :] + d[:, None, :] * z_vals_fine[..., None]
    return pts_fine, z_vals_fine


def sample_stratified(
    o: torch.Tensor,
    d: torch.Tensor,
    near,
    far,
    n_samples: int,
    perturb: bool = True,
):
    return stratified_sample(o, d, near, far, n_samples, perturb)
