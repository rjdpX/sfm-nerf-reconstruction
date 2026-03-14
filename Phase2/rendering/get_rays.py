import torch
import numpy as np
import matplotlib.pyplot as plt


# Getting the rays
def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor):

    # Ensure we operate on the same device/dtype as c2w
    device = c2w.device
    dtype = c2w.dtype

    o = c2w[:3, 3]

    u, v = torch.meshgrid(
        torch.arange(W, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        indexing="xy",
    )  # [H, W]
    u = u.reshape(-1)  # [H * W]
    v = v.reshape(-1)  # [H * W]

    d = torch.stack((u - W / 2, -(v - H / 2), -torch.ones_like(u) * focal), dim=-1)
    d = (c2w[:3, :3] @ d[..., None]).squeeze(-1)
    d = d / torch.linalg.norm(d, dim=-1, keepdim=True)
    o = o.expand(d.shape[0], 3)

    return o, d


# Visualization of rays
def plot_rays(o: np.array, d: np.array, t=1.0):

    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection="3d")

    pt1 = o
    pt2 = o + t * d
    for p1, p2 in zip(pt1, pt2):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c="C0")

    plt.show()
