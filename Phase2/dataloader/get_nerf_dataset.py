import os
import torch
import numpy as np
from PIL import Image


class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, json_data, root_dir, H=800, W=800, focal=None, downscale=4):
        self.frames = json_data["frames"]
        self.root_dir = root_dir
        self.focal = focal
        self.H, self.W = H // downscale, W // downscale

        self.images = []
        self.poses = []

        self.camera_info = {"H": self.H, "W": self.W, "focal": self.focal}

        for frame in self.frames:
            image_path = os.path.join(root_dir, frame["file_path"][2:] + ".png")
            img = Image.open(image_path).convert("RGB").resize((self.W, self.H))
            img = np.array(img) / 255.0
            img = torch.tensor(img, dtype=torch.float32)

            # Get camera pose
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)

            self.images.append(img)
            self.poses.append(c2w)

        self.images = torch.stack(self.images, 0)
        self.poses = torch.stack(self.poses, 0)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.poses[idx], self.camera_info
