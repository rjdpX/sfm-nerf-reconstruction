import argparse
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt
import os
import json
import math

from NeRFModel import *
from dataloader.get_nerf_dataset import NeRFDataset
from Train import train_loop
from Test import test_loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


def loadJSONTrainData(jsonTrainFile="nerf_synthetic/lego/transforms_train.json"):
    """_summary_

    Args:
        jsonTrainFile (_type_): _description_
    """

    with open(jsonTrainFile, "r") as fp:
        jsonTrainData = json.load(fp)
    print(f"[INFO] Focal length train: {jsonTrainData['camera_angle_x']}")
    print(f"[INFO] Number of frames train: {len(jsonTrainData['frames'])}")
    firstFrame = jsonTrainData["frames"][0]
    tMat = np.array(firstFrame["transform_matrix"])
    fName = firstFrame["file_path"]
    print(tMat)
    print(fName)
    return jsonTrainData


def load_split(
    json_path, root_dir, H=400, W=400, downscale=4, batch_size=1024, shuffle=True
):
    """
    Input:
        json_path: path to transforms_*.json
        root_dir: dataset root directory
    Outputs:
        dataloader: torch DataLoader
        camera_info: image width, height, focal length
    """

    with open(json_path, "r") as fp:
        json_data = json.load(fp)

    # Convert horizontal FOV (radians) to focal length in pixels.
    # Compute at original resolution, then scale with downscale.
    focal = 0.5 * W / math.tan(0.5 * json_data["camera_angle_x"])
    focal = focal / downscale

    dataset = NeRFDataset(
        json_data,
        root_dir=root_dir,
        H=H,
        W=W,
        focal=focal,
        downscale=downscale,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True
    )

    return dataloader, dataset.camera_info


# def PixelToRay(camera_info, pose, pixelPosition, args):
#     """
#     Input:
#         camera_info: image width, height, camera matrix
#         pose: camera pose in world frame
#         pixelPoition: pixel position in the image
#         args: get near and far range, sample rate ...
#     Outputs:
#         ray origin and direction
#     """


# def generateBatch(images, poses, camera_info, args):
#     """
#     Input:
#         images: all images in dataset
#         poses: corresponding camera pose in world frame
#         camera_info: image width, height, camera matrix
#         args: get batch size related information
#     Outputs:
#         A set of rays
#     """


# def render(model, rays_origin, rays_direction, args):
#     """
#     Input:
#         model: NeRF model
#         rays_origin: origins of input rays
#         rays_direction: direction of input rays
#     Outputs:
#         rgb values of input rays
#     """


def loss(groundtruth, prediction):
    # Pixel-wise mean squared error for RGB regression.
    return torch.nn.functional.mse_loss(prediction, groundtruth)


def train(images, poses, camera_info, args):
    raise NotImplementedError("Use train_loop in Phase2/train/train.py")


def test(images, poses, camera_info, args):
    raise NotImplementedError("Use test_loop in Phase2/train/Test.py")


def main(args):
    # load data
    print("Loading data...")
    train_json = os.path.join(args.data_path, "transforms_train.json")
    val_json = os.path.join(args.data_path, "transforms_val.json")
    test_json = os.path.join(args.data_path, "transforms_test.json")

    model = NeRFmodel(args.n_pos_freq, args.n_dirc_freq).to(device)

    if args.mode == "train":
        print("Start training")
        train_loader, camera_info = load_split(
            train_json,
            args.data_path,
            H=args.H,
            W=args.W,
            downscale=args.downscale,
            batch_size=args.batch_size,
            shuffle=True,
        )
        train_loop(model, train_loader, camera_info, args, device, loss)

        if os.path.exists(val_json):
            print("Start validation")
            val_loader, camera_info = load_split(
                val_json,
                args.data_path,
                H=args.H,
                W=args.W,
                downscale=args.downscale,
                batch_size=args.batch_size,
                shuffle=False,
            )
            test_loop(model, val_loader, camera_info, args, device, loss)
    elif args.mode == "test":
        print("Start testing")
        args.load_checkpoint = True
        test_loader, camera_info = load_split(
            test_json,
            args.data_path,
            H=args.H,
            W=args.W,
            downscale=args.downscale,
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loop(model, test_loader, camera_info, args, device, loss)
    elif args.mode == "val":
        print("Start validation")
        val_loader, camera_info = load_split(
            val_json,
            args.data_path,
            H=args.H,
            W=args.W,
            downscale=args.downscale,
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loop(model, val_loader, camera_info, args, device, loss)


def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="nerf_synthetic/lego", help="dataset path"
    )
    parser.add_argument("--mode", default="train", help="train/test/val")
    parser.add_argument("--lrate", default=5e-4, help="training learning rate")
    parser.add_argument(
        "--n_pos_freq",
        default=10,
        help="number of positional encoding frequencies for position",
    )
    parser.add_argument(
        "--n_dirc_freq",
        default=4,
        help="number of positional encoding frequencies for viewing direction",
    )
    parser.add_argument(
        "--n_rays_batch", default=32 * 32 * 4, help="number of rays per batch"
    )

    parser.add_argument("--n_sample", default=400, help="number of sample per ray")
    parser.add_argument("--n_importance", default=0, help="number of fine samples")
    parser.add_argument("--near", default=2.0, help="near bound")
    parser.add_argument("--far", default=6.0, help="far bound")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--log_every", default=100, help="logging interval")
    parser.add_argument("--render_chunk", default=0, help="chunk size for rendering")
    parser.add_argument("--H", type=int, default=400, help="image height")
    parser.add_argument("--W", type=int, default=400, help="image width")
    parser.add_argument("--downscale", type=int, default=4, help="downscale factor")
    parser.add_argument("--batch_size", type=int, default=1, help="images per batch")
    parser.add_argument(
        "--max_iters", default=10000, help="number of max iterations for training"
    )
    parser.add_argument("--logs_path", default="./logs/", help="logs path")
    parser.add_argument(
        "--checkpoint_path",
        default="./Phase2/example_checkpoint/",
        help="checkpoints path",
    )
    parser.add_argument(
        "--load_checkpoint", default=True, help="whether to load checkpoint or not"
    )
    parser.add_argument(
        "--save_ckpt_iter", default=1000, help="num of iteration to save checkpoint"
    )
    parser.add_argument(
        "--images_path", default="./image/", help="folder to store images"
    )
    return parser


if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)
