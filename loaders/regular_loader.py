"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from .utils import Rays
import tqdm
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

def _load_renderings(root_fp: str, subject_id: str, split: str, args):
    """Load images from disk."""

    with open(
        os.path.join(root_fp, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    for i in tqdm.tqdm(range(len(meta["frames"]))):

        frame = meta["frames"][i]
        fname = os.path.join(root_fp, frame["filepath"][:-2]+"h5")
        rgba = read_h5(fname)[...,args.t_min:args.t_max, :3]
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)

    
    images = np.stack(images, axis=0)
    
    if args.version == "captured":
        images = images.mean(-1)[..., None]
        
    if split == "test":
        quotient = images.shape[1]//args.img_shape_test
        times_downsample = int(np.log2(quotient))
    
        for i in range(times_downsample):
            images = (images[:, 1::2, ::2] + images[:, ::2, ::2] + images[:, 1::2, 1::2] + images[:, ::2, 1::2])/4
        
    images = np.clip(images/args.dataset_scale, 0, 1)**(1/args.gamma)

    images = images.astype(np.float32)

    camtoworlds = np.stack(camtoworlds, axis=0)

    if args.version == "simulated":
        h, w = images.shape[1:3]
        camera_angle_x = float(meta["camera"])
        focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
        K = torch.tensor(
            [
                [focal, 0, w / 2.0],
                [0, focal, h / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32)
    else:
        K = torch.tensor(np.array(meta["camera"])).to(torch.float32)
        if split == "test":
            K /= 2**times_downsample

    return images, camtoworlds, K


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    OPENGL_CAMERA = True

    def __init__(
        self,
        root_fp: str,
        split: str,
        subject_id = None,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        args = None, 
    ):
        super().__init__()
        if split == "test":
            self.WIDTH = args.img_shape_test
            self.HEIGHT = args.img_shape_test
        else:
            self.WIDTH = args.img_shape
            self.HEIGHT = args.img_shape

        self.args = args
        self.split = split
        self.num_rays = num_rays
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.images, self.camtoworlds, K = _load_renderings(
            root_fp, subject_id, split, args
        )
        self.images = torch.from_numpy(self.images).to(torch.float32)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = K
        self.camtoworlds = self.camtoworlds.to(args.device)
        self.K = self.K.to(args.device)
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data
    
    @torch.no_grad()
    def __iter__(self):
        while True:
            data = self.fetch_data(0)
            data = self.preprocess(data)
            yield data



    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels = rgba

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        pixels = pixels.to(self.args.device)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.camtoworlds.device,
                )
            else:
                image_id = [index] * num_rays
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=self.camtoworlds.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.camtoworlds.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.camtoworlds.device),
                torch.arange(self.HEIGHT, device=self.camtoworlds.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgba = self.images[image_id, y, x]  # (num_rays, 4)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            if self.args.version == "simulated":
                rgba = torch.reshape(rgba, (num_rays, self.args.n_bins, 3))
            else:
                rgba = torch.reshape(rgba, (num_rays, self.args.n_bins, 1))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            if self.args.version == "simulated":
                rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, self.args.n_bins, 3))
            else:
                rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, self.args.n_bins, 1))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }
