import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from .utils import Rays
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def rays_from_h5(start, end, path, device="cuda:0"):
    with h5py.File(os.path.join(path, "samples.h5"), 'r') as f:
        data = np.array(f['dataset'][start:end, :, :3])
        data = torch.from_numpy(data).to(torch.float32)
        data = data.to(device)
    return data

def get_x(start, end, path, device="cuda:0"):
    with h5py.File(os.path.join(path, "x.h5"), 'r') as f:
        x = torch.tensor(f['dataset'][start:end], device=device)
    return x

def get_y(start, end, path, device="cuda:0"):
    with h5py.File(os.path.join(path, "y.h5"), 'r') as f:
        y = torch.tensor(f['dataset'][start:end], device=device)
    return y

def get_file_indices(start, end, path):
    with h5py.File(os.path.join(path, "file_indices.h5"), 'r') as f:
        indices = torch.tensor(f['dataset'][start:end])
        indices = indices.type(torch.long)
    return indices


def _load_renderings_metadata(root_fp: str, subject_id: str, split: str, args):
    """Load images from disk."""

    with open(
        os.path.join(root_fp, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    fnames = []
    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        camtoworlds.append(frame["transform_matrix"])

    camtoworlds = np.stack(camtoworlds, axis=0)

    if args.version == "simulated":
        h, w = args.img_shape, args.img_shape
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

    with h5py.File(os.path.join(root_fp, "samples.h5"), 'r') as f:
        length = f['dataset'].shape[0]
    return length, camtoworlds, K

class SubjectLoaderIterable(torch.utils.data.IterableDataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    OPENGL_CAMERA = True

    def __init__(
        self,
        counter,
        num_rays,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        subject_id = None, 
        near: float = None,
        far: float = None,
        args = None,
         
    ):
        super().__init__()
        self.WIDTH = args.img_shape
        self.HEIGHT = args.img_shape

        self.args = args
        self.split = split
        self.num_rays = num_rays
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.root_fp = root_fp
        self.subject_id = subject_id
        self.len, self.camtoworlds, K = _load_renderings_metadata(
                self.root_fp, self.subject_id, self.split, args
        )

        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        
        self.K = K
        self.camtoworlds = self.camtoworlds.to(self.args.device)
        self.K = self.K.to(self.args.device)
        self.counter = counter
    

    def __iter__(self):
        while True:
            data = self.fetch_data()
            data = self.preprocess(data)
            yield data
            
    def __len__(self):
        return self.len

    def preprocess(self, data):
        pixels, rays = data["rgba"], data["rays"]   
        pixels = torch.clip(pixels[...,self.args.t_min:self.args.t_max, :3]/self.args.dataset_scale, 0, 1)**(1/self.args.gamma)
    
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": None,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }


    def fetch_data(self):
        with self.num_rays.get_lock():  # ensure that only one worker can update num_rays at a time
            num_rays = self.num_rays.value 
        with self.counter.get_lock():  # ensure that only one worker can update the counter at a time
                start_index = self.counter.value
                end_index = start_index + num_rays
                if end_index > self.len:
                    self.counter.value = end_index - self.len
                elif end_index == self.len:
                    self.counter.value = 0
                    start_index = 0
                    end_index = num_rays
                else:
                    self.counter.value += num_rays

        # Load chunk of data.
        if end_index <= self.len:
            data = rays_from_h5(start_index, end_index, self.root_fp, self.args.device)
            x = get_x(start_index, end_index, self.root_fp, self.args.device)
            y = get_y(start_index, end_index, self.root_fp, self.args.device)
            cam_index = get_file_indices(start_index, end_index, self.root_fp)

        elif start_index < self.len and end_index > self.len:
            first_data = rays_from_h5(start_index, self.len, self.root_fp, self.args.device)
            second_data = rays_from_h5(0, end_index - self.len, self.root_fp, self.args.device)
            first_x = get_x(start_index, self.len, self.root_fp, self.args.device)
            second_x = get_x(0, end_index - self.len, self.root_fp, self.args.device)
            first_y = get_y(start_index, self.len, self.root_fp, self.args.device)
            second_y = get_y(0, end_index - self.len, self.root_fp, self.args.device)
            first_cam_index = get_file_indices(start_index, self.len, self.root_fp)
            second_cam_index = get_file_indices(0, end_index - self.len, self.root_fp)

            data = torch.cat((first_data, second_data),axis=0)
            x = torch.cat((first_x, second_x),axis=0)
            y = torch.cat((first_y, second_y),axis=0)
            cam_index = torch.cat((first_cam_index,second_cam_index),axis=0)

    
        c2w = self.camtoworlds[cam_index]
        rgba = data
        
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

        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, -1))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }