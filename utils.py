from datetime import datetime
import random
from typing import Optional
import ast
import configargparse
import os
import numpy as np
import torch
from loaders.utils import Rays, namedtuple_map
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.grid import ray_aabb_intersect, traverse_grids
from misc.transient_volrend import (
    accumulate_along_rays_,
    render_weight_from_density,
    rendering,
    shift_transient_grid_sample_3d,
)
from torch.utils.tensorboard import SummaryWriter
import shutil


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_image_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    args = None,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    
    rendering_func = rendering
    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        
        return rgbs, sigmas.squeeze(-1)
    
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            sigmas = radiance_field.query_density(positions, t)
        else:
            sigmas = radiance_field.query_density(positions)
        return sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = estimator.sampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, _ = rendering_func(
            t_starts,
            t_ends,
            ray_indices,
            args = args,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )

@torch.no_grad()
def render_image_with_occgrid_test(
    max_samples: int,
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    args = None,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    early_stop_eps: float = 1e-4,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    n_output_dim = 3, 
    prop_delay = True, 
    unwarp = False, 
    test_chunk_size = 48000
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = rays.origins[ray_indices]
        t_dirs = rays.viewdirs[ray_indices]
        positions = (
            t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0
        )
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)

        return rgbs, sigmas.squeeze(-1)

    device = rays.origins.device
    opacity = torch.zeros(num_rays, 1, device=device)
    depth = torch.zeros(num_rays, 1, device=device)
    rgb = torch.zeros(num_rays, n_output_dim, device=device)

    ray_mask = torch.ones(num_rays, device=device).bool()

    # 1 for synthetic scenes, 4 for real scenes
    min_samples = 1 if cone_angle == 0 else 4

    iter_samples = total_samples = 0

    rays_o = rays.origins
    rays_d = rays.viewdirs

    near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
    far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

    t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, estimator.aabbs)

    n_grids = estimator.binaries.size(0)

    if n_grids > 1:
        t_sorted, t_indices = torch.sort(torch.cat([t_mins, t_maxs], -1), -1)
    else:
        t_sorted = torch.cat([t_mins, t_maxs], -1)
        t_indices = torch.arange(
            0, n_grids * 2, device=t_mins.device, dtype=torch.int64
        ).expand(num_rays, n_grids * 2)

    opc_thre = 1 - early_stop_eps

    while iter_samples < max_samples:

        n_alive = ray_mask.sum().item()
        if n_alive == 0:
            break

        # The number of samples to add on each ray.
        n_samples = max(min(num_rays // n_alive, 64), min_samples)
        iter_samples += n_samples

        # ray marching
        (intervals, samples, termination_planes) = traverse_grids(
            # rays
            rays_o,  # [n_rays, 3]
            rays_d,  # [n_rays, 3]
            # grids
            estimator.binaries,  # [m, resx, resy, resz]
            estimator.aabbs,  # [m, 6]
            # options
            near_planes,  # [n_rays]
            far_planes,  # [n_rays]
            render_step_size,
            cone_angle,
            n_samples,
            True,
            ray_mask,
            # pre-compute intersections
            t_sorted,  # [n_rays, m*2]
            t_indices,  # [n_rays, m*2]
            hits,  # [n_rays, m]
        )
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = samples.ray_indices[samples.is_valid]
        packed_info = samples.packed_info

        # Get rgb and sigma from radiance field.
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        if rgbs.shape[0] != 0:
            if prop_delay:
                shift = (t_starts + t_ends)[:, None]/ 2.0
                rgbs = rgbs.reshape(rgbs.shape[0], args.n_bins, -1)
                chunk_size = test_chunk_size
                for chunk_number in range(0, rgbs.shape[0], chunk_size):
                    chunk = rgbs[chunk_number : chunk_number + chunk_size]
                    chunk_d = shift[chunk_number : chunk_number + chunk_size]
                    rgbs[chunk_number : chunk_number + chunk_size] = shift_transient_grid_sample_3d(chunk, chunk_d, args.exposure_time, args.n_bins)
                    
            rgbs = rgbs.reshape(rgbs.shape[0], -1)

        # Volume rendering using native cuda scan.
        weights, _, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=num_rays,
            prefix_trans=1 - opacity[ray_indices].squeeze(-1),
        )
        if alpha_thre > 0:
            vis_mask = alphas >= alpha_thre
            ray_indices, rgbs, weights, t_starts, t_ends = (
                ray_indices[vis_mask],
                rgbs[vis_mask],
                weights[vis_mask],
                t_starts[vis_mask],
                t_ends[vis_mask],
            )

        accumulate_along_rays_(
            weights,
            values=rgbs,
            ray_indices=ray_indices,
            outputs=rgb,
        )
        accumulate_along_rays_(
            weights,
            values=None,
            ray_indices=ray_indices,
            outputs=opacity,
        )
        accumulate_along_rays_(
            weights,
            values=(t_starts + t_ends)[..., None] / 2.0,
            ray_indices=ray_indices,
            outputs=depth,
        )
        # Update near_planes using termination planes.
        near_planes = termination_planes
        # Update rays status.
        ray_mask = torch.logical_and(
            # Early stopping.
            opacity.view(-1) <= opc_thre,
            # Remove rays that have reached the far plane.
            packed_info[:, 1] == n_samples,
        )
        total_samples += ray_indices.shape[0]

    depth = depth / opacity.clamp_min(torch.finfo(rgbs.dtype).eps)
    
    # Unwapring different methods -- values hardocded.
    if unwarp:
        if args.unwarp_method == "depth":
            dists = depth
        elif args.unwarp_method == "sphere":
            dists = intersect_sphere(rays.origins.cpu(), rays.viewdirs.cpu(), np.sqrt(2)).to(depth.device)
        elif args.unwarp_method == "cube":
            dists = intersect_cube(rays.origins.cpu(), rays.viewdirs.cpu(), 1).to(depth.device)

        chunk_size = test_chunk_size
        for chunk_number in range(0, rgb.shape[0], chunk_size):
            chunk = rgb[chunk_number : chunk_number + chunk_size]
            chunk_d = dists[chunk_number : chunk_number + chunk_size]
            rgb[chunk_number : chunk_number + chunk_size] = depth_unwarp(chunk.reshape(chunk.shape[0], args.n_bins, -1), chunk_d, args.exposure_time, args.n_bins).reshape(chunk.shape[0], -1)
    

    return (
        rgb.view((*rays_shape[:-1], -1)),
        opacity.view((*rays_shape[:-1], -1)),
        depth.view((*rays_shape[:-1], -1)),
        total_samples,
    )


def intersect_sphere(rays_o, rays_d, R):
    # Intersection of rays with sphere, distances.
    odotd = torch.sum(rays_o*rays_d, 1)
    d_norm_sq = torch.sum(rays_d**2, 1)
    o_norm_sq = torch.sum(rays_o**2, 1)
    determinant = odotd**2+(R-o_norm_sq)*d_norm_sq
    valid = determinant>=0
    distances = torch.zeros(rays_o.shape[0])
    distances[valid] = (torch.sqrt(determinant[valid])-odotd[valid])/d_norm_sq[valid]
    return distances[:, None]

def intersect_cube(rays_o, rays_d, R):
    # Intersection of rays with cube, distances.
    t1 = (R - rays_o)/rays_d
    t2 = (-R - rays_o)/rays_d
    combined_mins = torch.concat([t1, t2], dim=-1)
    dists = torch.topk(combined_mins, 3, dim=1)[0][:, 2][:, None]
    return dists 


def parse_list(arg):
    try:
        return ast.literal_eval(arg)
    except (SyntaxError, ValueError):
        raise configargparse.ArgumentTypeError(f"Invalid list format: {arg}")

def depth_unwarp(transient, depth, exposure_time, n_bins):
    # Depth based unwapring.
    x_dim = transient.shape[0]
    bins_move = depth/exposure_time
    if x_dim%2 == 0:
        x = (torch.arange(x_dim, device=transient.device)-x_dim//2+0.5)/(x_dim//2-0.5)
    else:
        x = (torch.arange(x_dim, device=transient.device)-x_dim//2)/(x_dim//2)

    z = torch.arange(n_bins, device=transient.device).float()
    X, Z = torch.meshgrid(x, z, indexing="ij")
    Z = Z + bins_move
    Z[Z<0] = n_bins+1
    Z = (Z-n_bins//2+0.5)/(n_bins//2-0.5)
    grid = torch.stack((Z, X), dim=-1)[None, ...]
    shifted_transient = torch.nn.functional.grid_sample(transient.permute(2, 0, 1)[None], grid, align_corners=True).squeeze(0).permute(1, 2, 0)
    return shifted_transient

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def load_args(eval = False, parser= None):
    if not eval:
        parser = configargparse.ArgumentParser()
    parser.add('-c', '--my-config', 
        is_config_file=True, 
        default="./configs_release_train/captured/coke.txt", 
        help='Path to config file.'
    )
    parser.add_argument(
        '--exp_name', 
        type=str, 
        default='coke', 
        help='Experiment name.'
    )
    parser.add_argument(
        "--prop_delay",
        type= str2bool,
        default="true",
        help="Model propagation delay.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/anagh/PycharmProjects/multiview_lif/data/regular-cornell/diverse_cams/training_files",
        help="The root of the dataset directory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers.",
    )
    parser.add_argument(
        "--final",
        type= str2bool,
        default="false",
        help="Train final version or not.",
    )
    parser.add_argument(
        "--streaming",
        type= str2bool,
        default="true",
        help="Stream with dataloader.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="simulated",
        choices=["captured", "simulated"],
        help="Dataset being trained, captured or simulated.",
    )
    parser.add_argument(
        "--outpath",
                type=str,
                default="/home/anagh/PycharmProjects/multiview_lif/results",
        help="Path to results folder.",
    )
    parser.add_argument(
        "--n_bins",
                type=int,
                default=700,
        help="Number of bins.",
    )
    parser.add_argument(
        "--exposure_time",
                type=float,
                default=0.01,
        help="Exposure time per bin.",
    )
    parser.add_argument(
        "--steps_till_checkpoint",
                type=int,
                default=5000,
        help="Number of steps till checkpoint saved.",
    )
    parser.add_argument(
        "--summary_freq",
                type=int,
                default=5000,
        help="Number of steps till summary logged.",
    )
    parser.add_argument(
        "--max_steps",
                type=int,
                default=200000,
        help="Total number of steps trained for.",
    )
    parser.add_argument(
        "--alpha_thre",
                type=float,
                default=0,
        help="Alpha threshold for skipping empty space.",
    )
    parser.add_argument(
        "--lr",
                type=float,
                default=0.005,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
                type=float,
                default=1e-6,
        help="Weight decay value for optimizer.",
    )
    parser.add_argument(
        "--render_step_size",
                type=float,
                default=5e-3,
        help="Step size in rendering.",
    )
    parser.add_argument(
        "--cone_angle",
                type=float,
                default=0,
        help="Cone angle for linearly increased step size.",
    )
    parser.add_argument(
        "--target_sample_batch_size",
                type=int,
                default=2**16,
        help="Number of points per batch.",
    )
    parser.add_argument(
        "--near_plane",
                type=float,
                default=0,
        help="Near plane for sampling.",
    )
    parser.add_argument(
        "--gamma",
                type=int,
                default=2,
        help="Near plane for sampling.",
    )
    parser.add_argument(
        "--far_plane",
                type=int,
                default=2**15,
        help="Far plane for sampling.",
    )
    parser.add_argument(
        "--dataset_scale",
                type=int,
                default=46,
                # default=(5.5*10**6),
        help="Scale for all transient images.",
    )
    parser.add_argument(
        "--t_min",
                type=int,
                default=200,
        help="Time to start bins at.",
    )
    parser.add_argument(
        "--t_max",
                type=int,
                default=800,
        help="Time to end bins at.",
    )
    parser.add_argument(
        "--init_batch_size",
                type=int,
                default=32,
        help="Starting batch size.",
    )
    parser.add_argument(
        "--seed",
                type=int,
                default=42,
        help="Starting batch size.",
    )
    parser.add_argument(
        "--occ_thre",
                type=float,
                default=0.01,
        help="Occupancy threshold",
    )
    parser.add_argument(
        "--thold_warmup",
                type=int,
                default=-1,
        help="Warmup period for the occupancy threshold.",
    )
    parser.add_argument(
        "--grid_resolution",
                type=int,
                default=128,
        help="Starting batch size.",
    )
    parser.add_argument(
        "--grid_nlvl",
                type=int,
                default=1,
        help="Starting batch size.",
    )
    parser.add_argument(
        "--img_shape",
                type=int,
                default=64,
        help="Image shape.",
    )
    parser.add_argument(
        "--summary",
        type= str2bool,
        default="true",
        help="Print tensorboard summary.",
    )
    parser.add_argument(
        "--img_shape_test",
                type=int,
                default=64,
        help="Image shape for summary.",
    )
    parser.add_argument(
        "--aabb",
        nargs='+',
        type = lambda s: ast.literal_eval(s),
        default="[-1.5,-1.5,-1.5,1.5,1.5, 1.5]",
        help="AABB size.",
    )
    parser.add_argument(
        "--laser_pos",
        nargs='+',
        type = lambda s: ast.literal_eval(s),
        default=[(16, 16), (20, 16), (28, 25)],
        help="Laser position.",
    )
    parser.add_argument(
        "--pixels_to_plot",
        nargs='+',
        type = lambda s: ast.literal_eval(s),
        default=[(16, 16), (20, 16), (28, 25)],
        help="Pixels used for plotting in the summary.",
    )
    parser.add_argument(
        "--img_scale",
                type=int,
                default=100,
        help="Image scale used in summary.",
    )
    parser.add_argument(
        "--device",
                type=str,
                default="cuda:4",
        help="Cuda device.",
    )
    # args, _ = parser.parse_known_args()
    args = parser.parse_args()
    assert args.n_bins == args.t_max - args.t_min
    return args

def make_save_folder(args):
    now = datetime.now()
    now = now.strftime("%m-%d_%H:%M:%S")
    exp_name = args.exp_name + "_" + now
    outpath = os.path.join(args.outpath, exp_name)
    os.mkdir(outpath)
    shutil.copy(args.my_config, os.path.join(outpath, "params.txt"))
    
    with open(os.path.join(outpath, "params_full.txt"), "w") as out_file:
        param_list = []    
        for key, value in vars(args).items():
            if type(value) == list:
                value = [eval(f"{x}") for x in value]
            elif type(value) != int and type(value) != float:
                value = str(value)
                value = f"'{value}'"
            param_list.append("%s= %s" % (key, value))
         
        out_file.write('\n'.join(param_list))
    return outpath

def make_save_folder_final(args, optimizer, scheduler, radiance_field, occupancy_grid):
    outpath = os.path.join(args.outpath, args.exp_name)

    if not os.path.isdir(outpath):

        os.mkdir(outpath)
        shutil.copy(args.my_config, os.path.join(outpath, "params.txt"))

        
        with open(os.path.join(outpath, "params_full.txt"), "w") as out_file:
            param_list = []    
            for key, value in vars(args).items():
                if type(value) != int and type(value) != float:
                    value = str(value)
                    value = f"'{value}'"
                param_list.append("%s= %s" % (key, value))
            
            out_file.write('\n'.join(param_list))
        step = 0
        writer = SummaryWriter(log_dir=outpath)

    else:
        ckpt_path_var = os.path.join(outpath, 'variables.pth')
        ckpt = torch.load(ckpt_path_var)
        step = ckpt['step']

        ckpt_path_rf = os.path.join(outpath, 'radiance_field_%04d.pth' % (step))
        ckpt_path_oc = os.path.join(outpath, 'occupancy_grid_%04d.pth' % (step))
        ckpt_path_opt = os.path.join(outpath, 'optimizer_%04d.pth' % (step))
        ckpt_path_sch = os.path.join(outpath, 'scheduler_%04d.pth' % (step))

        ckpt = torch.load(ckpt_path_rf, map_location=args.device)
        radiance_field.load_state_dict(ckpt)
        radiance_field = radiance_field.to(args.device)

        ckpt = torch.load(ckpt_path_oc, map_location=args.device)
        occupancy_grid.load_state_dict(ckpt)
        occupancy_grid = occupancy_grid.to(args.device)

        ckpt = torch.load(ckpt_path_opt)
        optimizer.load_state_dict(ckpt)

        ckpt = torch.load(ckpt_path_sch)
        scheduler.load_state_dict(ckpt)
        print(f"previous checkpoint loaded; current step: {step}")
        writer = SummaryWriter(log_dir=outpath)
    
    return writer, step, outpath
    
if __name__=="__main__":
    pass