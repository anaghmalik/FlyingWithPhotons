import os
from datetime import datetime
import json 
import ast
import imageio
from nerfacc import OccGridEstimator
import torch
from eval_utils import generate_video
from radiance_fields.ngp import NGPRadianceField
import numpy as np
import lpips
import tqdm
import h5py
from utils import load_args, render_image_with_occgrid_test, str2bool
import configargparse
from loaders.utils import Rays
from misc.eval_utils import get_rays

loss_fn_vgg = lpips.LPIPS(net='vgg')

def load_rel_args():
    parser = configargparse.ArgumentParser()
    parser.add('-tc', '--test_config', 
        is_config_file=True, 
        default="./configs_test/peppers_quantitative.txt", 
        help='Test config path.'
    )
    parser.add_argument(
        "--transforms_path",
        type=str,
        default="",
        help="Path to evaluation transforms file.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Directory of the checkpoint.",
    )
    parser.add_argument(
        "--warped",
        type=str2bool,
        default="true",
        help="Render warped or unwarped transients.",
    )
    parser.add_argument(
        "--img_scale_test",
        type=float,
        default=10,
        help="Scale to divide integrated transient by.",
    )
    parser.add_argument(
        "--video_alpha",
        type=float,
        default=0.2,
        help="Alpha value for compositing background over transient for video.",
    )
    parser.add_argument(
        "--starting_ind",
        type=int,
        default=10,
        help="Starting index for video.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quantitative",
        choices=["render", "render_video", "quantitative"],
        help="Which mode to run script in, evaluation, rendering video or rendering transients.",
    )
    parser.add_argument(
        "--save_images",
        type=str2bool,
        default="true",
        help="Whether to save transients.",
    )
    parser.add_argument(
        "--t_min_test",
        type=int,
        default=200,
        help="Render min bin number.",
    )
    parser.add_argument(
        "--t_max_test",
        type=int,
        default=700,
        help="Render max bin number.",
    )
    parser.add_argument(
        "--unwarp_method",
        type=str,
        default="depth",
        help="Method to perform unwarping.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=80000,
        help="Which step to evaluate.",
    )
    parser.add_argument(
        "--num_rot",
        type=int,
        default=2,
        help="Loops of camera, useful when making a video which loops back.",
    )
    parser.add_argument(
        "--rendering_pixels",
        nargs='+',
        type = lambda s: ast.literal_eval(s),
        default=[(16, 16), (20, 16), (28, 25)],
        help="Pixels to render for plotting.",
    )
    parser.add_argument(
        "--dilation",
        type=str2bool,
        default="false",
        help="Whether to apply dilation effect.",
    )
    parser.add_argument(
        "--aberration",
        type=str2bool,
        default="false",
        help="Whether to apply aberration effect.",
    )
    parser.add_argument(
        "--deformation",
        type=str2bool,
        default="false",
        help="Whether to apply deformation effect.",
    )
    parser.add_argument(
        "--searchlight",
        type=str2bool,
        default="false",
        help="Whether to apply searchlight effect.",
    )
    parser.add_argument(
        "--betas",
        nargs='+',
        type = lambda s: ast.literal_eval(s),
        default="[0.1, 0.7, 0.9]",
        help="Betas.",
    )
    args = load_args(eval=True, parser=parser)
    return args


def rel_render():
    args = load_rel_args()
    
    step = args.step
    ckpt_dir = args.checkpoint_dir
    transforms_path = args.transforms_path

    ending = ""
    if args.searchlight: ending += "_searchlight"
    if args.dilation: ending += "_dilation"
    if args.aberration: ending += "_aberration"
    if args.deformation: ending += "_deformation"

    if args.warped:
        outpath = os.path.join(ckpt_dir, f"warped_{ending}")
    else:
        outpath = os.path.join(ckpt_dir, f"unwarped_{ending}")
    
    now = datetime.now()
    now = now.strftime("%m-%d_%H:%M:%S")
    outpath = outpath + "_" + now

    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath, "params.txt"), "w") as out_file:
        param_list = []    
        for key, value in vars(args).items():
            if type(value) != int and type(value) != float:
                value = str(value)
                value = f"'{value}'"
            param_list.append("%s= %s" % (key, value))
        
        out_file.write('\n'.join(param_list))

    
    device = args.device
    aabb = torch.tensor(args.aabb, dtype=torch.float32, device=args.device)
    with open(transforms_path, "r") as fp:
        meta = json.load(fp)
            

    estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=args.grid_resolution, levels=args.grid_nlvl
    ).to(device)
    
    
    radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1], radiance_activation=torch.nn.Sigmoid(), use_viewdirs=True, args = args).to(device)

    ckpt_path_rf = os.path.join(ckpt_dir, 'radiance_field_%04d.pth' % (step))
    ckpt_path_oc = os.path.join(ckpt_dir, 'occupancy_grid_%04d.pth' % (step))


    ckpt = torch.load(ckpt_path_rf, map_location=device)
    radiance_field.load_state_dict(ckpt)
    radiance_field = radiance_field.to(device)

    ckpt = torch.load(ckpt_path_oc, map_location=device)
    estimator.load_state_dict(ckpt)
    estimator = estimator.to(device)
    
    
    if args.version == "simulated":
        color_channels = 3
    else:
        color_channels = 1

    # set betas
    betas = np.array(args.betas)
    
    camtoworlds = []
    for kk in range(len(meta["frames"])):
        frame = meta["frames"][kk]
        c2w = frame["transform_matrix"]
        c2w = torch.tensor(c2w).to(device)
        camtoworlds.append(c2w)
        
    camtoworlds = torch.stack(camtoworlds)
    vel_vectors = -camtoworlds[:, :3, -2]
    
    def relativistic_viewdirs(v, M, beta):
        # M is velocity vector 
        # v is viewdirs vector
        vdotM = (v@M)
        tangent_vect = v - vdotM[:, None]*M[None, :].repeat(v.shape[0], 1)
        tangent_vect /= torch.linalg.norm(tangent_vect, axis=-1)[:, None]
        new_cos = (vdotM + beta)/(1+beta*vdotM)
        vdirs = new_cos[:, None]*M[None, :].repeat(v.shape[0], 1) + ((1-new_cos[:, None]**2)**(1/2))*tangent_vect
        return vdirs, new_cos
        
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(meta["frames"]))):
            beta = betas[i]
            gamma = 1/(1-beta**2)**(1/2)
            
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

            # camera deformation
            if args.deformation:
                if args.version == "simulated":
                    focal = 0.5 * args.img_shape / np.tan(0.5 * camera_angle_x)
                    focal = focal/gamma
                    K = torch.tensor(
                    [
                        [focal, 0, args.img_shape / 2.0],
                        [0, focal, args.img_shape / 2.0],
                        [0, 0, 1],
                    ],
                    dtype=torch.float32,
                    )
                else:
                    K[0, 0] /= gamma
                    K[1, 1] /= gamma
            
            frame = meta["frames"][i]
            fname = frame["filepath"][:-2]+"h5"
            c2w = frame["transform_matrix"]
            c2w = torch.tensor(c2w).to(device)
            rays = get_rays(args.img_shape, c2w, K, device)
            
            # Relativistic aberration
            abberrated_viewdirs, cos_theta = relativistic_viewdirs(rays.viewdirs.reshape(-1, 3), vel_vectors[i], beta)
            if args.aberration:
                rays = Rays(origins=rays.origins, viewdirs=abberrated_viewdirs.reshape(rays.viewdirs.shape))
                
                

            # rendering
            rgb, acc, depth, _ = render_image_with_occgrid_test(
                1024,
                # scene
                radiance_field,
                estimator,
                rays,
                args=args,
                # rendering options
                near_plane=args.near_plane,
                far_plane=args.far_plane,
                render_step_size=args.render_step_size,
                cone_angle=args.cone_angle,
                alpha_thre=args.alpha_thre,
                n_output_dim=args.n_bins*color_channels, 
                prop_delay=args.prop_delay, 
                unwarp = not args.warped
            )
            
            rgb = (torch.clip(rgb, 0, 1))**args.gamma


            # time dilation - tune disp for transient to be in the right time range 
            def time_dilation(transient, depth, exposure_time, n_bins, gamma, cos_theta, disp=1500):
                x_dim = transient.shape[0]
                bins_move = depth/exposure_time
                if x_dim%2 == 0:
                    x = (torch.arange(x_dim, device=transient.device)-x_dim//2+0.5)/(x_dim//2-0.5)
                else:
                    x = (torch.arange(x_dim, device=transient.device)-x_dim//2)/(x_dim//2)

                z = torch.arange(n_bins, device=transient.device).float()
                X, Z = torch.meshgrid(x, z, indexing="ij")
                eta = (1+(1-gamma**2)/(gamma**2)*cos_theta**2)**(1/2)
                Z = Z/gamma + bins_move/gamma - bins_move*eta +disp
                Z[Z<0] = n_bins+1
                Z = (Z-n_bins//2+0.5)/(n_bins//2-0.5)
                grid = torch.stack((Z, X), dim=-1)[None, ...]
                shifted_transient = torch.nn.functional.grid_sample(transient.permute(2, 0, 1)[None], grid, align_corners=True).squeeze(0).permute(1, 2, 0)
                return shifted_transient

            if args.dilation:
                rgb = time_dilation(rgb.reshape(-1, args.n_bins, color_channels), depth.reshape(-1, 1), args.exposure_time, args.n_bins, gamma, cos_theta.reshape(-1, 1))            

            rgb = rgb.reshape(rays.origins.shape[0], rays.origins.shape[1], -1, color_channels).cpu().numpy()

            # searchlight effect 
            if args.searchlight:
                D = (gamma*(1+beta*cos_theta)).cpu().numpy()
                D /= D.max()
                rgb = rgb/D.reshape(rays.origins.shape[0], rays.origins.shape[1], 1, 1)**5
                

            if args.mode == "render_video":
                ind = args.starting_ind
                normalization = 1
                img = rgb.sum(-2)
                img = np.clip(img/args.img_scale_test, 0, 1)**(1/2.2)                

                if i == 0:
                    video = np.zeros(((len(meta["frames"]))*args.num_rot, args.img_shape, args.img_shape, 3), dtype=np.float32)
                    video_bkgd = np.zeros(((len(meta["frames"]))*args.num_rot, args.img_shape, args.img_shape, 3), dtype=np.float32)
                for x in range(args.num_rot):
                    if x%2 == 0:
                        pos = len(meta["frames"])*x + i 
                    else:
                        pos = len(meta["frames"])*(x+1) - i -1 
                    tran_img = np.clip(rgb[:, :, pos+ind]/normalization, 0, 1)**(1/2.2)
                    frame = img*args.video_alpha +tran_img*(1-args.video_alpha)

                    video[pos] =  frame
                    video_bkgd[pos] =  img

            
            if args.mode == "render":
                our_img = np.clip(rgb.sum(-2)/args.img_scale_test, 0, 1)**(1/2.2)
                imageio.imwrite(os.path.join(outpath, f"{fname.split('.')[0]}_{beta}_ours.png"), (our_img * 255).astype(np.uint8))
                np.save(os.path.join(outpath, f"{fname.split('.')[0]}_{beta}_depth.npy"), depth.cpu().numpy())
                
                images = []
                
                for xx in range(rgb.shape[-2]):
                    tran_img_ours = np.clip(rgb[:, :, xx], 0, 1)**(1/2.2)
                    frame_ours = our_img*args.video_alpha +tran_img_ours*(1-args.video_alpha)
                    images.append((255*((frame_ours))).astype(np.uint8))
                    
                savepath = os.path.join(outpath, f"{fname.split('.')[0]}_{beta}_video.mp4")
                generate_video(images, savepath, 20)
                
                if args.save_images:
                    savepath = os.path.join(outpath, f"{fname.split('.')[0]}_{beta}_tran.h5")
                    file = h5py.File(savepath, 'w')
                    dataset = file.create_dataset("data", rgb.shape, dtype='f', data=rgb)
                    file.close()

    if args.mode == "render_video":
        np.save(os.path.join(outpath, "video.npy"), video)
        video = list(video)
        video = [(255*f).astype(np.uint8) for f in video]
        savepath = os.path.join(outpath, "video.mp4")
        generate_video(video, savepath, 10)
        np.save(os.path.join(outpath, "video_bkgd.npy"), video_bkgd)
        video_bkgd = list(video_bkgd)
        video_bkgd = [(255*f).astype(np.uint8) for f in video_bkgd]
        savepath = os.path.join(outpath, "video_bkgd.mp4")
        generate_video(video_bkgd, savepath, 10)


if __name__=="__main__":
    rel_render()