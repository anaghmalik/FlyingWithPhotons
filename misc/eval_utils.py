import numpy as np 
import imageio 
import json 
import torch 
import sys 
import os
import configargparse
import ast
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders.utils import Rays
from utils import str2bool, load_args

def calc_psnr(img1, img2):
    # Calculate the mean squared error
    mse = np.mean((img1 - img2) ** 2)
    # Calculate the maximum possible pixel value (for data scaled between 0 and 1)
    max_pixel = 1.0
    # Calculate the PSNR
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def get_rays(img_shape, c2w, K, device):
    OPENGL_CAMERA = True
    x, y = torch.meshgrid(
                torch.arange(img_shape, device=device),
                torch.arange(img_shape, device=device),
                indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()
    
    c2w = c2w.repeat(img_shape**2, 1, 1)
    camera_dirs = torch.nn.functional.pad(
            torch.stack(
                [
                    (x - K[0, 2] + 0.5) / K[0, 0],
                    (y - K[1, 2] + 0.5)
                    / K[1, 1]
                    * (-1.0 if OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )
    origins = torch.reshape(origins, (img_shape, img_shape, 3))
    viewdirs = torch.reshape(viewdirs, (img_shape, img_shape, 3))

    
    rays = Rays(origins=origins, viewdirs=viewdirs)

    return rays 


def read_json(json_path):
    f = open(json_path)
    positions = json.load(f)
    f.close()
    return positions

def generate_video(images, output_path, fps):
    # Determine the width and height of the images
    writer = imageio.get_writer(output_path, fps=fps)
    for image in images:
        writer.append_data(image)
    writer.close()

def calc_iou(rgb, gt_tran):
    intersection = np.minimum(rgb, gt_tran)
    union = np.maximum(rgb, gt_tran)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def load_eval_args():
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
        default="/",
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
    args = load_args(eval=True, parser=parser)
    return args

if __name__=="__main__":
    pass