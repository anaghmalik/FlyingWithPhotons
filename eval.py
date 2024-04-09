import os
from datetime import datetime
import json 
import imageio
from nerfacc import OccGridEstimator
import torch
import matplotlib.pyplot as plt
from misc.dataset_utils import read_h5
from misc.eval_utils import generate_video, calc_psnr, calc_iou, get_rays, load_eval_args
from radiance_fields.ngp import NGPRadianceField
import numpy as np
import lpips
import tqdm
import h5py
from utils import render_image_with_occgrid_test
import statistics
from skimage.metrics import structural_similarity
loss_fn_vgg = lpips.LPIPS(net='vgg')


def eval():
    args = load_eval_args()    
    
    if args.warped:
        outpath = os.path.join(args.checkpoint_dir, f"{args.mode}_warped")
    else:
        outpath = os.path.join(args.checkpoint_dir, f"{args.mode}_unwarped")
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

    
     # training parameters
    device = args.device
    aabb = torch.tensor(args.aabb, dtype=torch.float32, device=args.device)

    with open(args.transforms_path, "r") as fp:
        meta = json.load(fp)
    
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

    estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=args.grid_resolution, levels=args.grid_nlvl
    ).to(device)
    
    
    radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1], radiance_activation=torch.nn.Sigmoid(), use_viewdirs=True, args = args).to(device)
    ckpt_path_rf = os.path.join(args.checkpoint_dir, 'radiance_field_%04d.pth' % (args.step))
    ckpt_path_oc = os.path.join(args.checkpoint_dir, 'occupancy_grid_%04d.pth' % (args.step))


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

    with torch.no_grad():
        for i in tqdm.tqdm(range(len(meta["frames"]))):
            frame = meta["frames"][i]
            fname = frame["filepath"][:-2]+"h5"
            c2w = frame["transform_matrix"]
            c2w = torch.tensor(c2w).to(device)
            rays = get_rays(args.img_shape, c2w, K, device)
            
            if args.mode == "quantitative":
                path = os.path.join("/".join(args.transforms_path.split("/")[:-1]), fname)
                gt_tran = read_h5(path)[...,args.t_min_test:args.t_max_test, :3]
                gt_tran = np.clip(gt_tran/args.dataset_scale, 0, 1)
                
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
            
            rgb = rgb.reshape(rays.origins.shape[0], rays.origins.shape[1], -1, color_channels).cpu().numpy()
            rgb = (np.clip(rgb, 0, 1))**args.gamma
            rgb = rgb[..., args.t_min_test:args.t_max_test, :]

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
                imageio.imwrite(os.path.join(outpath, f"{fname.split('.')[0]}_ours.png"), (our_img * 255).astype(np.uint8))
                np.save(os.path.join(outpath, f"{fname.split('.')[0]}_depth.npy"), depth.cpu().numpy())
                
                images = []
                
                for xx in range(rgb.shape[-2]):
                    tran_img_ours = np.clip(rgb[:, :, xx], 0, 1)**(1/2.2)
                    frame_ours = our_img*args.video_alpha +tran_img_ours*(1-args.video_alpha)
                    images.append((255*((frame_ours))).astype(np.uint8))
                    
                savepath = os.path.join(outpath, f"{fname.split('.')[0]}_video.mp4")
                generate_video(images, savepath, 20)
                
                if args.save_images:
                    savepath = os.path.join(outpath, f"{fname.split('.')[0]}_tran.h5")
                    file = h5py.File(savepath, 'w')
                    dataset = file.create_dataset("data", rgb.shape, dtype='f', data=rgb)
                    file.close()


            if args.mode == "quantitative":
                if i == 0:
                    image_psnrs = []
                    transient_psnrs = []
                    transient_l1 = []
                    image_lpips = []
                    image_ssims = []
                    transient_ious = []
                
                gt_img = np.clip((gt_tran.sum(-2)/args.img_scale_test), 0, 1)**(1/2.2)
                our_img = np.clip(rgb.sum(-2)/args.img_scale_test, 0, 1)**(1/2.2)

                tran_psnr = calc_psnr(gt_tran, rgb)
                img_psnr = calc_psnr(gt_img, our_img)
                l1 = np.abs(gt_tran - rgb).mean()
                lpips_ = loss_fn_vgg(torch.from_numpy(gt_img * 2 - 1).unsqueeze(-1).permute((3, 2, 0, 1)).to(torch.float32), torch.from_numpy(our_img * 2 - 1).unsqueeze(-1).permute((3, 2, 0, 1)).to(torch.float32))
                lpips_ = lpips_.detach().cpu().numpy().flatten()[0]
                ssim_, _ = structural_similarity(gt_img, our_img, full=True, channel_axis=2)
                iou = calc_iou(rgb, gt_tran)

                image_psnrs.append(img_psnr)
                transient_psnrs.append(tran_psnr)
                transient_l1.append(l1)
                image_lpips.append(lpips_)
                image_ssims.append(ssim_)
                transient_ious.append(iou)
                
                # save image and ground truth 
                imageio.imwrite(os.path.join(outpath, f"{fname.split('.')[0]}_ours.png"), (our_img * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(outpath, f"{fname.split('.')[0]}_gt.png"), (gt_img * 255).astype(np.uint8))
                
                # save plot of transient
                figure = plt.figure(figsize=((len(args.rendering_pixels)+1), 4), dpi=250)
                plt.subplot(2, (len(args.rendering_pixels)+1)//2, 1)
                plt.imshow(our_img)
                for xx, pixel in enumerate(args.rendering_pixels):
                    plt.plot(pixel[1], pixel[0], '.', markersize=10, color='red')
                    plt.text(pixel[1], pixel[0], str(xx), color="yellow", fontsize=10)

                for ii, pixel in enumerate(args.rendering_pixels):
                    plt.subplot(2, (len(args.rendering_pixels)+1)//2, ii+2)
                    plt.plot(np.arange(rgb.shape[-2]), rgb[pixel[0], pixel[1], :, 0], label='ours', linewidth=0.5)
                    plt.plot(np.arange(rgb.shape[-2]), gt_tran[pixel[0], pixel[1], :, 0], label='gt', linewidth=0.5)
                    plt.legend(borderpad=0, labelspacing=0)
                    plt.gca().set_aspect(1.0 / plt.gca().get_data_ratio(), adjustable='box')
                plt.tight_layout()
                plt.savefig(os.path.join(outpath, f"{fname.split('.')[0]}_plot.svg"),format='svg')

                # save video                     
                images = []
                for x in range(rgb.shape[-2]):
                    tran_img_gt = np.clip(gt_tran[:, :, x], 0, 1)**(1/2.2)
                    tran_img_ours = np.clip(rgb[:, :, x], 0, 1)**(1/2.2)
                    
                    frame_ours = our_img*args.video_alpha +tran_img_ours*(1-args.video_alpha)
                    frame_gt = gt_img*args.video_alpha +tran_img_gt*(1-args.video_alpha)
                    frame = np.concatenate([frame_ours, frame_gt], axis=1)
                    images.append((255*((frame))).astype(np.uint8))
                    
                savepath = os.path.join(outpath, f"{fname.split('.')[0]}_video.mp4")
                generate_video(images, savepath, 20)
                
                if args.save_images:
                    savepath = os.path.join(outpath, f"{fname.split('.')[0]}_tran.h5")
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
              
    elif args.mode == "quantitative":
        image_psnrs.append(statistics.mean(image_psnrs))
        transient_psnrs.append(statistics.mean(transient_psnrs))
        image_lpips.append(statistics.mean(image_lpips))
        image_ssims.append(statistics.mean(image_ssims))
        transient_l1.append(statistics.mean(transient_l1))
        transient_ious.append(statistics.mean(transient_ious))
        
        data = np.array([image_psnrs, transient_psnrs, image_lpips, image_ssims, transient_l1, transient_ious])
        # Specify the file path
        file_path = os.path.join(outpath, "results.txt")
        # Save the data to a text file
        np.savetxt(file_path, data, fmt='%.10f', delimiter='\t')


if __name__ == "__main__":
    eval()