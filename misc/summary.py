import imgviz
import numpy as np
from utils import render_image_with_occgrid_test
import matplotlib.pyplot as plt
import torchvision
import torch
import os 

@torch.no_grad()
def write_summary_histogram(radiance_field, estimator, writer, test_dataset, step, args):
    img_scale = args.img_scale
    radiance_field.eval()
    estimator.eval()
    rgb_images = []
    depth_images = [] 
    gt_imgs = []
    accs = []

    pixels_to_plot = args.pixels_to_plot
    plotting_transients = []
    plotting_transients_depth = []

    plotting_transients_gt = []

    test_list = list(range(len(test_dataset)))
    if args.version == "simulated":
        color_channels = 3
    else:
        color_channels = 1
    n_output_dim = args.n_bins*(color_channels)
    with torch.no_grad():

        # sample transients from network
        for ind, i in enumerate(test_list):
            data = test_dataset[i]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

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
                render_bkgd=render_bkgd,
                cone_angle=args.cone_angle,
                alpha_thre=args.alpha_thre,
                n_output_dim=n_output_dim, 
                prop_delay=args.prop_delay
            )
            
            rgb = rgb.reshape(rays.origins.shape[0], rays.origins.shape[1], -1, color_channels)

            torch.save(rgb, os.path.join(args.outpath, f"test_{ind}_conv.pt"))
            torch.save(depth, os.path.join(args.outpath, f"test_{ind}_depth.pt"))

            if color_channels ==1:
                gt_imgs.append(torch.clip(pixels.sum(-2).cpu().repeat(1, 1, 3).permute(2, 0, 1)/img_scale, 0, 1)**(1/2.2))
                rgb_images.append(torch.clip(rgb.sum(-2).cpu().repeat(1, 1, 3).permute(2, 0, 1)/img_scale, 0, 1)**(1/2.2))
            else:
                gt_imgs.append(torch.clip(pixels.sum(-2).cpu().permute(2, 0, 1)/img_scale, 0, 1)**(1/2.2))
                rgb_images.append(torch.clip(rgb.sum(-2).cpu().permute(2, 0, 1)/img_scale, 0, 1)**(1/2.2))
            accs.append(acc.repeat(1, 1, 3).permute(2, 0, 1).cpu())
            dp = imgviz.depth2rgb(depth.cpu().squeeze().numpy(), colormap="inferno")
            depth_images.append(torch.from_numpy(dp).permute(2, 0, 1))

            if ind == 0:
                for pixel in pixels_to_plot:
                    plotting_transients.append(rgb[pixel[0], pixel[1], :, 0])
                    plotting_transients_gt.append(pixels[pixel[0], pixel[1], :, 0])
                    plotting_transients_depth.append(depth[pixel[0], pixel[1]])


        images = torchvision.utils.make_grid(torch.stack(gt_imgs + rgb_images + depth_images + accs), nrow=len(test_list), normalize=False)
        mse = torch.mean((torch.stack(gt_imgs, dim=0) - torch.stack(rgb_images, dim=0))**2, (1,2,3))
        psnr = -10.0 * torch.log(mse) / np.log(10.0)
        print(f"image psnr: {psnr.mean():.2f}\n")
        writer.add_image('rgbdn', images, step)

        figure = plt.figure(figsize=((len(pixels_to_plot)+1), 4), dpi=250)

        # plot the predicted intensity
        plt.subplot(2, (len(pixels_to_plot)+1)//2, 1)
        plt.imshow(gt_imgs[0].permute(1, 2, 0))
        for i, pixel in enumerate(pixels_to_plot):
            plt.plot(pixel[1], pixel[0], '.', markersize=10, color='red')
            plt.text(pixel[1], pixel[0], str(i), color="yellow", fontsize=10)
        plt.gca().set_aspect(1.0/plt.gca().get_data_ratio(), adjustable='box')
        plt.title('gt intensity')

        for i, pixel in enumerate(pixels_to_plot):
            # plot transients
            plt.subplot(2, (len(pixels_to_plot)+1)//2, i+2)
            plt.plot(np.arange(args.n_bins), plotting_transients[i].detach().cpu(), label='pred', linewidth=0.5)
            plt.plot(np.arange(args.n_bins), plotting_transients_gt[i].detach().cpu(), label='gt', linewidth=0.5)
            plt.axvline(x = (plotting_transients_depth[i]/args.exposure_time).detach().cpu().numpy(), color = 'y')
            plt.title(f"pixel {i}")
            plt.ylabel('intensity')
            plt.legend(borderpad=0, labelspacing=0)
            plt.gca().set_aspect(1.0 / plt.gca().get_data_ratio(), adjustable='box')
        
        plt.tight_layout()
        writer.add_figure("transient_plots", figure, step)


    radiance_field.train()
    estimator.train()
    

