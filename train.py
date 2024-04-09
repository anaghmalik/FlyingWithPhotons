import os
from loaders.streaming_loader import SubjectLoaderIterable
from loaders.regular_loader import SubjectLoader
from misc.summary import write_summary_histogram
import time
import numpy as np
import torch
import tqdm
from radiance_fields.ngp import NGPRadianceField
from torch.utils.tensorboard import SummaryWriter
import os
import torch.multiprocessing as mp
from multiprocessing import Value
from ctypes import c_longlong
from torch.utils.data import DataLoader
from nerfacc.estimators.occ_grid import OccGridEstimator

from utils import (
    make_save_folder,
    make_save_folder_final,
    render_image_with_occgrid,
    set_random_seed,
    load_args
    )

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn') # or 'forkserver'

def run():
    torch.cuda.empty_cache()
    args = load_args()
    device = args.device
    set_random_seed(args.seed)
    aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}

    # Setup training dataset and loader.
    if args.streaming:
        num_rays_value = Value(c_longlong, args.init_batch_size)
        counter = Value(c_longlong, 0)
        
        train_dataset = SubjectLoaderIterable(
        root_fp=args.data_root,
        split="train",
        counter= counter,
        num_rays = num_rays_value,
        args = args,
        **train_dataset_kwargs,
        )
        train_data_loader = DataLoader(train_dataset, batch_size=None, num_workers=args.num_workers)
    else:
        train_dataset = SubjectLoader(
            root_fp=args.data_root,
            split="train",
            num_rays = args.init_batch_size,
            args = args,
            **train_dataset_kwargs,
        )
        train_data_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
    
    test_dataset = SubjectLoader(
        root_fp= args.data_root,
        split="test",
        num_rays=None,
        args = args,
        **test_dataset_kwargs,   
    )

    estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=args.grid_resolution, levels=args.grid_nlvl
    ).to(device)


    # Setup the radiance field we want to train.
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1], radiance_activation=torch.nn.Sigmoid(), use_viewdirs=True, args=args)
    radiance_field = radiance_field.to("cpu")
    radiance_field = radiance_field.to(device)

    optimizer = torch.optim.Adam(
        list(radiance_field.parameters()), lr=args.lr, eps=1e-15, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=100
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    args.max_steps // 2,
                    args.max_steps * 3 // 4,
                    args.max_steps * 9 // 10,
                ],
                gamma=0.33,
            ),
        ]
    )

    # Make save folder.
    if args.final:
        writer, step, outpath = make_save_folder_final(args, optimizer, scheduler, radiance_field, estimator)
        args.outpath = outpath
    else:
        outpath = make_save_folder(args)
        args.outpath = outpath
        writer = SummaryWriter(log_dir=outpath)
        step = 0


    # Training.
    tic = time.time()
    pbar = tqdm.tqdm(total=args.max_steps)

    while step < args.max_steps:
        for data in train_data_loader:

            pbar.update(1)
        
            radiance_field.train()
            estimator.train()
            
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]


            def occ_eval_fn(x):
                density = radiance_field.query_density(x)
                return density * args.render_step_size

            # Warmup occupancy threshold.
            if args.version == "captured":
                if step<args.thold_warmup:
                    occ_thre = args.occ_thre/2
                else:
                    occ_thre = args.occ_thre
            else:
                occ_thre = args.occ_thre
            
            estimator.update_every_n_steps(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=occ_thre,
            )
            

            # Render.
            rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
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
            )
            if n_rendering_samples == 0:
                continue

            if args.target_sample_batch_size > 0:
                # Dynamic batch size for rays to keep sample batch size constant.
                num_rays = len(pixels)
                num_rays = int(
                    num_rays * (args.target_sample_batch_size / float(n_rendering_samples))
                )
                if args.streaming:
                    with num_rays_value.get_lock():
                        num_rays_value.value = num_rays
                else:
                    train_dataset.update_num_rays(num_rays)
                    
            loss = torch.nn.functional.smooth_l1_loss(rgb.reshape(rgb.shape[0], args.n_bins, -1), pixels) 
            optimizer.zero_grad()
            
            # Double grad scaler to avoid nans and rescale loss.
            grad_scaler.scale(grad_scaler.scale(loss)).backward()
            grad_scaler.step(optimizer)
            scheduler.step()
            grad_scaler.update()
            writer.add_scalar('Loss/train', torch.log(loss).detach().cpu().numpy(), step)

            if step % args.summary_freq == 0:
                elapsed_time = time.time() - tic
                mse = torch.nn.functional.mse_loss(rgb.reshape(rgb.shape[0], args.n_bins, -1), pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)

                print(
                    f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                    f"loss={loss:.5f} | transient psnr={psnr:.2f} | "
                    f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                    f"max_depth={depth.max():.3f} | "
                )
                
                if args.summary:
                    write_summary_histogram(radiance_field, estimator, writer, test_dataset, step, args)


            if not step % args.steps_till_checkpoint:
                torch.save(radiance_field.state_dict(), os.path.join(outpath, 'radiance_field_%04d.pth' % (step)))
                torch.save(estimator.state_dict(), os.path.join(outpath, 'occupancy_grid_%04d.pth' % (step)))
                torch.save(optimizer.state_dict(), os.path.join(outpath, 'optimizer_%04d.pth' % (step)))
                torch.save(scheduler.state_dict(), os.path.join(outpath, 'scheduler_%04d.pth' % (step)))
                torch.save({'step': step}, os.path.join(outpath, 'variables.pth'))

            
            if step > args.max_steps:
                break

            step += 1
  

if __name__=="__main__":
    run()