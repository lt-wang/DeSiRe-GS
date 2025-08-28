#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import glob
import json
import os
import torch
import torch.nn.functional as F
from utils.loss_utils import psnr, ssim
from gaussian_renderer import get_renderer
from scene import Scene, GaussianModel, EnvLight
from utils.general_utils import seed_everything, visualize_depth
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
from omegaconf import OmegaConf
from pprint import pprint, pformat
from omegaconf import OmegaConf
from texttable import Texttable
import cv2
import numpy as np
import warnings


warnings.filterwarnings("ignore")
EPS = 1e-5
non_zero_mean = (
    lambda x: sum(x) / len(x) if len(x) > 0 else -1
)


@torch.no_grad()
def render_sets(iteration, scene : Scene, renderFunc, renderArgs, env_map=None):
    from lpipsPyTorch import lpips
    

   
    scale = scene.resolution_scales[0]
    if "kitti" in args.model_path:
        # follow NSG: https://github.com/princeton-computational-imaging/neural-scene-graphs/blob/8d3d9ce9064ded8231a1374c3866f004a4a281f8/data_loader/load_kitti.py#L766
        num = len(scene.getTrainCameras())//2
        eval_train_frame = num//5
        traincamera = sorted(scene.getTrainCameras(), key =lambda x: x.colmap_id)
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                            {'name': 'train', 'cameras': traincamera[:num][-eval_train_frame:]+traincamera[num:][-eval_train_frame:]})
    else:
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                        {'name': 'train', 'cameras': scene.getTrainCameras()})
    
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            outdir = os.path.join(args.model_path, "render_result", "render_on_" + config['name'] + "data" + "_render")
            image_folder = os.path.join(args.model_path, "images")

            normal_path = os.path.join(args.model_path, "render_result", "normal")
            depth_path = os.path.join(args.model_path, "render_result", "depth")
            depth_normal_path = os.path.join(args.model_path, "render_result", "depth_normal")
            img_path = os.path.join(args.model_path, "render_result", "image")

            os.makedirs(outdir,exist_ok=True)
            os.makedirs(image_folder,exist_ok=True)


            os.makedirs(normal_path, exist_ok=True)
            os.makedirs(depth_path, exist_ok=True)
            os.makedirs(depth_normal_path, exist_ok=True)
            os.makedirs(img_path, exist_ok=True)

            # opaciity_mask = scene.gaussians.get_opacity[:, 0] > 0.01
            # print("number of valid gaussians: {:d}".format(opaciity_mask.sum().item()))
            
            ##保存env -> latlong
            print("[INFO] env -> latlong  !!!")
            env_map.save_latlong(os.path.join(args.model_path, 'sky_latlong.png'))
            print("[INFO] start render。 !!!")
            for camera_id, viewpoint in enumerate(tqdm(config['cameras'], bar_format="{l_bar}{bar:50}{r_bar}")):
                # render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, env_map=env_map, mask=opaciity_mask)
                _, _, render_pkg = renderFunc(args, viewpoint, gaussians, background, scene.time_interval, env_map, iteration, camera_id)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                #cv2.imwrite(os.path.join(image_folder, f"{viewpoint.colmap_id:03d}_gt.png"), (gt_image[[2,1,0], :, :].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                #cv2.imwrite(os.path.join(image_folder, f"{viewpoint.colmap_id:03d}_render.png"), (image[[2,1,0], :, :].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(img_path, f"{viewpoint.colmap_id:03d}_render.png"), (image[[2,1,0], :, :].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                
                depth = render_pkg['depth']
                alpha = render_pkg['alpha']
                sky_depth = 900
                depth = depth / alpha.clamp_min(EPS)
                if env_map is not None:
                    if args.depth_blend_mode == 0:  # harmonic mean
                        depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
                    elif args.depth_blend_mode == 1:
                        depth = alpha * depth + (1 - alpha) * sky_depth
                sky_mask = viewpoint.sky_mask.to("cuda")
                # dynamic_mask = viewpoint.dynamic_mask.to("cuda") if viewpoint.dynamic_mask is not None else torch.zeros_like(alpha, dtype=torch.bool)            
                dynamic_mask = render_pkg['dynamic_mask']
                depth = visualize_depth(depth)
                alpha = alpha.repeat(3, 1, 1)

                rendered_normal = torch.clamp(render_pkg.get("normal", torch.zeros_like(image)), 0.0, 1.0)
                gt_normal = viewpoint.normal_map.cuda() if viewpoint.normal_map is not None else torch.zeros_like(rendered_normal)
                pseudo_normal = torch.clamp(render_pkg.get("depth_normal", gt_normal), 0.0, 1.0)
                #cv2.imwrite(os.path.join(normal_path, f"{viewpoint.colmap_id:03d}_normal.png"), (rendered_normal[[2,1,0], :, :].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                #cv2.imwrite(os.path.join(normal_path, f"{viewpoint.colmap_id:03d}_normal.png"), (rendered_normal[[2,1,0], :, :].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                save_image(rendered_normal, os.path.join(normal_path, f"{viewpoint.colmap_id:03d}_normal.png"))
                save_image(depth, os.path.join(depth_path, f"{viewpoint.colmap_id:03d}_depth.png"))
                save_image(pseudo_normal, os.path.join(depth_normal_path, f"{viewpoint.colmap_id:03d}_depth_normal.png"))
                grid = [image, alpha, depth, gt_image, torch.logical_not(sky_mask[:1]).float().repeat(3, 1, 1), dynamic_mask.float().repeat(3, 1, 1)]
                grid = make_grid(grid, nrow=3)

                # grid2 = make_grid([
                #     image, 
                #     alpha.repeat(3, 1, 1),
                #     visualize_depth(depth),
                #     rendered_normal,
                #     pseudo_normal,
                #     viewpoint.normal_map.cuda()
                # ], nrow=6)
                save_image(grid, os.path.join(outdir, f"{viewpoint.colmap_id:03d}.png"))
                cv2.imwrite(os.path.join(img_path, f"{viewpoint.colmap_id:03d}_render.png"), (image[[2,1,0], :, :].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

                # get the binary dynamic mask
                if dynamic_mask.sum() > 0:
                    dynamic_mask = dynamic_mask.repeat(3, 1, 1) > 0 # (C, H, W)
                    #masked_psnr_test.append(psnr(image[dynamic_mask], gt_image[dynamic_mask]).double().item())
                    unaveraged_ssim = ssim(image, gt_image, size_average=False) # (C, H, W)
                    #masked_ssim_test.append(unaveraged_ssim[dynamic_mask].mean().double().item())
                    




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config_path", type=str, required=True)
    params, _ = parser.parse_known_args()
    
    args = OmegaConf.load(params.config_path)
    args.resolution_scales = args.resolution_scales[:1]
    print('Configurations:\n {}'.format(pformat(OmegaConf.to_container(args, resolve=True, throw_on_missing=True))))
    
    seed_everything(args.seed)

    # sep_path = os.path.join(args.model_path, 'separation')
    # os.makedirs(sep_path, exist_ok=True)
    
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    
    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None



    checkpoints = glob.glob(os.path.join(args.model_path, "chkpnt*.pth"))
    assert len(checkpoints) > 0, "No checkpoints found."
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    print(f"Loading checkpoint {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, args)
    
    if env_map is not None:
        env_checkpoint = os.path.join(os.path.dirname(checkpoint), 
                                    os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
        (light_params, _) = torch.load(env_checkpoint)
        env_map.restore(light_params)
        uncertainty_model_path = os.path.join(os.path.dirname(checkpoint), 
                                    os.path.basename(checkpoint).replace("chkpnt", "uncertainty_model"))
        state_dict = torch.load(uncertainty_model_path)
        gaussians.uncertainty_model.load_state_dict(state_dict, strict=False)
            
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    render_func, render_wrapper = get_renderer(args.render_type)
    render_sets(first_iter, scene, render_wrapper, (args, background), env_map=env_map)

    print("render_sets complete.")
