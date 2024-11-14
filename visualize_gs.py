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
import os
import torch

from scene import Scene, GaussianModel, EnvLight
from utils.general_utils import seed_everything
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from torchvision.utils import save_image
from omegaconf import OmegaConf
from pprint import pformat
from gaussian_renderer import get_renderer
EPS = 1e-5

@torch.no_grad()
def separation(scene : Scene, renderFunc, renderArgs, env_map=None):
    scale = scene.resolution_scales[0]
    validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                            {'name': 'train', 'cameras': scene.getTrainCameras()})
    
    # we supppose area with altitude>0.5 is static
    # here z axis is downward so is gaussians.get_xyz[:, 2] < -0.5
    high_mask = gaussians.get_xyz[:, 2] < -0.5
    # import pdb;pdb.set_trace()
    mask = (gaussians.get_scaling_t[:, 0] > args.separate_scaling_t) | high_mask
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            outdir = os.path.join(args.model_path, "separation", config['name'])
            os.makedirs(outdir,exist_ok=True)
            for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                # v = scene.gaussians.get_inst_velocity
                # t_scale = scene.gaussians.get_scaling_t.clamp_max(2)
                # other = [t_scale, v]
                render_pkg = renderFunc(viewpoint, gaussians, *renderArgs, env_map=env_map)
                # render_pkg_static = renderFunc(viewpoint, gaussians, *renderArgs, env_map=env_map, mask=mask)
                normal = render_pkg["normal"]
                print(normal.shape)
                print("range: ", torch.min(normal), torch.max(normal))
                print(render_pkg.keys())
           
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                # image_static = torch.clamp(render_pkg_static["render"], 0.0, 1.0)
                # image_no_bg = torch.clamp(render_pkg["render_nobg"], 0.0, 1.0)
                
                save_image(image, os.path.join(outdir, f"{viewpoint.colmap_id:03d}.png"))
                save_image(normal, os.path.join(outdir, f"{viewpoint.colmap_id:03d}_normal.png"))
                exit(0)
                # save_image(image_static, os.path.join(outdir, f"{viewpoint.colmap_id:03d}_static.png"))
                # save_image(image_no_bg, os.path.join(outdir, f"{viewpoint.colmap_id:03d}_nobg.png"))
                # # print(viewpoint.timestamp)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config_path", type=str, required=True)
    params, _ = parser.parse_known_args()
    
    args = OmegaConf.load(params.config_path)
    args.resolution_scales = args.resolution_scales[:1]
    # print('Configurations:\n {}'.format(pformat(OmegaConf.to_container(args, resolve=True, throw_on_missing=True))))
    # convert to DictConfig
    conf = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**conf)))

    seed_everything(args.seed)

    sep_path = os.path.join(args.model_path, 'point_cloud')
    os.makedirs(sep_path, exist_ok=True)
    
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
    print("Number of Gaussians: ", gaussians.get_xyz.shape[0])   
    save_timestamp = 0.0
    gaussians.save_ply_at_t(os.path.join(args.model_path, "point_cloud", "iteration_{}".format(first_iter), "point_cloud.ply"), save_timestamp)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    render_func, _ = get_renderer(args.render_type)
    separation(scene, render_func, (args, background), env_map=env_map)

    print("Rendering statics and dynamics complete.")
