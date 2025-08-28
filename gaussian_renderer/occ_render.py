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

"""
PVG渲染器模块 - 用于DeSiRe-GS（动态场景重建）的高斯散点渲染
包含渲染函数、损失计算和多视角几何一致性约束

+++ GI-GS

"""

import torch
import math
from diff_gauss import (
    GaussianRasterizationSettings,  # 高斯栅格化设置
    GaussianRasterizer,            # 高斯栅格化器
)

from scene.gaussian_model import GaussianModel     # 高斯散点模型
from scene.dynamic_model import scale_grads        # 动态模型梯度缩放
from scene.cameras import Camera                   # 相机模型
from utils.sh_utils import eval_sh                 # 球谐函数评估
import torch.nn.functional as F
import numpy as np
import kornia                                      # 计算机视觉库
from utils.loss_utils import psnr, ssim, tv_loss, lncc  # 损失函数
from utils.graphics_utils import patch_offsets, patch_warp, render_normal  # 图形学工具
import cv2
import os
# from render import render_all
# from gs_render_original import render_original_gs
EPS = 1e-5  # 数值稳定性常数


def render_occ(
    viewpoint_camera: Camera,      # 视点相机
    pc: GaussianModel,             # 高斯散点云模型
    pipe,                          # 渲染管线参数
    bg_color: torch.Tensor,        # 背景颜色
    scaling_modifier=1.0,          # 尺度修饰器
    override_color=None,           # 覆盖颜色
    env_map=None,                  # 环境光照贴图
    time_shift=None,               # 时间偏移（用于动态场景）
    other=[],                      # 其他特征
    mask=None,                     # 遮罩
    is_training=False,             # 是否为训练模式
    return_depth_normal=False,     # 是否返回深度法向量
    radius: float = 0.8,
    bias: float = 0.01,
    thick: float = 0.05,
    delta: float = 0.0625,
    step: int = 16,
    start: int = 8
):
    """
    PVG渲染主函数 - 渲染动态高斯散点场景
    
    参数:
        viewpoint_camera: 视点相机对象，包含相机参数和位姿
        pc: 高斯散点云模型，包含点的位置、颜色、尺度等属性
        pipe: 渲染管线配置
        bg_color: 背景颜色张量，必须在GPU上
        scaling_modifier: 缩放修饰器，用于调整高斯散点的尺度
        override_color: 如果提供，将覆盖计算的颜色
        env_map: 环境光照贴图，用于天空等背景渲染
        time_shift: 时间偏移，用于动态场景的时间插值
        other: 其他需要渲染的特征列表
        mask: 点的可见性遮罩
        is_training: 训练模式标志
        return_depth_normal: 是否返回从深度计算的法向量
    
    返回:
        包含渲染结果的字典，包括图像、深度、法向量等
    """
    # 创建零张量用于屏幕空间点坐标，用于pytorch计算2D均值的梯度
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()  # 保留梯度用于后续优化
    except:
        pass

    # 设置栅格化配置参数
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)  # 水平视场角的正切值
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)  # 垂直视场角的正切值

    # 创建高斯栅格化设置对象
    # raster_settings = GaussianRasterizationSettings(
    #     image_height=int(viewpoint_camera.image_height),           # 图像高度
    #     image_width=int(viewpoint_camera.image_width),             # 图像宽度
    #     tanfovx=tanfovx,                                          # 水平视场角正切值
    #     tanfovy=tanfovy,                                          # 垂直视场角正切值
    #     bg=bg_color if env_map is not None else torch.zeros(3, device="cuda"),  # 背景颜色
    #     scale_modifier=scaling_modifier,                           # 尺度修饰器
    #     viewmatrix=viewpoint_camera.world_view_transform,          # 世界到视图变换矩阵
    #     projmatrix=viewpoint_camera.full_proj_transform,           # 完整投影变换矩阵
    #     sh_degree=pc.active_sh_degree,                            # 活跃的球谐函数度数
    #     campos=viewpoint_camera.camera_center,                     # 相机中心位置
    #     prefiltered=False,                                        # 是否预过滤
    #     debug=pipe.debug,                                         # 调试模式
    # )
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        radius=radius,
        bias=bias,
        thick=thick,
        delta=delta,
        step=step,
        start=start,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        inference=False,
        argmax_depth=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)  # 创建栅格化器

    # 初始化高斯散点的基本属性
    means3D = pc.get_xyz          # 3D中心点位置
    means2D = screenspace_points  # 2D屏幕空间点位置
    opacity = pc.get_opacity      # 不透明度
    scales = None                 # 尺度参数
    rotations = None              # 旋转参数
    cov3D_precomp = None         # 预计算的3D协方差矩阵

    # 处理动态场景的时间偏移
    if time_shift is not None:
        # 根据时间偏移获取新的3D位置
        means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp - time_shift)
        means3D = means3D + pc.get_inst_velocity * time_shift  # 加上瞬时速度的影响
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp - time_shift)
    else:
        # 直接获取当前时间戳的位置
        means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp)
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
    
    opacity = opacity * marginal_t  # 根据边际时间调整不透明度

    # 根据配置选择协方差矩阵的计算方式
    if pipe.compute_cov3D_python:
        # 在Python中预计算3D协方差矩阵
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # 使用尺度和旋转参数，让栅格化器计算协方差
        scales = pc.get_scaling     # 获取尺度参数
        rotations = pc.get_rotation # 获取旋转参数

    # 处理颜色信息 - 如果提供了预计算颜色就使用，否则从球谐函数计算
    shs = None           # 球谐函数系数
    colors_precomp = None # 预计算颜色
    
    if override_color is None:
        if pipe.convert_SHs_python:
            # 在Python中从球谐函数计算颜色
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, pc.get_max_sh_channels
            )
            # 计算从高斯点到相机的方向向量
            dir_pp = (
                means3D.detach()
                - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            ).detach()
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            # 评估球谐函数得到RGB颜色
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # 让栅格化器处理球谐函数到RGB的转换
            shs = pc.get_features
    else:
        # 使用提供的覆盖颜色
        colors_precomp = override_color

    feature_list = other  # 其他特征列表

    # TODO: MOVE TO MASKED MEANS3D
    # 计算法向量 - 从高斯散点的尺度和旋转得到
    # normals = pc.get_normal(viewpoint_camera.c2w, means3D, from_scaling=False)
    normal_global = pc.get_normal_v2(viewpoint_camera, means3D)
    # 将法向量转换到相机坐标系

    normals = normal_global @ viewpoint_camera.world_view_transform[:3, :3]

    # 处理其他特征
    if len(feature_list) > 0:
        features = torch.cat(feature_list, dim=1)  # 拼接特征
        S_other = features.shape[1]                # 其他特征的通道数
    else:
        features = torch.zeros_like(means3D[:, :0])  # 空特征张量
        S_other = 0

    # 预过滤 - 过滤掉不可见或边际时间太小的点
    if mask is None:
        mask = marginal_t[:, 0] > 0.05  # 基于边际时间的默认遮罩
    else:
        mask = mask & (marginal_t[:, 0] > 0.05)  # 与提供的遮罩结合
    # 计算点在相机坐标系中的位置和深度
    pts_in_cam = (
        means3D @ viewpoint_camera.world_view_transform[:3, :3]
        + viewpoint_camera.world_view_transform[3, :3]
    )
    depth_z = pts_in_cam[:, 2:3]  # Z方向深度
    
    # 计算法向量与相机射线的局部距离
    local_distance = (normals * pts_in_cam).sum(-1).abs()
    
    # 创建深度-alpha特征通道
    depth_alpha = torch.zeros(
        means3D.shape[0], 2, dtype=torch.float32, device=means3D.device
    )
    # 对可见点设置深度和alpha值
    depth_alpha[mask] = torch.cat([depth_z, torch.ones_like(depth_z)], dim=1)[mask]
    
    # 拼接所有特征：其他特征 + 深度alpha + 法向量 + 局部距离
    features = torch.cat([features, depth_alpha, normals, local_distance.unsqueeze(-1)], dim=1)

    # 栅格化可见的高斯散点到图像，获得它们在屏幕上的半径
    # contrib, rendered_image, rendered_feature, radii = rasterizer(
    #     means3D=means3D,              # 3D中心点位置
    #     means2D=means2D,              # 2D屏幕空间位置
    #     shs=shs,                      # 球谐函数系数
    #     colors_precomp=colors_precomp, # 预计算颜色
    #     features=features,            # 特征向量
    #     opacities=opacity,            # 不透明度
    #     scales=scales,                # 尺度
    #     rotations=rotations,          # 旋转
    #     cov3D_precomp=cov3D_precomp, # 预计算协方差
    #     mask=mask,                    # 可见性遮罩
    # )

    N = means3D.shape[0]
    albedo = torch.ones(N, 3, device=means3D.device, dtype=means3D.dtype) 
    roughness= torch.ones(N, 1, device=means3D.device, dtype=means3D.dtype)
    metallic = torch.ones(N, 1, device=means3D.device, dtype=means3D.dtype) 

    #预定义太阳方向
    #sun_dir = torch.tensor([0.1, 0.5, 0.4], device=means3D.device, dtype=means3D.dtype)
    sun_dir = env_map.get_sun_direction()
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (
        rendered_image,
        radii,
        opacity_map,
        depth_map,
        normal_map_from_depth,
        normal_map,
        occlusion_map,
        albedo_map,
        roughness_map,
        metallic_map,
        out_normal_view,
        depth_pos,
        sun_shadow_map
    ) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        normal=normal_global,
        shs=shs,
        colors_precomp=colors_precomp,
        albedo=albedo,
        roughness=roughness,
        metallic=metallic,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        derive_normal=True,
        sun_dir=sun_dir
    )

    # 将法向量从[-1, 1]范围映射到[0, 1]用于可视化
    # rendered_normal = F.normalize(rendered_normal, dim=0)
    rendered_normal = normal_map * 0.5 + 0.5  # [-1, 1] -> [0, 1]
    
    rendered_image_before = rendered_image  # 保存环境光照前的图像
    # 如果有环境光照贴图，混合背景颜色
    if env_map is not None:
        # 从环境贴图获取背景颜色
        bg_color_from_envmap = env_map(
            viewpoint_camera.get_world_directions(is_training).permute(1, 2, 0)
        ).permute(2, 0, 1)
        # 根据不透明度混合前景和背景
        rendered_image = rendered_image + (1 - opacity_map) * bg_color_from_envmap

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity_map": opacity_map,
        "depth_map": depth_map,
        "normal_map_from_depth": normal_map_from_depth,
        "normal_map": normal_map,
        "albedo_map": albedo_map,
        "roughness_map": roughness_map,
        "metallic_map": metallic_map,
        "occlusion_map": occlusion_map,
        "sun_shadow_map": sun_shadow_map,
        "out_normal_view": out_normal_view,
        "depth_pos": depth_pos
    }


def calculate_loss(
    gaussians: GaussianModel,      # 高斯散点模型
    viewpoint_camera: Camera,      # 视点相机
    bg_color: torch.Tensor,        # 背景颜色
    args,                          # 训练参数配置
    render_pkg: dict,              # 渲染结果包
    env_map,                       # 环境光照贴图
    iteration,                     # 当前迭代次数
    camera_id,                     # 相机ID
    nearest_cam: Camera = None,    # 最近的相机（用于多视角一致性）
):
    """
    计算训练损失函数
    
    包含多种损失项：
    - L1损失和SSIM损失（图像重建）
    - LiDAR深度损失
    - 法向量一致性损失
    - 动态正则化损失
    - 多视角几何一致性损失
    - 各种正则化项
    
    参数:
        gaussians: 高斯散点模型
        viewpoint_camera: 当前视点相机
        bg_color: 背景颜色
        args: 包含各种损失权重和配置的参数对象
        render_pkg: 渲染结果字典
        env_map: 环境光照贴图
        iteration: 当前训练迭代次数
        camera_id: 相机标识符
        nearest_cam: 用于多视角一致性的最近相机
    
    返回:
        loss: 总损失值
        log_dict: 记录各项损失的字典
        extra_render_pkg: 额外的渲染信息
    """
    log_dict = {}  # 用于记录各项损失的字典

    # 从渲染结果中提取各种图
    image = render_pkg["render"]          # 渲染图像
    depth = render_pkg["depth"]           # 深度图
    alpha = render_pkg["alpha"]           # 不透明度图
    visibility_filter = render_pkg["visibility_filter"]  # 可见性过滤器
    feature = render_pkg["feature"] / alpha.clamp_min(EPS)  # 特征图（归一化）
    t_map = feature[0:1]                  # 时间图
    v_map = feature[1:]                   # 速度图

    # 获取天空遮罩，用于处理天空区域
    sky_mask = (
        viewpoint_camera.sky_mask.cuda()
        if viewpoint_camera.sky_mask is not None
        else torch.zeros_like(alpha, dtype=torch.bool)
    )

    sky_depth = 900  # 天空的默认深度值
    
    # 计算实际深度值（除以alpha进行归一化）
    depth = depth / alpha.clamp_min(EPS)
    
    # 如果有环境光照贴图，需要混合天空深度
    if env_map is not None:
        if args.depth_blend_mode == 0:  # 调和平均混合模式
            depth = 1 / (
                alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth
            ).clamp_min(EPS)
        elif args.depth_blend_mode == 1:  # 线性混合模式
            depth = alpha * depth + (1 - alpha) * sky_depth

    # 获取真实图像（RGB和灰度版本）
    # gt_image = viewpoint_camera.original_image.cuda()
    gt_image, gt_image_gray = viewpoint_camera.get_image()

    # 计算基础重建损失
    loss_l1 = F.l1_loss(image, gt_image, reduction="none")  # L1损失 [3, H, W]
    loss_ssim = 1.0 - ssim(image, gt_image, size_average=False)  # SSIM损失 [3, H, W]

    # 记录损失到日志
    log_dict["loss_l1"] = loss_l1.mean().item()
    log_dict["loss_ssim"] = loss_ssim.mean().item()
    # 初始化不确定性相关变量
    uncertainty_loss = 0
    metrics = {}
    loss_mult = torch.ones_like(depth, dtype=depth.dtype)  # 损失乘数
    dino_part = torch.zeros_like(depth, dtype=depth.dtype)  # DINO特征部分
    
    # 如果有不确定性模型，计算不确定性损失
    if gaussians.uncertainty_model is not None:
        del loss_mult  # 删除默认的损失乘数

        # 从不确定性模型获取损失和权重
        uncertainty_loss, metrics, loss_mult, dino_part = (
            gaussians.uncertainty_model.get_loss(
                gt_image, image.detach(), sky_mask, _cache_entry=("train", camera_id)
            )
        )
        # 将损失乘数转换为二值化权重
        loss_mult = (loss_mult > 1 - EPS).to(dtype=loss_mult.dtype)  # [1, H, W]

        # 在第一阶段处理不确定性
        if args.uncertainty_stage == "stage1":
            # 不确定性预热阶段的处理
            if iteration < args.uncertainty_warmup_start:
                # 预热开始前，使用均匀权重
                loss_mult = torch.ones_like(depth, dtype=depth.dtype)
            elif (
                iteration
                < args.uncertainty_warmup_start + args.uncertainty_warmup_iters
            ):
                # 预热期间，逐渐引入不确定性权重
                p = (
                    iteration - args.uncertainty_warmup_start
                ) / args.uncertainty_warmup_iters
                loss_mult = 1 + p * (loss_mult - 1)
                
            # 可选的权重中心化处理
            if args.uncertainty_center_mult:  # default: False
                loss_mult = loss_mult.sub(loss_mult.mean() - 1).clamp(0, 2)
                
            # 可选的梯度缩放处理
            if args.uncertainty_scale_grad:  # default: False
                image = scale_grads(image, loss_mult)
                loss_mult = torch.ones_like(depth, dtype=depth.dtype)

    # 根据训练阶段计算总损失
    if args.uncertainty_stage == "stage1":
        # 第一阶段：在不透明度重置后的保护期内分离不确定性损失
        last_densify_iter = min(iteration, args.densify_until_iter - 1)
        last_dentify_iter = (
            last_densify_iter // args.opacity_reset_interval
        ) * args.opacity_reset_interval
        
        # 在保护期内分离不确定性损失的梯度
        if iteration < last_dentify_iter + args.uncertainty_protected_iters:
            # 保持在图像空间中最大半径的记录用于剪枝
            try:
                uncertainty_loss = uncertainty_loss.detach()  # type: ignore
            except AttributeError:
                pass

        # 计算加权的总损失
        loss = (
            (1.0 - args.lambda_dssim) * (loss_l1 * loss_mult).mean()    # 加权L1损失
            + args.lambda_dssim * (loss_ssim * loss_mult).mean()        # 加权SSIM损失
            + uncertainty_loss                                           # 不确定性损失
        )
    else:
        # 其他阶段：使用标准损失
        loss = (
            1.0 - args.lambda_dssim
        ) * loss_l1.mean() + args.lambda_dssim * loss_ssim.mean()

    # 计算PSNR用于日志记录
    psnr_for_log = psnr(image, gt_image).double()
    log_dict["psnr"] = psnr_for_log
    alpha_mask = alpha.data > EPS  # 分离的alpha遮罩 (1, H, W)

    # LiDAR深度损失
    if args.lambda_lidar > 0:
        assert viewpoint_camera.pts_depth is not None
        pts_depth = viewpoint_camera.pts_depth.cuda()

        # 找到有效的LiDAR点
        mask = pts_depth > 0
        # 计算逆深度差异（更稳定的深度比较方式）
        loss_lidar = torch.abs(
            1 / (pts_depth[mask] + 1e-5) - 1 / (depth[mask] + 1e-5)
        ).mean()
        
        # 可选的迭代衰减
        if args.lidar_decay > 0:
            iter_decay = np.exp(-iteration / 8000 * args.lidar_decay)
        else:
            iter_decay = 1
            
        log_dict["loss_lidar"] = loss_lidar.item()
        loss += iter_decay * args.lambda_lidar * loss_lidar

    # 法向量一致性损失
    if args.lambda_normal > 0:
        rendered_normal = render_pkg["normal"]  # 渲染的法向量 (3, H, W)
        
        if args.load_normal_map:
            # 使用加载的法向量贴图作为监督
            gt_normal = viewpoint_camera.normal_map.cuda()
            loss_normal = F.l1_loss(rendered_normal * alpha_mask, gt_normal * alpha_mask)
            loss_normal += tv_loss(rendered_normal)  # 添加总变差正则化
            log_dict["loss_normal"] = loss_normal.item()
            loss += args.lambda_normal * loss_normal
        elif "depth_normal" in render_pkg:
            # 使用从深度计算的法向量作为监督
            depth_normal = render_pkg["depth_normal"]
            loss_normal = F.l1_loss(
                rendered_normal * alpha_mask, depth_normal * alpha_mask
            )
            loss_normal += tv_loss(rendered_normal)  # 添加总变差正则化
            log_dict["loss_normal"] = loss_normal.item()
            loss += 0.1 * args.lambda_normal * loss_normal

    # 时间正则化损失（动态场景）
    if args.lambda_t_reg > 0 and args.enable_dynamic:
        if (
            gaussians.uncertainty_model is not None
            and iteration > args.dynamic_mask_epoch
        ):
            # 在动态掩码时期后，使用不确定性权重
            loss_t_reg = -torch.abs(t_map * loss_mult).mean() * 10.0
        # else:
        #     loss_t_reg = -torch.abs(t_map).mean()
            log_dict['loss_t_reg'] = loss_t_reg.item()
            loss += args.lambda_t_reg * loss_t_reg

    # 速度正则化损失（动态场景）
    if args.lambda_v_reg > 0 and args.enable_dynamic:
        if (
            gaussians.uncertainty_model is not None
            and iteration > args.dynamic_mask_epoch
        ):
            # 在动态掩码时期后，使用不确定性权重惩罚速度
            loss_v_reg = (torch.abs(v_map) * loss_mult).mean() * 10.0
        else:
            # 直接惩罚速度幅度
            loss_v_reg = (torch.abs(v_map)).mean()
        log_dict["loss_v_reg"] = loss_v_reg.item()
        loss += args.lambda_v_reg * loss_v_reg

    # 逆深度平滑损失
    if args.lambda_inv_depth > 0:
        inverse_depth = 1 / (depth + 1e-5)
        # 使用kornia的逆深度平滑损失，促进深度一致性
        loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(
            inverse_depth[None], gt_image[None]
        )
        log_dict["loss_inv_depth"] = loss_inv_depth.item()
        loss = loss + args.lambda_inv_depth * loss_inv_depth

    # 速度平滑损失（动态场景）
    if args.lambda_v_smooth > 0 and args.enable_dynamic:
        # 对速度图施加平滑约束
        loss_v_smooth = kornia.losses.inverse_depth_smoothness_loss(
            v_map[None], gt_image[None]
        )
        log_dict["loss_v_smooth"] = loss_v_smooth.item()
        loss = loss + args.lambda_v_smooth * loss_v_smooth

    # 时间平滑损失（动态场景）
    if args.lambda_t_smooth > 0 and args.enable_dynamic:
        # 对时间图施加平滑约束
        loss_t_smooth = kornia.losses.inverse_depth_smoothness_loss(
            t_map[None], gt_image[None]
        )
        log_dict["loss_t_smooth"] = loss_t_smooth.item()
        loss = loss + args.lambda_t_smooth * loss_t_smooth

    # 天空不透明度损失
    if args.lambda_sky_opa > 0:
        o = alpha.clamp(1e-6, 1 - 1e-6)  # 限制不透明度范围避免log(0)
        sky = sky_mask.float()
        # 鼓励天空区域的不透明度接近0
        loss_sky_opa = (-sky * torch.log(1 - o)).mean()
        log_dict["loss_sky_opa"] = loss_sky_opa.item()
        loss = loss + args.lambda_sky_opa * loss_sky_opa

    # 不透明度熵损失
    if args.lambda_opacity_entropy > 0:
        o = alpha.clamp(1e-6, 1 - 1e-6)  # 限制不透明度范围避免log(0)
        # 鼓励不透明度的确定性（接近0或1）
        loss_opacity_entropy = -(o * torch.log(o)).mean()
        log_dict["loss_opacity_entropy"] = loss_opacity_entropy.item()
        loss = loss + args.lambda_opacity_entropy * loss_opacity_entropy

    # 尺度正则化损失
    if args.lambda_scaling > 0:
        scaling = render_pkg["scaling"]  # 高斯散点的尺度参数 (N, 3)
        # 惩罚尺度的各向异性（促进各向同性的高斯分布）
        scaling_loss = (
            (scaling - scaling.mean(dim=-1, keepdim=True)).abs().sum(-1).mean()
        )
        lambda_scaling = (
            args.lambda_scaling * 0.1
        )  # - 0.99 * args.lambda_scaling * min(1, 4 * iteration / args.iterations)
        log_dict["scaling_loss"] = scaling_loss.item()
        loss = loss + lambda_scaling * scaling_loss
    extra_render_pkg = {}  # 额外的渲染信息包
    
    # 尺度约束损失
    if visibility_filter.sum() > 0:
        scale = gaussians.get_scaling[visibility_filter] # 可见高斯点的尺度 (N, 3)
        sorted_scale, _ = torch.sort(scale, dim=-1) # 排序后的尺度 (N, 3)
        
        # 最小尺度约束
        if args.lambda_min_scale > 0:
            # 惩罚过小的尺度，确保高斯点不会退化
            min_scale_loss = torch.relu(sorted_scale[..., 0] - args.min_scale) + torch.relu(args.min_scale - sorted_scale[..., 0]) ** 2 # (N)
            log_dict["min_scale"] = min_scale_loss.mean().item()
            loss += args.lambda_min_scale * min_scale_loss.mean()
        
        # 最大尺度约束
        if args.lambda_max_scale > 0:
            # 惩罚过大的尺度，防止高斯点过度膨胀
            max_scale_loss = torch.relu(sorted_scale[..., -1] - args.max_scale)
            valid_mask = max_scale_loss > 0
            log_dict["max_scale"] = max_scale_loss[valid_mask].mean().item()
            loss += args.lambda_max_scale * max_scale_loss[valid_mask].mean()

    # 多视角一致性约束（从指定迭代开始）
    if iteration >= args.multi_view_weight_from_iter:
        use_virtul_cam = False  # 是否使用虚拟相机

        if nearest_cam is not None:
            # 多视角几何一致性参数
            patch_size = args.multi_view_patch_size        # 补丁大小
            sample_num = args.multi_view_sample_num        # 采样数量
            pixel_noise_th = args.multi_view_pixel_noise_th # 像素噪声阈值
            total_patch_size = (patch_size * 2 + 1) ** 2   # 总补丁大小
            ncc_weight = args.multi_view_ncc_weight        # NCC权重
            geo_weight = args.multi_view_geo_weight        # 几何权重
            
            ## 计算几何一致性遮罩和损失
            H, W = render_pkg["depth"].squeeze().shape
            # 创建像素网格
            ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
            pixels = (
                torch.stack([ix, iy], dim=-1).float().to(render_pkg["depth"].device)
            )

            # 渲染最近相机的视图
            nearest_render_pkg = render_pvg(
                nearest_cam,
                gaussians,
                args,
                bg_color,
                env_map=env_map,
                is_training=True,
            )

            # 从深度图反投影得到3D点
            pts = gaussians.get_points_from_depth(viewpoint_camera, render_pkg["depth"])
            
            # 将3D点投影到最近相机坐标系
            pts_in_nearest_cam = (
                pts @ nearest_cam.world_view_transform[:3, :3]
                + nearest_cam.world_view_transform[3, :3]
            )
            
            # 获取最近相机深度图中对应点的深度
            map_z, d_mask = gaussians.get_points_depth_in_depth_map(
                nearest_cam, nearest_render_pkg["depth"], pts_in_nearest_cam
            )

            # 使用深度图中的深度值重新计算3D点位置
            pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:, 2:3])
            pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[..., None]
            
            # 变换回世界坐标系，然后投影到当前相机
            R = torch.tensor(nearest_cam.R).float().cuda()
            T = torch.tensor(nearest_cam.T).float().cuda()
            pts_ = (pts_in_nearest_cam - T) @ R.transpose(-1, -2)
            pts_in_view_cam = (
                pts_ @ viewpoint_camera.world_view_transform[:3, :3]
                + viewpoint_camera.world_view_transform[3, :3]
            )
            # 将3D点投影到当前相机的像素坐标
            pts_projections = torch.stack(
                [
                    pts_in_view_cam[:, 0] * viewpoint_camera.fx / pts_in_view_cam[:, 2]
                    + viewpoint_camera.cx,
                    pts_in_view_cam[:, 1] * viewpoint_camera.fy / pts_in_view_cam[:, 2]
                    + viewpoint_camera.cy,
                ],
                -1,
            ).float()
            
            # 计算重投影误差（像素噪声）
            pixel_noise = torch.norm(
                pts_projections - pixels.reshape(*pts_projections.shape), dim=-1
            )
            
            # 创建静态点遮罩
            static_mask = (loss_mult > EPS).reshape(-1)
            
            # 综合深度遮罩：深度一致性 & 重投影误差小 & 静态点 & 可见点
            d_mask = d_mask & (pixel_noise < pixel_noise_th) &  (static_mask) & (alpha_mask.reshape(-1))
            
            # 计算基于重投影误差的权重
            weights = (1.0 / torch.exp(pixel_noise)).detach()
            weights[~d_mask] = 0
            
            # 保存额外的渲染信息
            extra_render_pkg['rendered_distance'] = render_pkg['rendered_distance']
            extra_render_pkg['d_mask'] = weights.reshape(1, H, W)
            # 计算几何一致性损失
            if d_mask.sum() > 0 and geo_weight > 0:
                geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                loss += geo_loss
                
                # 计算归一化互相关（NCC）损失
                if use_virtul_cam is False and ncc_weight > 0:
                    with torch.no_grad():
                        ## 采样有效点的遮罩
                        d_mask = d_mask.reshape(-1)
                        valid_indices = torch.arange(
                            d_mask.shape[0], device=d_mask.device
                        )[d_mask]
                        
                        # 如果有效点太多，随机采样
                        if d_mask.sum() > sample_num:
                            index = np.random.choice(
                                d_mask.sum().cpu().numpy(), sample_num, replace=False
                            )
                            valid_indices = valid_indices[index]

                        weights = weights.reshape(-1)[valid_indices]
                        
                        ## 采样参考帧补丁
                        pixels = pixels.reshape(-1, 2)[valid_indices]
                        offsets = patch_offsets(patch_size, pixels.device)
                        ori_pixels_patch = (
                            pixels.reshape(-1, 1, 2) / viewpoint_camera.ncc_scale
                            + offsets.float()
                        )

                        H, W = gt_image_gray.squeeze().shape
                        pixels_patch = ori_pixels_patch.clone()
                        pixels_patch[:, :, 0] = (
                            2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                        )
                        pixels_patch[:, :, 1] = (
                            2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                        )
                        ref_gray_val = F.grid_sample(
                            gt_image_gray.unsqueeze(1),
                            pixels_patch.view(1, -1, 1, 2),
                            align_corners=True,
                        )
                        ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                        ref_to_neareast_r = (
                            nearest_cam.world_view_transform[:3, :3].transpose(-1, -2)
                            @ viewpoint_camera.world_view_transform[:3, :3]
                        )
                        ref_to_neareast_t = (
                            -ref_to_neareast_r
                            @ viewpoint_camera.world_view_transform[3, :3]
                            + nearest_cam.world_view_transform[3, :3]
                        )

                    ## compute Homography
                    ref_local_n = render_pkg["normal"].permute(1, 2, 0) # (H, W, 3) [0, 1]
                    ref_local_n = ref_local_n * 2.0 - 1.0
                    ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices]

                    ref_local_d = render_pkg['rendered_distance'].squeeze()

                    ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                    H_ref_to_neareast = (
                        ref_to_neareast_r[None]
                        - torch.matmul(
                            ref_to_neareast_t[None, :, None].expand(
                                ref_local_d.shape[0], 3, 1
                            ),
                            ref_local_n[:, :, None]
                            .expand(ref_local_d.shape[0], 3, 1)
                            .permute(0, 2, 1),
                        )
                        / ref_local_d[..., None, None]
                    )
                    H_ref_to_neareast = torch.matmul(
                        nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(
                            ref_local_d.shape[0], 3, 3
                        ),
                        H_ref_to_neareast,
                    )
                    H_ref_to_neareast = H_ref_to_neareast @ viewpoint_camera.get_inv_k(
                        viewpoint_camera.ncc_scale
                    )

                    ## compute neareast frame patch
                    grid = patch_warp(
                        H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch
                    )
                    grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                    grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                    _, nearest_image_gray = nearest_cam.get_image()
                    sampled_gray_val = F.grid_sample(
                        nearest_image_gray[None],
                        grid.reshape(1, -1, 1, 2),
                        align_corners=True,
                    )
                    sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

                    ## compute loss
                    ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                    mask = ncc_mask.reshape(-1)
                    ncc = ncc.reshape(-1) * weights
                    ncc = ncc[mask].squeeze()

                    if mask.sum() > 0:
                        ncc_loss = ncc_weight * ncc.mean()
                        loss += ncc_loss

    
    extra_render_pkg["t_map"] = t_map
    extra_render_pkg["v_map"] = v_map
    extra_render_pkg["depth"] = depth

    extra_render_pkg["dynamic_mask"] = loss_mult
    extra_render_pkg["dino_cosine"] = dino_part

    log_dict.update(metrics)

    return loss, log_dict, extra_render_pkg


def render_occ_wrapper(
    args,
    viewpoint_camera: Camera,
    gaussians: GaussianModel,
    background: torch.Tensor,
    time_interval: float,
    env_map,
    iterations,
    camera_id,
    nearest_cam: Camera = None,
):

    # render v and t scale map
    v = gaussians.get_inst_velocity
    t_scale = gaussians.get_scaling_t.clamp_max(2)
    other = [t_scale, v]

    if np.random.random() < args.lambda_self_supervision:
        time_shift = 3 * (np.random.random() - 0.5) * time_interval
    else:
        time_shift = None

    render_pkg = render_occ(
        viewpoint_camera,
        gaussians,
        args,
        background,
        env_map=env_map,
        other=other,
        time_shift=time_shift,
        is_training=True,
    )
    # if iterations > args.uncertainty_warmup_start:
    #     # we supppose area with altitude>0.5 is static
    #     # here z axis is downward so is gaussians.get_xyz[:, 2] < -0.5
    #     high_mask = gaussians.get_xyz[:, 2] < -0.5
    #     # import pdb;pdb.set_trace()
    #     static_mask = (gaussians.get_scaling_t[:, 0] > args.separate_scaling_t) | high_mask
    #     static_render_package = render_occ(viewpoint_camera, gaussians, args, background, env_map=env_map, mask=static_mask)
    #     static_image = static_render_package['render']
    #     render_pkg['static_image'] = static_image
    #     render_pkg['static_occlusion'] = static_render_package['occlusion_map']
    #     render_pkg['static_sun_shadow'] = static_render_package['sun_shadow_map']

    # loss, log_dict, extra_render_pkg = calculate_loss(
    #     gaussians, viewpoint_camera, background, args, render_pkg, env_map, iterations, camera_id, nearest_cam=nearest_cam,
    # )

    #render_pkg.update(extra_render_pkg)


    # we supppose area with altitude>0.5 is static
    # here z axis is downward so is gaussians.get_xyz[:, 2] < -0.5
    high_mask = gaussians.get_xyz[:, 2] < -0.5
    # import pdb;pdb.set_trace()
    static_mask = (gaussians.get_scaling_t[:, 0] > args.separate_scaling_t) | high_mask
    static_render_package = render_occ(viewpoint_camera, gaussians, args, background, env_map=env_map, mask=static_mask)
    static_image = static_render_package['render']
    render_pkg['static_image'] = static_image
    render_pkg['static_occlusion'] = static_render_package['occlusion_map']
    render_pkg['static_sun_shadow'] = static_render_package['sun_shadow_map']
    loss = None
    log_dict = {}
    return loss, log_dict, render_pkg
