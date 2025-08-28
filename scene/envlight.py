import torch
import nvdiffrast.torch as dr
from utils.env_utils import cubemap_to_latlong,cube_to_dir
import imageio
import numpy as np
class EnvLight(torch.nn.Module):

    def __init__(self, resolution=1024):
        super().__init__()
        self.resolution = resolution
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        )
        
    def capture(self):
        return (
            self.base,
            self.optimizer.state_dict(),
        )
        
    def restore(self, model_args, training_args=None):
        self.base, opt_dict = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)
            
    def training_setup(self, training_args):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=training_args.envmap_lr, eps=1e-15)
        
    def forward(self, l):
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape)
        l = l.contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])

        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        light = light.view(*prefix, -1)

        return light

    def save_latlong(self, path):
        # save LDR cubemap now
        sky_latlong = cubemap_to_latlong(self.base, [self.resolution, self.resolution * 2])
        sky_latlong = (sky_latlong.clamp(0., 1.).detach().cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(path, sky_latlong)
        
    # def get_sun_direction(self):
    #     envmap = self.base.detach()  # [6, res, res, 3]
    #     res = envmap.shape[1]

    #     # 1. 线性亮度
    #     env_lin = envmap ** 2.2
    #     Y = (0.2126 * env_lin[...,0] +
    #          0.7152 * env_lin[...,1] +
    #          0.0722 * env_lin[...,2])   # [6,res,res]

    #     # 2. 构造像素中心坐标网格 [-1,1]
    #     t = torch.linspace(0, 1, res, device=envmap.device)
    #     u = 2.0*(t + 0.5/res) - 1.0  # [res]
    #     v = 2.0*(t + 0.5/res) - 1.0  # [res]
    #     uu, vv = torch.meshgrid(u, v, indexing="xy")  # [res,res]

    #     # 3. 每个 face 的方向向量
    #     dirs = []
    #     for face in range(6):
    #         d = cube_to_dir(face, uu, vv)   # [res,res,3]
    #         d = d / torch.norm(d, dim=-1, keepdim=True)
    #         dirs.append(d)
    #     dirs = torch.stack(dirs, dim=0)  # [6,res,res,3]

    #     # 4. 立体角权重
    #     Δ = 2.0 / res
    #     domega = (Δ*Δ) / (1 + uu**2 + vv**2)**1.5   # [res,res]

    #     domega = domega[None,...]  # [1,res,res]
    #     W = Y * domega  # [6,res,res]

    #     # 5. 加权质心
    #     m = (dirs * W[...,None]).sum(dim=(0,1,2))  # [3]
    #     if W.sum() > 0:
    #         sun_dir = m / torch.norm(m)
    #     else:
    #         # 保证 fallback 分支返回 float32，与渲染管线一致
    #         sun_dir = torch.tensor([0.0, 1.0, 0.0], device=envmap.device, dtype=envmap.dtype)
    #     return sun_dir
    def get_sun_direction(self, threshold=0.9):
        """
        从 cubemap 估计太阳方向
        threshold: 相对最大亮度的阈值，比如 0.9 表示只取最亮 10% 区域
        return: (3,) torch tensor, 单位向量
        """
        H, W = self.resolution, self.resolution  # 假设 env map 正方形
        device = self.base.device  # 或 'cuda'

        # 生成 [-1,1] 网格
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # shape [H, W]

        dirs_list = []
        for face in range(6):
            dirs_face = cube_to_dir(face, grid_x, grid_y)  # [H, W, 3]
            dirs_list.append(dirs_face)

        dirs = torch.stack(dirs_list, dim=0)  # [6, H, W, 3]

        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

        # 亮度
        luminance = 0.2126 * self.base[..., 0] + \
                    0.7152 * self.base[..., 1] + \
                    0.0722 * self.base[..., 2]

        # 阈值筛选高亮区域
        max_val = luminance.max()
        mask = (luminance > max_val * threshold).float()

        # 加权平均方向
        weights = luminance * mask
        sun_dir = (dirs * weights[..., None]).sum(dim=(0, 1, 2))
        sun_dir = sun_dir / torch.norm(sun_dir)

        return sun_dir

