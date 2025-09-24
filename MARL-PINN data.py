import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
import h5py
from skimage import measure  # 添加此行
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

def v_directional(x):
    """v = 1.0 + 0.4*x + 0.6*sin(3πy) + 0.3*z**2"""
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    z_coord = x[:, 2:3]
    return 1.0 + 0.4 * x_coord + 0.6 * torch.sin(3 * np.pi * y_coord) + 0.3 * z_coord ** 2



def v_uniform(x):
    in_cube = (
        (x[:, 0] >= 0.0) & (x[:, 0] <= 1.0) &
        (x[:, 1] >= 0.0) & (x[:, 1] <= 1.0) &
        (x[:, 2] >= 0.0) & (x[:, 2] <= 1.0)
    )
    return torch.where(in_cube.unsqueeze(1), 1.0, 0.0).to(x.device)


def v_layered_3d(x):
    v0 = 1.0
    k = 0.5
    y = x[:, 1:2]
    return v0 + k * y


def v_complex_3d(x):

    v_layered = 1.0 + 0.5 * x[:, 1]


    center = torch.tensor([0.7, 0.3, 0.5], device=x.device)
    r = 0.15
    dist = torch.norm(x - center, dim=1)
    v_anomaly = torch.where(dist <= r, 2.0, 1.0)

    return v_layered * v_anomaly


def v_anomaly_3d(x):
    v_background = 1.0
    v_inside = 2.0
    center = torch.tensor([0.5, 0.5, 0.5], device=x.device)
    r = 0.2
    dist = torch.norm(x - center, dim=1, keepdim=True)
    return torch.where(dist <= r, v_inside, v_background)


def v_anisotropic_3d(x):
    vx, vy, vz = 2.0, 1.0, 1.5
    grad_x = x[:, 0:1]
    grad_y = x[:, 1:2]
    grad_z = x[:, 2:3]
    return 1.0 / torch.sqrt((grad_x / vx) ** 2 + (grad_y / vy) ** 2 + (grad_z / vz) ** 2 + 1e-6)


def v_sinusoidal(x):
    """v = 1.0 + 0.3*(sin(4πx) + sin(4πy) + sin(4πz))"""
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    z_coord = x[:, 2:3]
    return 1.0 + 0.3*(torch.sin(4*np.pi*x_coord) +
                     torch.sin(4*np.pi*y_coord) +
                     torch.sin(4*np.pi*z_coord))


class MarmousiVelocity:
    def __init__(self, device, mat_path, vmin=1500, vmax=5500, data_key='velocity',
                 dims=(64, 64, 32)):
        self.device = device
        with h5py.File(mat_path, 'r') as f:
            velocity_dataset = f[data_key]
            velocity = np.array(velocity_dataset).astype(np.float32)


        if velocity.ndim == 1:

            expected_size = dims[0] * dims[1] * dims[2]
            if velocity.size != expected_size:
                raise ValueError(f"一维数据长度{velocity.size}与给定维度{dims}不匹配（应={expected_size}）")
            velocity = velocity.reshape(dims)  # 重塑为三维数组

        elif velocity.ndim != 3:
            raise ValueError("速度场数据必须为一维展平或三维数组")


        velocity = np.transpose(velocity, (2, 1, 0))


        self.raw_velocity = np.nan_to_num(velocity)
        self.vmin = vmin
        self.vmax = vmax
        self.velocity = (self.raw_velocity - vmin) / (vmax - vmin) * 2.0 + 1.0
        self.values = torch.from_numpy(self.velocity).to(device).float()

    def sample(self, coords):

        x = coords[:, 0] * (self.velocity.shape[2] - 1)
        y = coords[:, 1] * (self.velocity.shape[1] - 1)
        z = coords[:, 2] * (self.velocity.shape[0] - 1)


        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        z0 = torch.floor(z).long()
        x1 = torch.clamp(x0 + 1, 0, self.velocity.shape[2] - 1)
        y1 = torch.clamp(y0 + 1, 0, self.velocity.shape[1] - 1)
        z1 = torch.clamp(z0 + 1, 0, self.velocity.shape[0] - 1)

        dx = x - x0.float()
        dy = y - y0.float()
        dz = z - z0.float()


        c000 = (1 - dx) * (1 - dy) * (1 - dz)
        c001 = (1 - dx) * (1 - dy) * dz
        c010 = (1 - dx) * dy * (1 - dz)
        c011 = (1 - dx) * dy * dz
        c100 = dx * (1 - dy) * (1 - dz)
        c101 = dx * (1 - dy) * dz
        c110 = dx * dy * (1 - dz)
        c111 = dx * dy * dz

        return (
                c000 * self.values[z0, y0, x0] +
                c001 * self.values[z1, y0, x0] +
                c010 * self.values[z0, y1, x0] +
                c011 * self.values[z1, y1, x0] +
                c100 * self.values[z0, y0, x1] +
                c101 * self.values[z1, y0, x1] +
                c110 * self.values[z0, y1, x1] +
                c111 * self.values[z1, y1, x1]
        ).unsqueeze(-1)

    def __call__(self, x):
        return self.sample(x)
