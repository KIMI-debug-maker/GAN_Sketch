# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

from random import random
import jt
import jt.nn as nn
import jt.nn.functional as F


class DiffAugment(nn.Module):
    def __init__(self, policy='', channels_first=True):
        super(DiffAugment, self).__init__()
        self.policy = policy
        self.channels_first = channels_first

    def forward(self, x):
        if self.policy:
            if not self.channels_first:
                x = x.permute(0, 3, 1, 2)
            for p in self.policy.split(','):
                for f in AUGMENT_FNS[p]:
                    x = f(x)
            if not self.channels_first:
                x = x.permute(0, 2, 3, 1)
            x = x.contiguous()
        return x


def rand_brightness(x):
    x = x + (jt.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (jt.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (jt.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = jt.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = jt.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = jt.meshgrid(
        jt.arange(x.size(0), dtype=jt.long, device=x.device),
        jt.arange(x.size(2), dtype=jt.long, device=x.device),
        jt.arange(x.size(3), dtype=jt.long, device=x.device),
    )
    grid_x = jt.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = jt.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = jt.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = jt.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = jt.meshgrid(
        jt.arange(x.size(0), dtype=jt.long, device=x.device),
        jt.arange(cutout_size[0], dtype=jt.long, device=x.device),
        jt.arange(cutout_size[1], dtype=jt.long, device=x.device),
    )
    grid_x = jt.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = jt.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = jt.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def rand_upscale(x, ratio=0.25):
    up_ratio = 1.0 + ratio * random()
    sz = x.size(2)
    x = jt.nn.functional.interpolate(x, scale_factor=up_ratio, mode='bilinear')
    return center_crop(x, sz)


def center_crop(x, sz):
    h, w = x.size(2), x.size(3)
    x1 = int(round((h - sz) / 2.))
    y1 = int(round((w - sz) / 2.))
    return x[:, :, x1:x1+sz, y1:y1+sz]


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'upscale': [rand_upscale],
}
