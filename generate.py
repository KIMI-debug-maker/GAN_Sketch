import os
import argparse

import random
import numpy as np
import jittor as jt
from torchvision import utils
import torch
from training.networks.stylegan2 import Generator


def save_image_pyjt(img, name):
    """Helper function to save jt tensor into an image file."""
    utils.save_image(
        img,
        name,
        nrow=1,
        padding=0,
        normalize=True,
        range=(-1, 1),
    )


def generate(args, netG, device, mean_latent):
    """Generates images from a generator."""
    # loading a w-latent direction shift if given
    if args.w_shift is not None:
        w_shift = jt.from_numpy(np.load(args.w_shift)).to(device)
        w_shift = w_shift[None, :]
        mean_latent = mean_latent + w_shift
    else:
        w_shift = jt.float32(0.)

    ind = 0
    with jt.no_grad():
        netG.eval()

        # Generate images from a file of input noises
        if args.fixed_z is not None:
            sample_z = jt.load(args.fixed_z, map_location=device) + w_shift
            for start in range(0, sample_z.size(0), args.batch_size):
                end = min(start + args.batch_size, sample_z.size(0))
                z_batch = sample_z[start:end]
                sample, _ = netG([z_batch], truncation=args.truncation, truncation_latent=mean_latent)
                for s in sample:
                    save_image_pyjt(s, f'{args.save_dir}/{str(ind).zfill(6)}.png')
                    ind += 1
            return

        # Generate image by sampling input noises
        for start in range(0, args.samples, args.batch_size):
            end = min(start + args.batch_size, args.samples)
            batch_sz = end - start
            sample_z = jt.randn(batch_sz, 512) + w_shift

            sample, _ = netG([sample_z], truncation=args.truncation, truncation_latent=mean_latent)
            with jt.no_grad():
                for s in sample:
                    save_image_pyjt(torch.Tensor(s.numpy()), f'{args.save_dir}/{str(ind).zfill(6)}.png')
                    ind += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='./output', help="place to save the output")
    parser.add_argument('--ckpt', type=str, default='pretrained/stylegan2-church/netG.pth', help="checkpoint file for the generator")
    parser.add_argument('--size', type=int, default=256, help="output size of the generator")
    parser.add_argument('--fixed_z', type=str, default=None, help="expect a .pth file. If given, will use this file as the input noise for the output")
    parser.add_argument('--w_shift', type=str, default=None, help="expect a .pth file. Apply a w-latent shift to the generator")
    parser.add_argument('--batch_size', type=int, default=10, help="batch size used to generate outputs")
    parser.add_argument('--samples', type=int, default=10, help="number of samples to generate, will be overridden if --fixed_z is given")
    parser.add_argument('--truncation', type=float, default=0.5, help="strength of truncation")
    parser.add_argument('--truncation_mean', type=int, default=4096, help="number of samples to calculate the mean latent for truncation")
    parser.add_argument('--seed', type=int, default=None, help="if specified, use a fixed random seed")
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    device = args.device
    # use a fixed seed if given
    if args.seed is not None:
        random.seed(args.seed)
        jt.manual_seed(args.seed)
        jt.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    netG = Generator(args.size, 512, 8)#.to(device)
    checkpoint = jt.load(args.ckpt)

    netG.load_state_dict(checkpoint)

    # get mean latent if truncation is applied
    if args.truncation < 1:
        with jt.no_grad():
            mean_latent = netG.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, netG, device, mean_latent)