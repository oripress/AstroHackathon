import torch.nn as nn
import torch
import argparse

from nets import Generator, Discriminator, Infer, weights_init
from utils import GalaxySet, display_noise
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm

import os


def to_var(x, device):
    return Variable(x, requires_grad=False).to(device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = Generator(args.nz, args.nc, args.ngf)
    gen = gen.to(device)
    gen.apply(weights_init)

    discriminator = Discriminator(args.nc, args.ndf)
    discriminator = discriminator.to(device)
    discriminator.apply(weights_init)

    infer = Infer(args.nc, args.ndf)
    infer = infer.to(device)
    infer.apply(weights_init)

    bce = nn.BCELoss()
    bce = bce.to(device)

    mse = nn.MSELoss()
    mse = mse.to(device)

    galaxy_dataset = GalaxySet(args.data_path, normalized=args.normalized)
    loader = DataLoader(galaxy_dataset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    loader_iter = iter(loader)

    d_optimizer = Adam(discriminator.parameters(), betas=(0.5, 0.999), lr=args.lr)
    g_optimizer = Adam(gen.parameters(), betas=(0.5, 0.999), lr=args.lr)
    i_optimizer = Adam(infer.parameters(), betas=(0.5, 0.999), lr=args.lr)

    real_labels = to_var(torch.ones(args.bs), device)
    fake_labels = to_var(torch.zeros(args.bs), device)
    fixed_noise = to_var(torch.randn(1, args.nz, 1), device)

    for i in tqdm(range(args.iters)):
        try:
            batch_data = loader_iter.next()
        except StopIteration:
            loader_iter = iter(loader)
            batch_data = loader_iter.next()

        batch_data = to_var(batch_data, device).unsqueeze(1)

        ### Train Discriminator ###

        d_optimizer.zero_grad()

        # train Infer with real
        pred_real = discriminator(batch_data)
        d_loss = bce(pred_real, real_labels)

        # train infer with fakes
        z = to_var(torch.randn((args.bs, args.nz, 1)), device)
        fakes = gen(z)
        pred_fake = discriminator(fakes.detach())
        d_loss += bce(pred_fake, fake_labels)

        d_loss.backward()

        d_optimizer.step()

        ### Train Gen ###

        g_optimizer.zero_grad()

        z = to_var(torch.randn((args.bs, args.nz, 1)), device)
        fakes = gen(z)
        pred_fake = discriminator(fakes)
        gen_loss = bce(pred_fake, real_labels)

        gen_loss.backward()
        g_optimizer.step()

        if i % 100 == 0:
            print("Iteration %d >> g_loss: %.4f., d_loss: %.4f." % (i, gen_loss, d_loss))
            torch.save(gen.state_dict(), os.path.join(args.out, 'gen_%d.pkl' % i))
            torch.save(discriminator.state_dict(), os.path.join(args.out, 'disc_%d.pkl' % i))
            fixed_fake = gen(fixed_noise)
            display_noise(fixed_fake.squeeze(), os.path.join(args.out, "gen_sample_%d.png" % i))

    # for i in tqdm(range(args.iters)):
    #     ### Train Infer ###
    #
    #     i_optimizer.zero_grads()
    #
    #     z = to_var(torch.randn((args.bs, 1, args.nz)), device)
    #     fakes = gen(z)
    #     infer_fakes = infer(fakes)
    #     infer_loss = bce(infer_fakes, z.detach())
    #
    #     infer_loss.backward()
    #     i_optimizer.step()
    #
    #     if i % 100 == 0:
    #         print("Iteration %d >> infer_loss: %.4f" % (i, infer_loss))
    #         torch.save(infer.state_dict(), os.path.join(args.out, 'infer_%d.pkl' % i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='out')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--iters', type=int, default=1250000)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--normalized', type=bool, default=False)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    train(args)
