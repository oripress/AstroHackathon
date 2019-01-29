import torch.nn as nn
import torch
import argparse

from nets import Generator, Discriminator, Infer, weights_init
from utils import GalaxySet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam


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

    bce = nn.BCELoss()
    bce = bce.to(device)

    mse = nn.MSELoss()
    mse = mse.to(device)

    galaxy_dataset = GalaxySet(args.data_path, normalized=True)
    loader = DataLoader(galaxy_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    loader_iter = iter(loader)

    d_optimizer = Adam(discriminator.parameters(), betas=(0.5, 0.999), lr=args.lr)
    g_optimizer = Adam(gen.parameters(), betas=(0.5, 0.999), lr=args.lr)

    real_labels = to_var(torch.ones(args.bs), device)
    fake_labels = to_var(torch.zeros(args.bs), device)

    for i in range(args.iters):
        try:
            batch_data = loader_iter.next()
        except StopIteration:
            loader_iter = iter(loader)
            batch_data = loader_iter.next()

        batch_data = to_var(batch_data, device).unsqueeze(1).float()

        ### Train Discriminator ###

        d_optimizer.zero_grad()

        # train Infer with real
        pred_real = discriminator(batch_data)
        discriminator_loss = bce(pred_real, real_labels)

        # train infer with fakes
        z = to_var(torch.randn((args.bs, 1, args.nz)), device)
        fakes = gen(z)
        pred_fake = discriminator(fakes)
        discriminator_loss += bce(pred_fake, fake_labels)

        discriminator_loss.backward()

        d_optimizer.step()

        ### Train Gen ###

        g_optimizer.zero_grads()

        z = to_var(torch.randn((args.bs, 1, args.nz)), device)
        fakes = gen(z)
        pred_fake = discriminator(fakes)
        gen_loss = bce(pred_fake, real_labels)

        gen_loss.backward()

        g_optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='out')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--iters', type=int, default=1250000)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)

    args = parser.parse_args()

    train(args)
