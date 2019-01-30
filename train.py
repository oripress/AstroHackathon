import torch.nn as nn
import torch
import argparse

from nets import Generator, Discriminator, weights_init
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

    gen = Generator(args.nz, 800)
    gen = gen.to(device)
    gen.apply(weights_init)

    discriminator = Discriminator(800)
    discriminator = discriminator.to(device)
    discriminator.apply(weights_init)

    bce = nn.BCELoss()
    bce = bce.to(device)

    galaxy_dataset = GalaxySet(args.data_path, normalized=args.normalized, out=args.out)
    loader = DataLoader(galaxy_dataset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    loader_iter = iter(loader)

    d_optimizer = Adam(discriminator.parameters(), betas=(0.5, 0.999), lr=args.lr)
    g_optimizer = Adam(gen.parameters(), betas=(0.5, 0.999), lr=args.lr)

    real_labels = to_var(torch.ones(args.bs), device)
    fake_labels = to_var(torch.zeros(args.bs), device)
    fixed_noise = to_var(torch.randn(1, args.nz), device)

    for i in tqdm(range(args.iters)):
        try:
            batch_data = loader_iter.next()
        except StopIteration:
            loader_iter = iter(loader)
            batch_data = loader_iter.next()

        batch_data = to_var(batch_data, device).unsqueeze(1)

        batch_data = batch_data[:, :, :1600:2]
        batch_data = batch_data.view(-1, 800)

        ### Train Discriminator ###

        d_optimizer.zero_grad()

        # train Infer with real
        pred_real = discriminator(batch_data)
        d_loss = bce(pred_real, real_labels)

        # train infer with fakes
        z = to_var(torch.randn((args.bs, args.nz)), device)
        fakes = gen(z)
        pred_fake = discriminator(fakes.detach())
        d_loss += bce(pred_fake, fake_labels)

        d_loss.backward()

        d_optimizer.step()

        ### Train Gen ###

        g_optimizer.zero_grad()

        z = to_var(torch.randn((args.bs, args.nz)), device)
        fakes = gen(z)
        pred_fake = discriminator(fakes)
        gen_loss = bce(pred_fake, real_labels)

        gen_loss.backward()
        g_optimizer.step()

        if i % 5000 == 0:
            print("Iteration %d >> g_loss: %.4f., d_loss: %.4f." % (i, gen_loss, d_loss))
            torch.save(gen.state_dict(), os.path.join(args.out, 'gen_%d.pkl' % 0))
            torch.save(discriminator.state_dict(), os.path.join(args.out, 'disc_%d.pkl' % 0))
            gen.eval()
            fixed_fake = gen(fixed_noise).detach().cpu().numpy()
            real_data = batch_data[0].detach().cpu().numpy()
            gen.train()
            display_noise(fixed_fake.squeeze(), os.path.join(args.out, "gen_sample_%d.png" % i))
            display_noise(real_data.squeeze(), os.path.join(args.out, "real_%d.png" % 0))


def rank_anamolies(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    galaxy_dataset = GalaxySet(args.data_path, normalized=args.normalized, out=args.out)
    loader = DataLoader(galaxy_dataset, batch_size=args.bs, shuffle=False, num_workers=2, drop_last=True)
    loader_iter = iter(loader)

    for i in tqdm(range(args.iters)):
        try:
            batch_data = loader_iter.next()
        except StopIteration:
            loader_iter = iter(loader)
            batch_data = loader_iter.next()

        batch_data = to_var(batch_data, device).unsqueeze(1)

        batch_data = batch_data[:, :, :1600:2]
        batch_data = batch_data.view(-1, 800)


def find_best_z():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='out')
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--iters', type=int, default=1250000)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    train(args)
