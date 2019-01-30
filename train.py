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
import numpy as np


def to_var(x, device, grads=False):
    return Variable(x, requires_grad=grads).to(device)


def train(args):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

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

    real_labels = to_var(torch.ones(args.bs), device_str)
    fake_labels = to_var(torch.zeros(args.bs), device_str)
    fixed_noise = to_var(torch.randn(1, args.nz), device_str)

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


def distance_score_from_gan_dist(args):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    gen = Generator(args.nz, 800)
    gen.load_state_dict(torch.load(args.gen_file))
    gen = gen.to(device)
    gen.eval()

    dataset = GalaxySet(args.data_path, args.normalized, args.out)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=2, drop_last=False)
    scores = torch.zeros(len(loader) * args.bs)

    loss_crit = nn.L1Loss().to(device)

    for i, batch in tqdm(enumerate(loader)):
        batch = to_var(batch, device)[:, :1600:2]
        z = torch.randn(batch.size(0), args.nz, device=device_str, requires_grad=True)
        z_optim = Adam([z], lr=args.lr)
        for j in range(args.infer_iter):
            z_optim.zero_grad()
            fakes = gen(z)
            loss = loss_crit(fakes, batch)

            loss.backward()

            z_optim.step()

            if j % 20 == 0:
                print("Iter %d: loss %d" % (j, loss))

        fakes = gen(z)
        batch_scores = torch.sum(torch.abs(fakes - batch), dim=1)
        scores[i * args.bs:  i * args.bs + batch_scores.size(0)] = batch_scores

    return scores


def rank_anamolies(args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # galaxy_dataset = GalaxySet(args.data_path, normalized=args.normalized, out=args.out)
    # loader = DataLoader(galaxy_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=True)
    #
    # for i, batch_data in enumerate(loader):
    #     batch_data = to_var(batch_data, device).unsqueeze(1)
    #
    #     batch_data = batch_data[:, :, :1600:2]
    #     batch_data = batch_data.view(-1, 800)
    #
    #     recon_losses = distance_score_from_gan_dist(batch_data)
    #     anomoly_dict[str(i)] = recon_losses[0]

    scores = distance_score_from_gan_dist(args)
    scores = sorted(list(enumerate(scores)), key= lambda x: x[1], reverse=True)

    np.savetxt("all_scores.csv", scores, delimiter=",")
    np.savetxt("top_100.csv", scores[:100], delimiter=",")

    # anamoly_100 = sorted(anomoly_dict.items(), key=lambda x: -x[1])[:100]
    # anamoly_100 = [x[0] for x in anamoly_100]
    #
    # wall = np.load(args.wall_path)
    # wall_dict = dict()
    #
    # for i in range(wall_dict.shape[0]):
    #     wall_dict[i] = wall[i]
    #
    # wall_100 = sorted(wall_dict.items(), key=lambda x: -x[1])[:100]
    # wall_100 = [x[0] for x in wall_100]
    #
    # correct = 0
    # total = 0
    #
    # for anamoly in anamoly_100:
    #     if anamoly in wall_100:
    #         correct += 1
    #     total += 1
    #
    # print('Percentage of anamolies hit: %.4f', (float(correct) / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='out')
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--iters', type=int, default=1250000)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--wall_path', type=str)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--infer_iter', type=int, default=100)
    parser.add_argument('--gen_file')

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    if args.train:
        train(args)

    if args.infer:
        rank_anamolies(args)

    # if args.infer:
    #     scores = distance_score_from_gan_dist(args)
    #     torch.save(scores, os.path.join(args.out, "infer_scores.pkl"))
