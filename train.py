import torch.nn as nn
import torchvision
import torch
import argparse

from nets import Generator, Discriminator, weights_init
from utils import GalaxySet, display_images
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm

import torchvision.transforms as transforms

import os
import numpy as np


def to_var(x, device, grads=False):
    return Variable(x, requires_grad=grads).to(device)


def train(args):
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]),
        download=args.dl,
    )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    gen = Generator(args.nz, 784)
    gen = gen.to(device)
    gen.apply(weights_init)

    discriminator = Discriminator(784)
    discriminator = discriminator.to(device)
    discriminator.apply(weights_init)

    bce = nn.BCELoss()
    bce = bce.to(device)

    loader = DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)

    d_optimizer = Adam(discriminator.parameters(), betas=(0.5, 0.999), lr=args.lr)
    g_optimizer = Adam(gen.parameters(), betas=(0.5, 0.999), lr=args.lr)

    fixed_noise = to_var(torch.randn(64, args.nz), device_str)

    total_iters = 0
    while total_iters < args.iters:
        for i, batch in enumerate(loader):
            batch_data, labels = batch

            batch_data = batch_data[labels != args.digit]
            batch_data = to_var(batch_data, device)

            real_labels = to_var(torch.ones(batch_data.size(0)), device_str)
            fake_labels = to_var(torch.zeros(batch_data.size(0)), device_str)

            batch_data = batch_data.view(-1, 784)

            ### Train Discriminator ###

            d_optimizer.zero_grad()

            # train Infer with real
            pred_real = discriminator(batch_data)
            d_loss = bce(pred_real, real_labels)

            # train infer with fakes
            z = to_var(torch.randn((batch_data.size(0), args.nz)), device)
            fakes = gen(z)
            pred_fake = discriminator(fakes.detach())
            d_loss += bce(pred_fake, fake_labels)

            d_loss.backward()

            d_optimizer.step()

            ### Train Gen ###

            g_optimizer.zero_grad()

            z = to_var(torch.randn((batch_data.size(0), args.nz)), device)
            fakes = gen(z)
            pred_fake = discriminator(fakes)
            gen_loss = bce(pred_fake, real_labels)

            gen_loss.backward()
            g_optimizer.step()

            if total_iters % 5000 == 0:
                print("Iteration %d >> g_loss: %.4f., d_loss: %.4f." % (total_iters, gen_loss, d_loss))
                torch.save(gen.state_dict(), os.path.join(args.out, 'gen_%d.pkl' % 0))
                torch.save(discriminator.state_dict(), os.path.join(args.out, 'disc_%d.pkl' % 0))
                gen.eval()
                display_images(args, gen, fixed_noise)
                test(args, gen)
                gen.train()
                # fixed_fake = gen(fixed_noise).detach().cpu().numpy()
                # real_data = batch_data[0].detach().cpu().numpy()
                # display_noise(fixed_fake.squeeze(), os.path.join(args.out, "gen_sample_%d.png" % i))
                # display_noise(real_data.squeeze(), os.path.join(args.out, "real_%d.png" % 0))
            total_iters += 1


def test(args, gen):
    scores_dict = dict()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    test_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=False,
        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]),
        download=args.dl,
    )

    loader = DataLoader(test_data, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    loss_crit = nn.L1Loss().to(device)

    for batch in loader:
        batch_data, labels = batch
        batch_data = to_var(batch_data, device)
        batch_data = batch_data.view(-1, 784)

        z = torch.randn(batch_data.size(0), args.nz, device=device_str, requires_grad=True)
        z_optim = Adam([z], lr=args.infer_lr)
        for j in range(args.infer_iter):
            z_optim.zero_grad()
            fakes = gen(z)
            loss = loss_crit(fakes, batch_data)

            loss.backward()
            z_optim.step()

        fakes = gen(z)
        batch_scores = torch.sum(torch.abs(fakes - batch_data), dim=1)

        for k in range(batch_scores.size(0)):
            scores_dict[batch_scores[k]] = labels[k]


    anamoly_100 = sorted(scores_dict.items(), key=lambda x: x[0], reverse=True)[:100]
    total_counts = 10 * [0]
    for key_val in anamoly_100:
        key, val = key_val
        total_counts[val] += 1

    print('>>>>>>>>>>>>>>>')
    print('Total counts: ', total_counts)
    print('>>>>>>>>>>>>>>>')



def distance_score_from_gan_dist(args):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    gen = Generator(args.nz, 784)
    gen.load_state_dict(torch.load(args.gen_file))
    gen = gen.to(device)
    gen.eval()

    dataset = GalaxySet(args.data_path, args.normalized, args.out)
    loader = DataLoader(dataset, batch_size=args.infer_bs, shuffle=False, num_workers=2, drop_last=False)
    scores = torch.zeros(len(loader) * args.infer_bs)

    loss_crit = nn.L1Loss().to(device)

    for i, batch in tqdm(enumerate(loader)):
        batch = to_var(batch, device)[:, :1600:2]
        z = torch.randn(batch.size(0), args.nz, device=device_str, requires_grad=True)
        z_optim = Adam([z], lr=args.infer_lr)
        for j in range(args.infer_iter):
            z_optim.zero_grad()
            fakes = gen(z)
            loss = loss_crit(fakes, batch)

            loss.backward()
            z_optim.step()

            # if j % 20 == 0:
            #     print("Iter %d: loss %.4f" % (j, loss))
        fakes = gen(z)
        batch_scores = torch.sum(torch.abs(fakes - batch), dim=1)
        scores[i * args.infer_bs:  i * args.infer_bs + batch_scores.size(0)] = batch_scores

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
    scores = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)

    np.savetxt("all_scores.csv", scores, delimiter=",")
    np.savetxt("top_100.csv", scores[:100], delimiter=",")
    np.savetxt("top_500.csv", scores[:500], delimiter=",")

    # anamoly_100 = sorted(anomoly_dict.items(), key=lambda x: -x[1])[:100]
    # anamoly_100 = [x[0] for x in anamoly_100]
    #
    wall = np.load(args.wall_path)
    wall = sorted(list(enumerate(wall)), key=lambda x: x[1], reverse=True)

    np.savetxt("wall_scores.csv", wall, delimiter=",")
    np.savetxt("wall_100.csv", wall[:100], delimiter=",")

    list_scores = [x[0] for x in scores[:100]]
    list_wall = [x[0] for x in wall[:100]]

    scores_set = set(list_scores)
    wall_set = set(list_wall)

    print('Intersection 100 : {}'.format(scores_set.intersection(wall_set).__len__()))

    list_scores = [x[0] for x in scores[:500]]
    list_wall = [x[0] for x in wall[:500]]

    scores_set = set(list_scores)
    wall_set = set(list_wall)

    print('Intersection 500 : {}'.format(scores_set.intersection(wall_set).__len__()))

    # sum_100 = 0
    # sum_rand = 0
    #
    # list_scores = [x[0] for x in scores]
    #
    # for i in range(100):
    #     cur = wall[i][0]
    #     sum_100 += list_scores.index(cur)
    #
    #     cur_rand = random.choice(wall)[0]
    #     sum_rand += list_scores.index(cur_rand)
    #
    # print('Average top score %.4f', float(sum_100)/100)
    # print('Average rand score %.4f', float(sum_rand)/100)




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
    parser.add_argument('--digit', type=int, default=0)
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--wall_path', type=str)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--infer_iter', type=int, default=50)
    parser.add_argument('--infer_lr', type=float, default=0.2)
    parser.add_argument('--infer_bs', type=int, default=128)
    parser.add_argument('--dl', action='store_true')
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
