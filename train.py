import torch.nn as nn
import torch.optim as optim

import argparse

from nets import Generator, Infer, weights_init


def train(args):
    gen = Generator()
    gen.cuda()
    gen.apply(weights_init())

    infer = Infer()
    infer.cuda()
    infer.apply(weights_init())

    mse = nn.MSELoss()
    mse.cuda()

    opt_g = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_i = optim.Adam(infer.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for i in range(args.iters):
        opt_g.zero_grad()
        opt_i.zero_grad()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='')
    parser.add_argument('--out', default='out')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--iters', type=int, default=1250000)

    args = parser.parse_args()

    train(args)
