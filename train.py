import torch.nn as nn
import torch
import argparse

from nets import Generator, Infer, weights_init
from utils import GalaxySet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam


def to_var(x, device):
    return Variable(x, requires_grad=False).device(device)


def train(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = Generator()
    gen = gen.device(device)
    gen.apply(weights_init())

    infer = Infer()
    infer = infer.device(device)
    infer.apply(weights_init())

    mse = nn.MSELoss()
    mse = mse.device(device)

    galaxy_dataset = GalaxySet(args.data_path)
    loader = DataLoader(galaxy_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    loader_iter = iter(loader)

    i_optimizer = Adam(infer.parameters(), args.lr)
    g_optimizer = Adam(gen.parameters(), args.lr)

    real_labels = to_var(torch.ones(args.bs), device)
    fake_labels = to_var(torch.zeros(args.bs), device)

    for i in range(args.iters):
        try:
            batch_data = loader_iter.next()
        except StopIteration:
            loader_iter = iter(loader)
            batch_data = loader_iter.next()

        batch_data = to_var(batch_data)

        ### Train Infer ###

        i_optimizer.reset_grads()
        g_optimizer.reset_grads()

        # train Infer with real
        pred_real = infer(batch_data)
        infer_loss = mse(pred_real, real_labels)

        # train infer with fakes
        z = to_var(torch.randn((args.bs, args.z)), device)
        fakes = gen(z)
        pred_fake = infer(fakes)
        infer_loss += mse(pred_fake, fake_labels)

        infer_loss.backward()

        i_optimizer.step()

        ### Train Gen ###

        i_optimizer.reset_grads()
        g_optimizer.reset_grads()

        z = to_var(torch.randn((args.bs, args.z)), device)
        fakes = gen(z)
        pred_fake = infer(fakes)
        gen_loss = mse(pred_fake, real_labels)

        gen_loss.backward()

        g_optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='out')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--iters', type=int, default=1250000)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--z', type=int, default=100)

    args = parser.parse_args()

    train(args)
