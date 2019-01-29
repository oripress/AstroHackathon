import argparse

def train(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='')
    parser.add_argument('--out', default='out')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--iters', type=int, default=1250000)

    args = parser.parse_args()

    train(args)
