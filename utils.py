import os

import numpy as np
from torch.utils.data import Dataset
import torch

import matplotlib.pyplot as plt
import torchvision.utils as vutils


class GalaxySet(Dataset):
    def __init__(self, data_path, normalized=False, out=''):
        super(GalaxySet, self).__init__()

        self.data = self.load_data(data_path, normalized, out)

    def load_data(self, data_path, normalized, out):
        np_data = np.load(data_path)

        if not normalized:
            # find max feature value:
            max_data = np.max(np_data, axis=0)
            clipped_data = (np_data / max_data) * 2 - 1

            means = np.mean(clipped_data, axis=0)
            stds = np.std(clipped_data, axis=0)

            normalized_data = (clipped_data - means) / stds
            np.save(os.path.join(out, "specs_normalized.npy"), normalized_data)
            print('Saving normalized data')
        else:
            normalized_data = np_data

        return normalized_data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        features = self.data[item]
        features = torch.from_numpy(features).float()

        return features


def display_noise(noise, out):
    fig = plt.figure()
    plt.plot(noise)
    plt.savefig(out)
    fig.clf()


def display_images(args, gen, fixed_noise):
    imgs = gen(fixed_noise)
    # imgs = imgs.unsqueeze(1)
    # imgs = imgs.repeat(1, 3, 1)
    imgs = imgs.view(-1, 1, 28, 28)
    vutils.save_image(imgs,
                      '%s/experiments.png' % (args.out),
                      normalize=True, nrow=8)

if __name__ == "__main__":
    # x = np.random.randint(0, 1000, (1000, 8295))
    #
    # np.save("test_normalize.npy", x)
    #
    # galaxy_set = GalaxySet("test_normalize.npy")
    #
    # y = galaxy_set[0]
    #
    # print(torch.mean(y, dim=0))
    # print(torch.std(y, dim=0))
    #

    noise = np.random.randint(0, 1000, 8295)
    display_noise(noise, './')
