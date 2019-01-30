import os

import numpy as np
from torch.utils.data import Dataset
import torch
import shutil

import matplotlib.pyplot as plt


class GalaxySet(Dataset):
    def __init__(self, data_path, normalized=False, out=''):
        super(GalaxySet, self).__init__()

        self.data_files, self.base_dir = self.prep_data(data_path, normalized, out)

    def prep_data(self, data_path, normalized, out):
        if not normalized:

            np_data = np.load(data_path)

            # find max feature value:
            max_data = np.max(np_data, axis=0)
            clipped_data = (np_data / max_data) * 2 - 1

            means = np.mean(clipped_data, axis=0)
            stds = np.std(clipped_data, axis=0)

            normalized_data = (clipped_data - means) / stds
            normalized_dir = os.path.join(out, "specs_normalized")
            if os.path.exists(normalized_dir):
                x = input("normalized dir exists. Do you want to delete? [y/n]")
                if x == 'y':
                    shutil.rmtree(normalized_dir)
                    os.mkdir(normalized_dir)
                else:
                    raise Exception
            else:
                os.mkdir(normalized_dir)
            for i in range(normalized_data.shape[0]):
                np.save(os.path.join(normalized_dir, "specs_normalized_%d" % i), normalized_data[i])
            print('Saving normalized data')
        else:
            normalized_dir = data_path

        data_files = os.listdir(normalized_dir)

        return data_files, normalized_dir

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, item):
        features_file = os.path.join(self.base_dir, self.data_files[item])
        features = np.load(features_file)
        features = torch.from_numpy(features).float()

        return features


class QuasarsSet(Dataset):
    def __init__(self, data_path, normalized=False, out=''):
        super(QuasarsSet, self).__init__()

        self.data_files, self.base_dir = self.prep_data(data_path, normalized, out)

    def prep_data(self, data_path, normalized, out):
        if not normalized:

            np_data = np.load(data_path)[:, :1600:2]

            # find max feature value:
            max_data = np.max(np_data, axis=0)
            clipped_data = (np_data / max_data) * 2 - 1

            means = np.mean(clipped_data, axis=0)
            stds = np.std(clipped_data, axis=0)

            normalized_data = (clipped_data - means) / stds
            normalized_dir = os.path.join(out, "quasars_specs_normalized")
            if os.path.exists(normalized_dir):
                x = input("normalized dir exists. Do you want to delete? [y/n]")
                if x == 'y':
                    shutil.rmtree(normalized_dir)
                    os.mkdir(normalized_dir)
                else:
                    raise Exception
            else:
                os.mkdir(normalized_dir)
            for i in range(normalized_data.shape[0]):
                np.save(os.path.join(normalized_dir, "specs_normalized_%d" % i), normalized_data[i])
            print('Saving normalized data')
        else:
            normalized_dir = data_path

        data_files = os.listdir(normalized_dir)

        return data_files, normalized_dir

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, item):
        features_file = os.path.join(self.base_dir, self.data_files[item])
        features = np.load(features_file)
        features = torch.from_numpy(features).float()

        return features


def display_noise(noise, out):
    fig = plt.figure()
    plt.plot(noise)
    plt.savefig(out)
    fig.clf()

if __name__ == "__main__":
    x = np.random.randint(0, 1000, (1000, 8295))

    np.save("test_normalize.npy", x)

    galaxy_set = GalaxySet("./specs_normalized", normalized=True)

    y = galaxy_set[0]

    print(torch.mean(y, dim=0))
    print(torch.std(y, dim=0))


    # noise = np.random.randint(0, 1000, 8295)
    # display_noise(noise, './')
