import numpy as np
from torch.utils.data import Dataset
import torch


class GalaxySet(Dataset):
    def __init__(self, data_path, normalized=False):
        super(GalaxySet, self).__init__()

        self.data = self.load_data(data_path, normalized)

    def load_data(self, data_path, normalized):
        np_data = np.load(data_path)

        if not normalized:
            # find max feature value:
            max_data = np.max(np_data, axis=0)
            clipped_data = (np_data / max_data) * 2 - 1

            means = np.mean(clipped_data, axis=0)
            stds = np.std(clipped_data, axis=0)

            normalized_data = (clipped_data - means) / stds

            np.save(data_path + "normalized", normalized_data)
        else:
            normalized_data = np_data

        return normalized_data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        features = self.data[item]

        features = torch.from_numpy(features)

        return features


if __name__ == "__main__":
    x = np.random.randint(0, 1000, (1000, 800))

    np.save("test_normalize.npy", x)

    galaxy_set = GalaxySet("test_normalize.npy")

    y = galaxy_set[0]

    print
    torch.mean(y, dim=0)
    print
    torch.std(y, dim=0)
