import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import random
from skimage.transform import rotate


class loader(Dataset):
    def __init__(self, X_list, aug=False, rgb_channel=3, **kwargs):
        self.X_list = X_list
        self.aug = aug
        self.rgb_channel = rgb_channel

    def __getitem__(self, index):
        filepath = self.X_list[index]
        X = np.load(filepath)

        # Replace with own loader. Output X should have size [channel=3, num_frame=30, x_dimension=112, y_dimension=112]

        X = torch.from_numpy(X).float()
        return X, Y, ID

    def __len__(self):
        return len(self.X_list)


def create_dataloader(mode, batch_size=16, num_workers=[4, 4], data_folder='../data/LUMC_A4C/ver3/',
                      split_folder='split_000all_400_401/', **kwargs):

    X_list = np.load(data_folder + split_folder + '/' + mode + '_list_RGB.npy').tolist()
    if mode == 'train':
        data = loader(X_list, aug=True, **kwargs)
        dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers[0], pin_memory=True)
    elif (mode == 'val') | (mode == 'test'):
        data = loader(X_list, aug=False, **kwargs)
        dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=1, shuffle=False, drop_last=False, num_workers=num_workers[1], pin_memory=True)
    return dataloader


# if __name__ == '__main__':
#     dataloader = create_dataloader('train')
#     print(len(dataloader))
#     for i, batch in enumerate(dataloader):
#         print(batch[0].shape, batch[1].shape, batch[2][0])