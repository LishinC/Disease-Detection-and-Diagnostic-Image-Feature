import numpy as np
#from glob2 import glob
import torch
from torch.utils.data.dataset import Dataset
import random
from skimage.transform import rotate


def cropping(volume, target_t, target_xy, t, x, y, last_dim):
    if last_dim:
        volume = volume[t:t+target_t, x:x+target_xy, y:y+target_xy, :]
    else:
        volume = volume[t:t+target_t, x:x+target_xy, y:y+target_xy]
    return volume


def rotation(X, deg):
    # tdim, xdim, ydim = X.shape[0], X.shape[1], X.shape[2]
    X = np.moveaxis(X,0,-1)     # move time dimension to the last dimension
    X = rotate(X, deg)          # requires input of shape [xdim, ydim, channel]
    X = np.moveaxis(X,-1,0)     # move time dimension back to the first dimension
    return X


class loader(Dataset):
    def __init__(self, X_list, aug=False, rgb_channel=3, **kwargs):
        self.X_list = X_list
        self.aug = aug
        self.rgb_channel = rgb_channel

    def __getitem__(self, index):
        filepath = self.X_list[index]
        X = np.load(filepath)
        filepath = filepath.split('/')[-1]
        ID = filepath[:filepath.find('.npy')]
        c = filepath[:3]

        # Regurgitation/impairedLV
        if (c=='000')|(c=='0_1')|(c=='0_2')|(c=='0_3'):
            Y = torch.zeros(1)
        elif (c=='104')            |(c=='400'):
            Y = torch.ones(1)
        elif (c=='105')|(c=='106') |(c=='401'):
            Y = torch.ones(1)*2

        if self.aug != False:
            t = random.randint(0, X.shape[0] - 30)
            xy = random.randint(0, 12)

            # Rotation
            deg = random.randint(-15, 15)
            X = rotation(X, deg)

        else:
            t = 0
            xy = 6

        # Translation
        X = cropping(X, 30, 112, t, xy, xy, False)

        # Color Channel
        if self.rgb_channel==3:
            X = np.stack((X,X,X), axis=0)
        elif self.rgb_channel==1:
            X = np.expand_dims(X, axis=0)

        # output X of size [3, 30, 112, 112] for RGB; [2, 30, 112, 112] for FLOW or ACC
        X = torch.from_numpy(X).float()
        return X, Y, ID

    def __len__(self):
        return len(self.X_list)


def create_dataloader(mode, batch_size=16, num_workers=[4, 4], data_folder='../data/LUMC_A4C/ver3/',
                      split_folder='split_000all_400_401/', **kwargs): #split_folder='split_000all_104_105_106/'

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