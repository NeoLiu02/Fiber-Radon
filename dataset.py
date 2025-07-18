import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils import show_images
import os
from Model import RTpad
from torch.fft import rfft, irfft
from skimage.transform import radon, iradon
import matplotlib.pyplot as plt
from torch_radon import Radon
import joblib
from scipy.ndimage import zoom

class Mydataset(Dataset):
    def __init__(self, path1, path2, pca_model_path=None, path3=None):
        # If pca_model_path is provided, it will subtract the background component from the data
        if pca_model_path == None:
            self.data = np.load(path1) #[N,H,W]
        else:
            print("Initializing data, Subtracting background ...")
            self.pca_model = joblib.load(pca_model_path)
            data_raw = np.load(path1)
            if data_raw.ndim > 2:
                data_raw_flattened = data_raw.reshape(data_raw.shape[0], -1)
            data_coefficients = self.pca_model.transform(data_raw_flattened)
            bg_component = np.outer(data_coefficients[:, 0], self.pca_model.components_[0, :])
            data_flattened = data_raw_flattened - bg_component
            self.data = data_flattened.reshape(data_raw.shape)
            print("Successfully subtracted background component.")

        self.label = np.load(path2)
        if path3 is not None:
            self.mask = np.load(path3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        speckle = self.data[idx]
        label = self.label[idx]
        speckle = torch.from_numpy(speckle)
        label = torch.from_numpy(label)
        if hasattr(self, 'mask'):
            mask = self.mask[idx]
            mask = torch.from_numpy(mask)
            return {'speckle': speckle, 'label': label, 'RTmask':mask}
        else:
            return {'speckle': speckle, 'label': label}

    def get_size(self):
        (c1, h1, w1) = self.data.shape
        (c2, h2, w2) = self.label.shape
        return h1, w1, h2, w2


if __name__ == "__main__":
    val_data = Mydataset('data/speckle/RT/valid', 'data/origin/RT/valid')
    speckle_h, speckle_w, image_h, image_w = val_data.get_size()
    assert speckle_h == speckle_w and image_h == image_w
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    angle_s = speckle_h
    angle_i = image_h
    RTs = Radon(angle_s, np.linspace(0, np.pi, speckle_w, endpoint=False), clip_to_circle=True)
    RTi = Radon(angle_i, np.linspace(0, np.pi, image_w, endpoint=False), clip_to_circle=True)

    flag = 0
    for data in val_loader:
        flag += 1
        images_sino = data['label'].unsqueeze(1).to(torch.float32).cuda()
        speckle_sino = data['speckle'].unsqueeze(1).to(torch.float32).cuda()
        print(torch.max(images_sino))
        print(torch.max(speckle_sino))

        # RT test
        if flag == 1:
            num = 0
            print(images_sino.shape)
            show_images(images_sino, keep_range=True)
            plt.show()

            images_fsino = RTi.filter_sinogram(images_sino)
            ifbp = RTi.backprojection(images_fsino)
            show_images([images_sino[num, :, :, :], images_fsino[num, :, :, :],
                         ifbp[num, :, :, :]], keep_range=False)
            plt.show()

            ####################################################################
            # Test RTpad
            # Pad = RTpad(pad_width=30, if_zero=True)
            # images_pad = Pad(images_sino)
            # print(images_pad.shape)
            # # show_images(images_pad,keep_range=False)
            # plt.imshow(images_pad[num, :, :, :].cpu().squeeze().numpy())
            ####################################################################

            print(speckle_sino.shape)
            speckle_fsino = RTs.filter_sinogram(speckle_sino)
            sfbp = RTs.backprojection(speckle_fsino)
            show_images([speckle_sino[num, :, :, :], speckle_fsino[num, :, :, :],
                         sfbp[num, :, :, :]], keep_range=False)
            plt.show()
            show_images([images_fsino[num, :, :, :],speckle_fsino[num, :, :, :]], keep_range=False)
            plt.show()