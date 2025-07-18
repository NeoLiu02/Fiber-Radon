import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import SSIM, MS_SSIM
from sklearn.decomposition import PCA
from torch.fft import fftn, fft, ifft
from torch.nn import functional as F

def fftshift(x, axes=None):
    if axes is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    shifts = [x.shape[ax] // 2 for ax in axes]
    return torch.roll(x, shifts=shifts, dims=axes)

class NPCC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, target):
        batch, channel, _, _ = image.shape
        x = image.contiguous().view(batch, channel,-1)
        target = target.contiguous().view(batch, channel, -1)
        mean1 = torch.mean(x, dim=2).unsqueeze(2)
        mean2 = torch.mean(target, dim=2).unsqueeze(2)
        cov1 = torch.matmul(x-mean1, (target-mean2).transpose(1,2))
        diag1 = torch.matmul(x-mean1, (x-mean1).transpose(1,2))
        diag2 = torch.matmul(target-mean2, (target-mean2).transpose(1,2))
        pearson = cov1 / torch.sqrt(diag2 * diag1)
        return 1-pearson.squeeze().mean()


class TORCH_SSIM(nn.Module):
    def __init__(self, data_range=1, size_average=True, channel=1):
        super().__init__()
        self.ssim_module = SSIM(data_range=data_range, size_average=size_average, channel=channel)
        self.ms_ssim_module = MS_SSIM(data_range=data_range, size_average=size_average, channel=channel)

    def forward(self, image, target, mode='ssim'):
        if mode == 'ssim':
            criterion = self.ssim_module(image, target)
        elif mode == 'msssim':
            criterion = self.ms_ssim_module(image, target)
        return criterion


class TV(nn.Module):
    def __init__(self):
        super(TV, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
        w_tv = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
        return h_tv, w_tv

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def RTpad(tensor, pad_width):
    tensor = nn.functional.pad(tensor, pad=(pad_width, pad_width, 0, 0), mode='constant', value=0)
    pad_left = torch.flip(tensor[:, :, -pad_width:, :], dims=[3]) #tensor[:, :, :, -pad_width:] #
    # pad_left = torch.roll(pad_left, shifts=1, dims=3)
    pad_right = torch.flip(tensor[:, :, 0:pad_width, :], dims=[3]) #tensor[:, :, :, 0:pad_width] #
    # pad_right = torch.roll(pad_right, shifts=1, dims=3)
    tensor_pad1 = torch.cat([pad_left, tensor, pad_right], dim=2)
    return tensor_pad1


class HL_condition(nn.Module):
    def __init__(self, shape):
        super(HL_condition, self).__init__()
        self.H = shape[0]
        self.W = shape[1]
        freq_range_s = np.linspace(0, self.W, self.W, endpoint=False)
        # to torch float32
        self.freq_range_s = torch.from_numpy(freq_range_s).cuda().float()/self.W
        self.j = torch.complex(torch.tensor([0.0]), torch.tensor([1.0])).cuda()
        self.pi = torch.tensor(np.pi).cuda()

    def forward(self, x, mask):
        # x: (B, C, H, W)
        assert x.dim() == 4, "Input tensor must be 4D (batch_size, channels, height, width)"
        assert x.size() == mask.size(), "Input tensor and mask must have the same shape"
        assert x.shape[2] == self.H and x.shape[3] == self.W, "Input tensor must have the same height and width as the specified shape"

        # 2D Fourier transform for sinogram
        forward_transform = fftn(x, dim=(-1,-2), norm='ortho')
        freq_grid_s = self.freq_range_s.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(x.shape[0], 1, self.H, 1)
        reverse_transform = ifft(fft(x, dim=-2, norm='ortho'), dim=-1, norm='ortho')*torch.exp(self.j*2*self.pi*freq_grid_s)
        transform = fftshift((forward_transform + reverse_transform)/2, axes=(-1, -2))
        power_outside_wedge = torch.sum((transform.abs()**2)*(1-mask), dim=(-1, -2))/torch.sum((transform.abs()**2), dim=(-1, -2))  

        return power_outside_wedge

class VAELoss(nn.Module):
    def __init__(self, pretrained_vae, lambda_latent=1.0, lambda_recon=1.0):
        super(VAELoss, self).__init__()
        self.vae = pretrained_vae
        self.lambda_latent = lambda_latent
        self.lambda_recon = lambda_recon
        print("VAELoss initialized. VAE model is frozen.")

    def forward(self, generated_image, target_image):
        
        assert generated_image.shape == target_image.shape, "Generated and target images must have the same shape."
        mu_gen, _ = self.vae.encode(generated_image)
        mu_target, _ = self.vae.encode(target_image)
        
        latent_loss = F.mse_loss(mu_gen, mu_target) / mu_gen.shape[0]  # Average over batch size

        recon_gen = self.vae.decode(mu_gen)
        recon_target = self.vae.decode(mu_target)
        recon_loss = F.mse_loss(recon_gen, recon_target, reduction='sum') / recon_gen.shape[0]  # Average over batch size

        total_loss = self.lambda_latent * latent_loss + self.lambda_recon * recon_loss
        
        return total_loss


def show_images(imgs, titles=None, keep_range=True, shape=None, figsize=(8, 8.5), colorbar=False):
    imgs = [x.squeeze().cpu().numpy() for x in imgs]
    combined_data = np.array(imgs)

    if titles is None:
        titles = [str(i) for i in range(combined_data.shape[0])]

    # Get the min and max of all images
    if keep_range:
        _min, _max = np.amin(combined_data), np.amax(combined_data)
    else:
        _min, _max = None, None

    if shape is None:
        shape = (1, len(imgs))

    fig, axes = plt.subplots(*shape, figsize=figsize, sharex=True, sharey=True)
    ax = axes.ravel()
    for i, (img, title) in enumerate(zip(imgs, titles)):
        # print(img.shape)
        ax[i].imshow(img, cmap=plt.cm.Greys_r, vmin=_min, vmax=_max)
        ax[i].set_title(title)
        if colorbar:
            plt.colorbar(ax[i].images[0], ax=ax[i], fraction=0.046, pad=0.04)

def plot_lines(imgs, vline_index_left=None, vline_index_right=None, titles=None, shape=None, figsize=(8, 8.5)):
    imgs = [x.squeeze().cpu().numpy() for x in imgs]
    combined_data = np.array(imgs)

    

    if titles is None:
        titles = [str(i) for i in range(combined_data.shape[0])]

    if shape is None:
        shape = (1, len(imgs))

    fig, axes = plt.subplots(*shape, figsize=figsize, sharex=True, sharey=True)
    ax = axes.ravel()
    
    if vline_index_left is not None and vline_index_right is not None:
        vline_indeces_left = [y.squeeze().cpu().numpy() for y in vline_index_left]
        vline_indeces_right = [y.squeeze().cpu().numpy() for y in vline_index_right]
        for i, (img, title, vleft, vright) in enumerate(zip(imgs, titles, vline_indeces_left, vline_indeces_right)):
            ax[i].plot(img)
            ax[i].axvline(x=vleft, color='r', linestyle='--', label='Left Line')
            ax[i].axvline(x=vright, color='g', linestyle='--', label='Right Line')
            ax[i].set_title(title)

    else:
        for i, (img, title) in enumerate(zip(imgs, titles)):
            ax[i].plot(img)
            ax[i].set_title(title)

def pca_check(data, n_components=32):
    if len(data.shape) == 3:
        data = data.reshape(data.shape[0], -1)
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data.reshape(data.shape[0], -1))
    return transformed_data, pca




if __name__ == "__main__":
    tensor = torch.randn(2,2,4,4)
    tensor = RTpad(tensor, 2)
    print(tensor)

