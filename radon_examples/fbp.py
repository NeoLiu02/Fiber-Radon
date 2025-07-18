import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
sys.path.append('/app/neo_fiber')
from utils import show_images
from torch.fft import fftn, fft, ifft
import torch
import matplotlib.pyplot as plt
from torch_radon import Radon

device = torch.device('cuda')

img = np.load("data/origin/mnist/raw/test2.npy")[9, :, :]  # Load a single image
image_size = img.shape[0]
n_angles = image_size

# Instantiate Radon transform. clip_to_circle should be True when using filtered backprojection.
angles = np.linspace(0, np.pi, n_angles, endpoint=False)
radon = Radon(image_size, angles, clip_to_circle=True)

with torch.no_grad():
    x = torch.FloatTensor(img).to(device)

    sinogram = radon.forward(x)
    filtered_sinogram = radon.filter_sinogram(sinogram)
    fbp = radon.backprojection(filtered_sinogram)

print("FBP Error", torch.norm(x - fbp).item())


# Show results
titles = ["Original Image", "Sinogram", "Filtered Sinogram", "Filtered Backprojection"]
show_images([x, sinogram, filtered_sinogram, fbp], titles, keep_range=False)
plt.show()




#################### fft analysis ####################
def fftshift(x, axes=None):
    if axes is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    shifts = [x.shape[ax] // 2 for ax in axes]
    return torch.roll(x, shifts=shifts, dims=axes)

def ifftshift(x, axes=None):
    if axes is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    shifts = [-(x.shape[ax] // 2) for ax in axes]
    return torch.roll(x, shifts=shifts, dims=axes)

def sinogram_freq_complete(data: torch.Tensor) -> torch.Tensor:
    assert data.ndim == 3, "Input data must be a 3D tensor (batch_size, H, W)"
    # forward_transform = fftshift(fftn(data, dim=(1,2), norm='ortho'), axes=(1,2))
    # reverse_transform = ifftshift(ifft(fftshift(fft(data, dim=1, norm='ortho'), axes=1), dim=2, norm='ortho'), axes=2)
    # forward_transform = fftn(data, dim=(1, 2), norm='ortho')
    data_inplace = data.clone()
    data_inplace[:,:,:] = torch.flip(data_inplace[:,:,:], dims=[2])  # Flip
    # reverse_transform = fftn(data_inplace, dim=(1, 2), norm='ortho')
    transform = fftn(data+data_inplace, dim=(1, 2), norm='ortho')
    # freq_range_s = np.linspace(0, 1/np.pi, data.shape[2], endpoint=False)
    # freq_range_s = torch.from_numpy(freq_range_s).cuda()
    # freq_grid_s = freq_range_s.unsqueeze(0).unsqueeze(0).repeat(data.shape[0], data.shape[1], 1)
    # transform = (forward_transform + reverse_transform*torch.exp(-j*pi*freq_grid_s))/2
    # print("Frequency grid shape:", freq_grid_s.shape)
    
    # return transform
    return fftshift(transform, axes=(1, 2))  # Shift the zero frequency component to the center

# sinogram_fft = fftshift(fftn(sinogram, dim=(0, 1), norm='ortho'))
# sinogram_fft_complete = sinogram_freq_complete(sinogram.unsqueeze(0))
# radon2 = Radon(image_size, np.linspace(0, 2*np.pi, 2*n_angles, endpoint=False), clip_to_circle=True)
# sinogram2 = radon2.forward(x)
# sinogram2_fft = fftshift(fftn(sinogram2, dim=(0, 1), norm='ortho'))
# show_images([torch.log(sinogram_fft.abs()), torch.log(sinogram_fft_complete.squeeze().abs()), torch.log(sinogram2_fft[0::2,:].abs())],
#             ["Sinogram FFT", "Sinogram FFT Complete", "2"], keep_range=False, colorbar=True)
# plt.show()

sinogram3 = np.load("data/origin/mnist/RT/test2.npy")[9, :, :]
sinogram3 = torch.FloatTensor(sinogram3).to(device)
image = radon.backprojection(radon.filter_sinogram(sinogram3))
sinogram3_fft = fftshift(fftn(sinogram3, dim=(0, 1), norm='ortho'))
sinogram3_fft_complete = sinogram_freq_complete(sinogram3.unsqueeze(0))
show_images([image, torch.log(sinogram3_fft.abs()), torch.log(sinogram3_fft_complete.squeeze().abs())],
            ["image", "Sinogram 3 FFT", "Sinogram 3 FFT Complete"], keep_range=False, colorbar=True)
plt.show()