import torch
from torch import nn
import numpy as np
from dataset import Mydataset
from torch.utils.data import DataLoader
from Model import RTMnet
import argparse
from utils import NPCC, TORCH_SSIM
from torch_radon import Radon
import os
import cv2
import torch.nn.functional as F
from vae import VAE
import scipy.io as sio

# python evalRT.py  --dataset=mnist --RT_pad=1 --mode=test2

p = argparse.ArgumentParser()
p.add_argument('--model_path', type=str, default='model/', help='torch save trained model')
p.add_argument('--dataset', type=str, default='mnist', help='mnist or fashionmnist')
p.add_argument('--RT_pad', type=int, default=1, help='0 or 1')
p.add_argument('--mode', type=str, default='test2', help='train, valid, test1, test2')

arg = p.parse_args()

dataset = arg.dataset
speckle_path = ['data/speckle/' + dataset + '/RT/train.npy', 'data/speckle/' + dataset + '/RT/valid.npy',
                'data/speckle/' + dataset + '/RT/test1.npy', 'data/speckle/' + dataset + '/RT/test2.npy',
                'data/speckle/' + dataset + '/RT/test3.npy', 'data/speckle/' + dataset + '/RT/train_3000.npy']
image_path = ['data/origin/' + dataset + '/RT/train.npy', 'data/origin/' + dataset + '/RT/valid.npy',
              'data/origin/' + dataset + '/RT/test1.npy', 'data/origin/' + dataset + '/RT/test2.npy',
              'data/origin/' + dataset + '/RT/test3.npy', 'data/origin/' + dataset + '/RT/train_3000.npy']

if arg.RT_pad:
    model_save_dir = arg.model_path + dataset + '/RTMnet_RTpad_7000.pth'
    RT_pad = True
    output_dir = 'output_RT/' + dataset + '/RTMnet_RTpad_7000'
    model = RTMnet(RT_pad=RT_pad, filter_size=256).cuda()
    print("RTpad enabled")
else:
    model_save_dir = arg.model_path + dataset + '/RTMnet.pth'
    RT_pad = False
    model = RTMnet(RT_pad=RT_pad, filter_size=256).cuda()
    output_dir = 'output_RT/' + dataset + '/RTMnet'


mode = arg.mode
mode_idx = ['train', 'valid', 'test1', 'test2', 'test3', 'train3000'].index(mode)
testset = Mydataset(speckle_path[mode_idx], image_path[mode_idx])
testloader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=True)
speckle_h, speckle_w, image_h, image_w = testset.get_size()
assert speckle_h == speckle_w and image_h == image_w

if not os.path.exists(output_dir + '/' + mode):
    os.makedirs(output_dir + '/' + mode + '/reconstruct')
    os.makedirs(output_dir + '/' + mode + '/RT')
    os.makedirs(output_dir + '/' + mode + '/RT_truth')
    os.makedirs(output_dir + '/' + mode + '/RT_speckle')

RTi = Radon(image_h, np.linspace(0, np.pi, image_w, endpoint=False), clip_to_circle=True)

pretrained = torch.load(model_save_dir)
print("load from model")
model.load_state_dict(pretrained)
model.eval()


ssim = TORCH_SSIM().cuda()
npcc = NPCC().cuda()
mse = nn.MSELoss().cuda()
l1 = []
l2 = []
l3 = []
global_step = 0
with torch.no_grad():
    for batch in testloader:
        speckle_RT = batch['speckle'].unsqueeze(1).to(torch.float32).cuda()
        image_RT = batch['label'].unsqueeze(1).to(torch.float32).cuda()
        output = model(speckle_RT)        

        reconstruct0 = RTi.backprojection(RTi.filter_sinogram(output))
        image0 = RTi.backprojection(RTi.filter_sinogram(image_RT))
        reconstruct0 = F.interpolate(reconstruct0[:, :, 38:38+180, 38:38+180], size=(256, 256))
        image0 = F.interpolate(image0[:, :, 38:38+180, 38:38+180], size=(256, 256))
        scale = torch.sum(image0, dim=(-2, -1), keepdim=True) / torch.sum(reconstruct0, dim=(-2, -1), keepdim=True)
        reconstruct0 = reconstruct0 * scale
        criterion = ssim(reconstruct0, image0, mode='msssim')
        l1.append(criterion.cpu().data.numpy())
        criterion2 = 1-npcc(reconstruct0, image0)
        l2.append(criterion2.cpu().data.numpy())
        criterion3 = mse(reconstruct0, image0)
        l3.append(criterion3.cpu().data.numpy())


        output = output.cpu().squeeze().numpy()
        reconstruct = reconstruct0.cpu().squeeze().numpy()
        image_RT = image_RT.cpu().squeeze().numpy()
        speckle_RT = speckle_RT.cpu().squeeze().numpy()
        image0 = image0.cpu().squeeze().numpy()


        global_step += 1
        # if global_step <= 1000:
            # cv2.imwrite(output_dir + '/' + mode + '/reconstruct/' + str(global_step) + '.png', reconstruct/np.max(reconstruct)*255)
            # cv2.imwrite(output_dir + '/' + mode + '/RT/' + str(global_step) + '.png', output / np.max(output) * 255)
            # cv2.imwrite(output_dir + '/' + mode + '/RT_truth/' + str(global_step) + '.png', image_RT / np.max(image_RT) * 255)
            # cv2.imwrite(output_dir + '/' + mode + '/RT_speckle/' + str(global_step) + '.png', speckle_RT / np.max(speckle_RT) * 255)
            # cv2.imwrite(output_dir + '/' + mode + '/truth/' + str(global_step) + '.png', image0 / np.max(image0) * 255)

loss = sum(l1)/len(l1)
loss2 = sum(l2)/len(l2)
loss3 = sum(l3)/len(l3)

sio.savemat(output_dir + '/' + mode + '/reconstruct/ssim.mat', mdict={'ssim':l1})
print("ssim =", loss)
sio.savemat(output_dir + '/' + mode + '/reconstruct/correlation.mat', mdict={'pcc':l2})
print("correlation =", loss2)
sio.savemat(output_dir + '/' + mode + '/reconstruct/mse.mat', mdict={'mse':l3})
print("mse =", loss3)

with open(output_dir + '/' + mode + '/reconstruct/loss.txt','a') as f:
    f.truncate(0)
    f.write("msssim = " + str(loss) + "\n")
    f.write("correlation = " + str(loss2) + "\n")
    f.write("mse = " + str(loss3) + "\n")