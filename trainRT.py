import torch
from torch import nn
import numpy as np
from dataset import Mydataset
from torch.utils.data import DataLoader
from Model import TM, RTMnet
import argparse
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from utils import NPCC, TV, pca_check, HL_condition, VAELoss
from torch_radon import Radon
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime
from vae import VAE
import torch.nn.functional as F

# python trainRT.py --epoch=8 --lr=3e-4 --lr_decay=0 --loss=vae --resume=0 --sinoproject=1 --batch_size=64 --TV_w=0

p = argparse.ArgumentParser()
p.add_argument('--model_path', type=str, default='model/', help='torch save trained model')
p.add_argument('--ckpt_path', type=str, default='checkpoint/', help='torch save checkpoint model')
p.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
p.add_argument('--epoch', type=int, default=10, help='total epoch')
p.add_argument('--batch_size', type=int, default=10, help='total epoch')
p.add_argument('--resume', type=int, default=0, help='Resume or not')
p.add_argument('--lr_decay', type=int, default=1, help='if weight decay')
p.add_argument('--loss', type=str, default='L2', help='Options: L1,L2')
p.add_argument('--TV_w', type=float, default=0, help='TV loss')
p.add_argument('--sinoproject', type=float, default=None, help='if sinogram filter')
p.add_argument('--dataset', type=str, default='mnist', help='mnist or fashionmnist')
p.add_argument('--RT_pad', type=int, default=1, help='0 or 1')
p.add_argument('--model', type=str, default='RTMnet')

arg = p.parse_args()
epoch = arg.epoch
batchsize = arg.batch_size
lr = arg.lr

dataset = arg.dataset
speckle_path = ['data/speckle/' + dataset + '/RT/train.npy', 'data/speckle/' + dataset + '/RT/valid.npy',
                'data/speckle/' + dataset + '/RT/test1.npy', 'data/speckle/' + dataset + '/RT/test2.npy',
                'data/speckle/' + dataset + '/RT/test3.npy', 'data/speckle/' + dataset + '/RT/train_3000.npy']
image_path = ['data/origin/' + dataset + '/RT/train.npy', 'data/origin/' + dataset + '/RT/valid.npy',
              'data/origin/' + dataset + '/RT/test1.npy', 'data/origin/' + dataset + '/RT/test2.npy',
              'data/origin/' + dataset + '/RT/test3.npy', 'data/origin/' + dataset + '/RT/train_3000.npy']

train_speckle_path = speckle_path[5]  # Load the 3000 samples
train_image_path = image_path[5]  # Load the 3000 samples
valid_speckle_path = speckle_path[1]  # Load the validation set
valid_image_path = image_path[1]  # Load the validation set

if arg.RT_pad:
    ckpt_dir = arg.ckpt_path + dataset + '/RTMnet_checkpoint_RTpad.pth'
    model_save_dir = arg.model_path + dataset + '/RTMnet_RTpad.pth'
    RT_pad = True
    model = RTMnet(RT_pad=RT_pad, filter_size=256).cuda()
else:
    ckpt_dir = arg.ckpt_path + dataset + '/RTMnet_checkpoint.pth'
    model_save_dir = arg.model_path + dataset + '/RTMnet.pth'
    RT_pad = False
    model = RTMnet(RT_pad=RT_pad, filter_size=256).cuda()

if not os.path.exists(arg.ckpt_path + dataset):
    os.makedirs(arg.ckpt_path + dataset)
print("RTpad", RT_pad)

trainset = Mydataset(train_speckle_path, train_image_path)
validset = Mydataset(valid_speckle_path, valid_image_path)
speckle_h, speckle_w, image_h, image_w = trainset.get_size()
assert speckle_h == speckle_w and image_h == image_w
print("train data:", len(trainset))


loss_fun = arg.loss
loss_fun = ['l1', 'l2', 'npcc', 'vae'].index(loss_fun.lower())
L1 = nn.L1Loss().cuda()
if loss_fun == 0:
    criterion = nn.L1Loss().cuda()
    print('Using L1 loss function')
elif loss_fun == 1:
    criterion = nn.MSELoss().cuda()
    print('Using MSE loss function')
elif loss_fun == 2:
    criterion = NPCC().cuda()
    print('Using NPCC loss function')
elif loss_fun == 3:
    print('Using VAE loss function')
    vae_model = VAE().cuda()
    vae_model.load_state_dict(torch.load('model/' + dataset + '/vae_mnistRT_train3000.pth'))
    vae_model.eval()  # Set VAE model to evaluation mode
    for param in vae_model.parameters():
        param.requires_grad = False
    print("VAE model loaded and frozen")
    criterion = VAELoss(vae_model, lambda_latent=2.0, lambda_recon=0.05).cuda()

# TV regularizer
TV_w = arg.TV_w
compute_TV = TV().cuda()
# Use summary writer to log the training process
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S') # Format: YearMonthDay-HourMinuteSecond
log_dir = f'runs/training_run_{timestamp}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

RTi = Radon(image_h, np.linspace(0, np.pi, image_w, endpoint=False), clip_to_circle=True)
RTs = Radon(speckle_h, np.linspace(0, np.pi, speckle_w, endpoint=False), clip_to_circle=True)

trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, drop_last=True)
validloader = DataLoader(validset, batch_size=batchsize, shuffle=False, drop_last=True)

if arg.resume:
    pretrained = torch.load(ckpt_dir)
    # pretrained = torch.load(save_dir)
    print("load from checkpoint")
    model.load_state_dict(pretrained)


global_step = 0
for k in trange(epoch):
    currenttloss = 0
    for batch in trainloader:
        global_step += 1
        speckleRT = batch['speckle'].unsqueeze(1).to(torch.float32).cuda()
        imageRT = batch['label'].unsqueeze(1).to(torch.float32).cuda()
        outputRT = model(speckleRT)
        
        if arg.sinoproject is not None:
            output0 = RTi.filter_sinogram(outputRT)
            image0 = RTi.filter_sinogram(imageRT)
            output0 = RTi.backprojection(output0)
            image0 = RTi.backprojection(image0)
            image0 = image0 / torch.amax(image0, dim=(-1, -2), keepdim=True)  # Normalize to [0, 1]
            output0 = output0 / torch.amax(output0, dim=(-1, -2), keepdim=True)  # Normalize to [0, 1]
            lossbp = F.mse_loss(output0, image0, reduction='sum')/output0.shape[0] * arg.sinoproject
        else:
            lossbp = 0

        
        imageRT = imageRT / torch.amax(imageRT, dim=(-1, -2), keepdim=True)  # Normalize to [0, 1]
        scale = torch.sum(imageRT, dim=(-2, -1), keepdim=True) / torch.sum(outputRT, dim=(-2, -1), keepdim=True)
        outputRT = outputRT * scale
        if arg.TV_w > 0:
            output_TVx, output_TVy = compute_TV(outputRT)
            image_TVx, image_TVy = compute_TV(imageRT)
            loss = criterion(outputRT, imageRT)  \
                + TV_w * L1(output_TVx, image_TVx)  \
                + TV_w * L1(output_TVy, image_TVy)
        else:
            loss = criterion(outputRT, imageRT)
        loss += lossbp

        writer.add_scalar("loss", loss.item(), global_step=global_step)
        # plot_model(writer, model, global_step)

        if global_step % (500*4//batchsize) == 0:
            # writer.add_images("speckleRT", speckle, global_step=global_step)
            writer.add_images("outputRT", outputRT, global_step=global_step)
            writer.add_images("imageRT", imageRT, global_step=global_step)
            torch.save(model.state_dict(), ckpt_dir)
            print('model ckpt saved to', ckpt_dir)

        # print('loss:', loss.cpu().data.numpy())
        currenttloss = currenttloss + loss.cpu().data.numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('trainloss:', currenttloss / len(trainloader))
    
    if arg.lr_decay:
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=global_step)
        scheduler.step()

    # Validation
    with torch.no_grad():
        currenttloss = 0
        valid_step = global_step
        for batch in validloader:
            valid_step = + 1
            speckleRT = batch['speckle'].unsqueeze(1).to(torch.float32).cuda()
            imageRT = batch['label'].unsqueeze(1).to(torch.float32).cuda()
            outputRT = model(speckleRT)
            
            if arg.sinoproject is not None:
                output0 = RTi.filter_sinogram(outputRT)
                image0 = RTi.filter_sinogram(imageRT)
                output0 = RTi.backprojection(output0)
                image0 = RTi.backprojection(image0)
                image0 = image0 / torch.amax(image0, dim=(-1, -2), keepdim=True)  # Normalize to [0, 1]
                output0 = output0 / torch.amax(output0, dim=(-1, -2), keepdim=True)  # Normalize to [0, 1]
                lossbp = F.mse_loss(output0, image0, reduction='sum')/output0.shape[0] * arg.sinoproject
            else:
                lossbp = 0


            imageRT = imageRT / torch.amax(imageRT, dim=(-1, -2), keepdim=True)  # Normalize to [0, 1]
            scale = torch.sum(imageRT, dim=(-2, -1), keepdim=True) / torch.sum(outputRT, dim=(-2, -1), keepdim=True)
            outputRT = outputRT * scale

            loss = criterion(outputRT, imageRT) + lossbp
            currenttloss = currenttloss + loss.cpu().data.numpy()

    # print('loss:', loss.cpu().data.numpy())
    print('validloss:', currenttloss / len(validloader))

torch.save(model.state_dict(), model_save_dir)
print('model saved')
