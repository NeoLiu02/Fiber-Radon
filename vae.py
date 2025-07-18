import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Mydataset
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import trange
from torch.utils.data import DataLoader
from torch_radon import Radon
import numpy as np

# python vae.py
LEARNING_RATE = 1e-4
BATCH_SIZE = 100
NUM_EPOCHS = 20
LATENT_DIM = 64
ALPHA = 0.2
BETA = 2.0
dataset = 'mnist'
resume = True  # Set to True if you want to resume training from a saved model

class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64, initial_dim=4): # remember to change the initial_dim in the decoder if you change it here
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, initial_dim, kernel_size=4, stride=2, padding=1), # -> 128x128
            nn.ReLU(),
            nn.Conv2d(initial_dim, initial_dim*2, kernel_size=4, stride=2, padding=1),      # -> 64x64
            nn.ReLU(),
            nn.Conv2d(initial_dim*2, initial_dim*4, kernel_size=4, stride=2, padding=1),      # -> 32x32
            nn.ReLU(),
            nn.Conv2d(initial_dim*4, initial_dim*8, kernel_size=4, stride=2, padding=1),     # -> 16x16
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(initial_dim*8 * 16 * 16, latent_dim)
        self.fc_log_var = nn.Linear(initial_dim*8 * 16 * 16, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, initial_dim*8 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(initial_dim*8, initial_dim*4, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(initial_dim*4, initial_dim*2, kernel_size=4, stride=2, padding=1),  # -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(initial_dim*2, initial_dim, kernel_size=4, stride=2, padding=1),   # -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(initial_dim, in_channels, kernel_size=4, stride=2, padding=1), # -> 256x256
            nn.Sigmoid()  
        )

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 4*8, 16, 16) 
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

def vae_loss_function(recon_x, x, mu, log_var, beta=1.0, BATCH_SIZE=1):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')/BATCH_SIZE
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())/BATCH_SIZE
    return recon_loss, kl_div, recon_loss + beta * kl_div

if __name__ == "__main__":
    speckle_path = ['data/speckle/' + dataset + '/RT/train.npy', 'data/speckle/' + dataset + '/RT/valid.npy',
                    'data/speckle/' + dataset + '/RT/test1.npy', 'data/speckle/' + dataset + '/RT/test2.npy',
                    'data/speckle/' + dataset + '/RT/test3.npy', 'data/speckle/' + dataset + '/RT/train_3000.npy']
    image_path = ['data/origin/' + dataset + '/RT/train.npy', 'data/origin/' + dataset + '/RT/valid.npy',
                'data/origin/' + dataset + '/RT/test1.npy', 'data/origin/' + dataset + '/RT/test2.npy',
                'data/origin/' + dataset + '/RT/test3.npy', 'data/origin/' + dataset + '/RT/train_3000.npy']

    train_speckle_path = speckle_path[5]  # Load the 3000 samples
    train_image_path = image_path[5]  # Load the 3000 samples
    valid_speckle_path = speckle_path[3]  # Load the validation set
    valid_image_path = image_path[3]  # Load the validation set

    trainset = Mydataset(train_speckle_path, train_image_path)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    validset = Mydataset(valid_speckle_path, valid_image_path)
    validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    model = VAE(in_channels=1, latent_dim=LATENT_DIM).cuda()
    if resume:
        model.load_state_dict(torch.load("model/mnist/VAE_mnistRT_train3000.pth"))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S') # Format: YearMonthDay-HourMinuteSecond
    log_dir = f'runs_vae/training_run_{timestamp}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    RT = Radon(256, np.linspace(0, np.pi, 256, endpoint=False), clip_to_circle=True)

    global_step = 0
    for epoch in trange(NUM_EPOCHS):
        model.train() 
        total_train_loss = 0

        for batch in trainloader:
            global_step += 1
            imagesRT = batch['label'].unsqueeze(1).to(torch.float32).cuda()
            
            imagesRT = imagesRT / torch.amax(imagesRT, dim=(-2, -1), keepdim=True)  # Normalize to [0, 1]
            recon_imagesRT, mu, log_var = model(imagesRT)
            RTloss = F.mse_loss(recon_imagesRT, imagesRT, reduction='sum')/BATCH_SIZE

            images = RT.backprojection(RT.filter_sinogram(imagesRT))
            recon_images = RT.backprojection(RT.filter_sinogram(recon_imagesRT))
            images = images / torch.amax(images, dim=(-2, -1), keepdim=True)
            recon_images = recon_images / torch.amax(recon_images, dim=(-2, -1), keepdim=True)
            recon_loss, kl_div, loss = vae_loss_function(recon_images, images, mu, log_var, beta=BETA, BATCH_SIZE=BATCH_SIZE)
            loss += ALPHA * RTloss  # Add Radon transform loss to the total loss
            
            if global_step % 100 == 0:
                writer.add_scalar('Loss/Train/RT_loss', RTloss.item(), global_step)
                writer.add_scalar('Loss/Train/Reconstruction', recon_loss.item(), global_step)
                writer.add_scalar('Loss/Train/KL_Divergence', kl_div.item(), global_step)
                writer.add_image('Image/Train/Original_RT', imagesRT[0], global_step)
                writer.add_image('Image/Train/Reconstructed_RT', recon_imagesRT[0], global_step)
                writer.add_image('Images/Train/Original', images[0], global_step)
                writer.add_image('Images/Train/Reconstructed', recon_images[0], global_step)

            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step() 
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(trainloader.dataset)
        print(f"====> Epoch: {epoch+1} Average loss: {avg_train_loss:.4f} <====")

        model.eval() 
        KL_valid_loss = 0
        recon_valid_loss = 0
        RT_valid_loss = 0
        total_valid_loss = 0
        with torch.no_grad():
            for batch in validloader:
                imagesRT = batch['label'].unsqueeze(1).to(torch.float32).cuda()
                imagesRT = imagesRT / torch.amax(imagesRT, dim=(-2, -1), keepdim=True)  # Normalize to [0, 1]
                recon_imagesRT, mu, log_var = model(imagesRT)
                RTloss = F.mse_loss(recon_imagesRT, imagesRT, reduction='sum')/BATCH_SIZE

                images = RT.backprojection(RT.filter_sinogram(imagesRT))
                recon_images = RT.backprojection(RT.filter_sinogram(recon_imagesRT))
                images = images / torch.amax(images, dim=(-2, -1), keepdim=True)
                recon_images = recon_images / torch.amax(recon_images, dim=(-2, -1), keepdim=True)
                recon_loss, kl_div, loss = vae_loss_function(recon_images, images, mu, log_var, beta=BETA, BATCH_SIZE=BATCH_SIZE)
                loss += ALPHA * RTloss  # Add Radon transform loss to the total loss
                
                KL_valid_loss += kl_div.item()
                recon_valid_loss += recon_loss.item()
                RT_valid_loss += RTloss.item()
                total_valid_loss += loss.item()

            writer.add_scalar('Loss/Valid/RT_loss', RTloss.item()/len(validloader.dataset), global_step)
            writer.add_scalar('Loss/Valid/Reconstruction', recon_loss.item()/len(validloader.dataset), global_step)
            writer.add_scalar('Loss/Valid/KL_Divergence', kl_div.item()/len(validloader.dataset), global_step)
            writer.add_image('Image/Valid/Original_RT', imagesRT[0], global_step)
            writer.add_image('Image/Valid/Reconstructed_RT', recon_imagesRT[0], global_step)
            writer.add_image('Images/Valid/Original', images[0], global_step)
            writer.add_image('Images/Valid/Reconstructed', recon_images[0], global_step)

            avg_valid_loss = total_valid_loss / len(validloader.dataset)
            print(f"====> Epoch: {epoch+1} Average valid loss: {avg_valid_loss:.4f} <====")



    torch.save(model.state_dict(), "model/mnist/VAE_mnistRT_train3000.pth")