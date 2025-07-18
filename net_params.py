from torchsummary import summary
from Model import RTFnet2, TM, RTUnet, RTMnet
from vae import VAE
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
RTMnet = RTFnet2().cuda()
MLP1 = TM(input_size=256, output_size=48).cuda()
MLP2 = TM(input_size=256, output_size=256).cuda()
CNN = RTUnet().cuda()
VAE = VAE().cuda()
# summary(RTMnet, input_size=(1, 256, 256))
summary(VAE, input_size=(1, 256, 256))
# summary(MLP2, input_size=(1, 256, 256))
# summary(CNN, input_size=(1, 256, 256))
# print(count_params(RTMnet))