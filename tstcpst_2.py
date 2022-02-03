#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

dataroot = "Desktop/wikiart/portraits/"
n_dl_thrds = 2
batch_size = 128
image_size = 64
#num channels = number of color components per pixel
nc = 3
# size of input latent vector
nz = 100
# feature map size for generator
ngf = 64
# feature map size for discriminator
ndf = 64
num_epochs = 2
# learning rate
lr = 0.0002
# see Adam's optimizers for more 
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


#begin classes of generator and discriminator. Basic for now but will add to later.
class gen(nn.Module):
    def __init__(self, ngpu):
        super(gen, self).__init__()
        self.ngpu = ngpu
        #set up layers
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
class disc(nn.Module):
    def __init__(self, ngpu):
        super(disc, self).__init__()
        self.ngpu = ngpu
        #set up layers
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
#end classes

#sample normal distribution to initilize weights with random values
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
def run():
	random.seed(2)
	torch.manual_seed(2)
	dataset = dset.ImageFolder(root=dataroot,
	                           transform=transforms.Compose([
	                               transforms.Resize(image_size),
	                               transforms.CenterCrop(image_size),
	                               transforms.ToTensor(),
	                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	                           ]))
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
	                                         shuffle=True, num_workers=n_dl_thrds)
	device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
	print(device)
	real_batch = next(iter(dataloader))
	plt.figure(figsize=(8,8))
	plt.axis("off")
	plt.title("Training Images")
	plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
	plt.show()
	netG = gen(ngpu).to(device)
	if ((device.type == 'cuda') and (ngpu > 1)):
		netG = nn.DataParallel(netG, list(range(ngpu)))
	netG.apply(weights_init)
	print(netG)
	netD = disc(ngpu).to(device)
	if ((device.type == 'cuda') and (ngpu > 1)):
		netD = nn.DataParallel(netD, list(range(ngpu)))
	netD.apply(weights_init)
	print(netD)
	criterion = nn.BCELoss()
	fixed_noise = torch.randn(64, nz, 1, 1, device=device)
	real_label = 1.
	fake_label = 0.
	optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
	img_list = []
	G_losses = []
	D_losses = []
	iters = 0
	for epoch in range(num_epochs):
	    for i, data in enumerate(dataloader, 0):
	        netD.zero_grad()
	        real_cpu = data[0].to(device)
	        b_size = real_cpu.size(0)
	        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
	        output = netD(real_cpu).view(-1)
	        errD_real = criterion(output, label)
	        errD_real.backward()
	        D_x = output.mean().item()
	        noise = torch.randn(b_size, nz, 1, 1, device=device)
	        fake = netG(noise)
	        label.fill_(fake_label)
	        output = netD(fake.detach()).view(-1)
	        errD_fake = criterion(output, label)
	        errD_fake.backward()
	        D_G_z1 = output.mean().item()
	        errD = errD_real + errD_fake
	        optimizerD.step()
	        netG.zero_grad()
	        label.fill_(real_label)  # fake labels are real for generator cost
	        output = netD(fake).view(-1)
	        errG = criterion(output, label)
	        errG.backward()
	        D_G_z2 = output.mean().item()
	        optimizerG.step()
	        if i % 50 == 0:
	            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
	                  % (epoch, num_epochs, i, len(dataloader),
	                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

	        # Save Losses for plotting later
	        G_losses.append(errG.item())
	        D_losses.append(errD.item())
	        # Check how the generator is doing by saving G's output on fixed_noise
	        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
	            with torch.no_grad():
	                fake = netG(fixed_noise).detach().cpu()
	            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
	        iters += 1
	real_batch = next(iter(dataloader))

	# Plot the real images
	plt.figure(figsize=(15,15))
	plt.subplot(1,2,1)
	plt.axis("off")
	plt.title("Real Images")
	plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
	# Plot the fake images from the last epoch
	plt.subplot(1,2,2)
	plt.axis("off")
	plt.title("Fake Images")
	plt.imshow(np.transpose(img_list[-1],(1,2,0)))
	plt.show()
if __name__ == '__main__':
    run()

