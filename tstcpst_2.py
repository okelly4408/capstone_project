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

dataroot = "../Desktop/abstract/small_abstract"
n_dl_thrds = 3
batch_size = 64
image_size = 128
#num channels = number of color components per pixel
nc = 3
# size of input latent vector
nz = 100
# feature map size for generator
ngf = 32
# feature map size for discriminator
ndf = 32
num_epochs = 1000
# learning rate
lr = 0.0001
# see Adam's optimizers for more 
beta1 = 0.5
beta2 = 0.999
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

noise_level = 0.1


#begin classes of generator and discriminator. Basic for now but will add to later.
class gen(nn.Module):
    def __init__(self, ngpu):
        super(gen, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.utils.spectral_norm(nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False)),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf * 16),

            # state size. (ngf*16) x 4 x 4
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf * 8),

            # state size. (ngf*8) x 8 x 8
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf * 4),
            # state size. (ngf*4) x 16 x 16 
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf * 2),
            # state size. (ngf*2) x 32 x 32
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf),
            # state size. (ngf) x 64 x 64
            nn.utils.spectral_norm(nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False)),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)
class disc(nn.Module):
    def __init__(self, ngpu):
        super(disc, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf * 2),
            
            # state size. (ndf*2) x 32 x 32
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf * 4),
            
            # state size. (ndf*4) x 16 x 16 
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf * 8),
            
            # state size. (ndf*8) x 8 x 8
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf * 16),
            
            # state size. (ndf*16) x 4 x 4
            nn.utils.spectral_norm(nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False)),
            nn.Sigmoid()
            # state size. 1
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
#returns generator network only for now
def train_GAN(device, dataloader):
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
	fixed_noise = torch.randn(128, nz, 1, 1, device=device)
	real_label = 1.
	fake_label = 0.
	optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
	optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
	img_list = []
	img_list_2 = []
	G_losses = []
	D_losses = []
	iters = 0
	for epoch in range(num_epochs):
	    for i, data in enumerate(dataloader, 0):

	    	#Update discriminator
	    	#training it with a batch from training set
	        netD.zero_grad()
	        real_cpu = data[0].to(device)
	        b_size = real_cpu.size(0)
	        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
	        #pass batch of real images
	        output = netD(real_cpu).view(-1)
	        #calculate loss
	        errD_real = criterion(output, label)
	        #calculate gradients
	        errD_real.backward()
	        D_x = output.mean().item()

	        #train discriminator with output from generator
	        noise = torch.randn(b_size, nz, 1, 1, device=device)
	        #generates batch of fake images
	        fake = netG(noise)
	        label.fill_(fake_label)
	        #classify the generated images
	        output = netD(fake.detach()).view(-1)
	        #discriminators loss on generated images
	        errD_fake = criterion(output, label)
	        #gradients for discriminator and sum with gradients from real batch
	        errD_fake.backward()
	        D_G_z1 = output.mean().item()
	        errD = errD_real + errD_fake
	        #updates disc
	        optimizerD.step()

	        #Update generator
	        netG.zero_grad()
	        label.fill_(real_label)  # fake labels are real for generator cost
	        #forward pass fake images again through discriminator
	        output = netD(fake).view(-1)
	        #calculates gen loss on disc pass
	        errG = criterion(output, label)
	        #gradients for gen
	        errG.backward()
	        D_G_z2 = output.mean().item()
	        #update generator
	        optimizerG.step()
	        if i % 8 == 0:
	            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
	                  % (epoch, num_epochs, i, len(dataloader),
	                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

	        # Save Losses for plotting later
	        G_losses.append(errG.item())
	        D_losses.append(errD.item())
	        # Check how the generator is doing by saving G's output on fixed_noise
	        if (iters % 25 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
	            with torch.no_grad():
	                fake = netG(fixed_noise).detach().cpu()
	            vutils.save_image(vutils.make_grid(fake[0], padding=2, normalize=True), 'nnimages_prog/'+str(iters)+'image' + str(random.random()) + '.jpg')
	        iters += 1
	torch.save({
		'netG_state_dict': netG.state_dict(),
		'netD_state_dict': netD.state_dict(),
		'optimizerG_state_dict': optimizerG.state_dict(),
		'optimizerD_state_dict': optimizerD.state_dict()
		}, 'model_1000.tar')
	return netG
def run():
	dataset = dset.ImageFolder(root=dataroot,
	                           transform=transforms.Compose([
	                               transforms.Resize(image_size),
	                               transforms.CenterCrop(image_size),
	                               transforms.ToTensor(),
	                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	                           ]))
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
	                                         shuffle=True, num_workers=n_dl_thrds)
	dataloader_2 = torch.utils.data.DataLoader(dataset, batch_size=32,
	                                         shuffle=True, num_workers=n_dl_thrds)
	device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
	real_batch = next(iter(dataloader_2))
	plt.figure(figsize=(16,16))
	plt.axis("off")
	plt.title("Training Images")
	plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:128], padding=2, normalize=True).cpu(),(1,2,0)))
	plt.show()
	generator = train_GAN(device, dataloader)
	torch.save(generator.state_dict(), 'gen_1_1000.pt')
	for i in range(128):
		noise = torch.randn(128, nz, 1, 1, device=device)
		nn_img = generator(noise).detach().cpu()
		vutils.save_image(vutils.make_grid(nn_img[0], padding=2, normalize=True), 'nnimages_2/z_'+str(i)+'image' + str(random.random()) + '.jpg')
	# Plot the fake images from the last epoch
	#plt.subplot(1,2,2)
	#plt.axis("off")
	#plt.title("Fake Images")
	#plt.imshow(np.transpose(img_list[-1],(1,2,0)))
	#fakes_2 = img_list[-1]
	#for img_nn in fakes_2:
	#	vutils.save_image(vutils.make_grid(img_nn, padding=2, normalize=True), "../nnimages_2/image" + str(random.random()) + ".jpg")
	#plt.show()
if __name__ == '__main__':
    run()



