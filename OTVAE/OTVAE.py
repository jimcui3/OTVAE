# Author: Shengyi Zong, Jiaheng Cui, College of Mathematics, Nankai University.

import os

import PIL.Image as Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from optimal_transport import *  # optimal_transport is the OT algorithm we designed

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs("OTVAE_img", exist_ok=True)


# VAE model.
class VAE(nn.Module):
    def __init__(self, dim_z):
        super(VAE, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer11 = nn.Linear(128 * 7 * 7, dim_z)
        self.layer12 = nn.Linear(128 * 7 * 7, dim_z)
        self.layer2 = nn.Linear(dim_z, 128 * 7 * 7)
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid())

    def encode(self, x):
        h = (self.layer1(x)).view((self.layer1(x)).size(0), -1)
        return self.layer11(h), self.layer12(h)

    def reparameterize(self, mu, logvar):
        # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        std = 0.5 * torch.exp(logvar)
        # N(mu, std^2) = N(0, 1) * std + mu
        z = torch.randn(std.size()) * std + mu
        return z

    def decode(self, z):
        h = self.layer2(z).view(self.layer2(z).size(0), 128, 7, 7)
        return self.layer3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar


#
def loss_function(recon_x, x, mu, logvar):
    """
    Loss function of VAE model training.
    Args:
        recon_x: generated image
        x: original image
        mu: latent mean of z
        logvar: latent log variance of z
    Returns:
        loss = reconstruction_loss + KL_divergence
    """

    BCE_loss = nn.BCELoss(reduction='sum')
    reconstruction_loss = BCE_loss(recon_x, x)
    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2)
    # KLD_ele = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = torch.sum(KLD_ele).mul_(-0.5)
    # print(reconstruction_loss, KL_divergence)

    return reconstruction_loss + KL_divergence


def VAE_OPTIMAL(pth_path='', k=2, epoches=100, dim_z=32):
    """
    Calculate the optimal transportation matrix pushing zeta to mu.
    Args:
        pth_path: a str, the path to load weights for VAE. If there is no input, VAE will start training from scratch.
        k: an integer, the number of nearest neighbors when generating new feature vectors.
        epoches: an integer, the number of training epoches of VAE.
        dim_z: an integer, the dimension of the bottleneck layer, a.e. the dimension of latent space Z.
    Returns:
        Pictures contains 100(10*10) generated numbers. We'll generate one picture after every epoch.
    """

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=int(10000/epoches), shuffle=True)

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

    vae = VAE(dim_z)
    if pth_path: # If we offered a path for the pre-trained weights, we can use it.
        weight = torch.load(pth_path)
        vae.load_state_dict(weight)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003)

    # Training
    def train(epoch):
        vae.train()
        all_loss = 0.
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to('cpu'), targets.to('cpu')
            # Train Discriminator
            mu, logvar = vae.encode(inputs)
            z=vae.reparameterize(mu,logvar)
            out=vae.decode(z)
            loss = loss_function(out, inputs, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_loss += loss.item()
            print('Current Epoch: {}, learned batch: {}/{}'.format(epoch,batch_idx,len(trainloader)))

        # fake_images = out.view(-1, 1, 28, 28)
        # save_image(fake_images, 'imgs/fake_images-{}.png'.format(epoch + 1))

    def evaluate(epoch):
        vae.eval()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to('cpu'), targets.to('cpu')
            mu, logvar = vae.encode(inputs)
            z = vae.reparameterize(mu, logvar)
            z = z.detach().numpy()
            zeta = np.random.rand(len(testloader),dim_z) # len(testloader) is the number of images to be generated.
            phi = Generate(zeta, z, 1, k) # phi is the generated feature vectors.
            phi = torch.from_numpy(phi)
            phi = phi.type(torch.float32)
            out = vae.decode(phi) # Decode phi into images.
            fake_images = out.view(-1, 1, 28, 28)
            save_image(fake_images, 'OTVAE_img/OTVAE-images-{}.png'.format(batch_idx))
            print('Current Epoch: {}, generating pictures: {}/{}'.format(epoch,batch_idx,len(testloader)))

        # Read all the generated images and put them together into one big picture.
        # Reference: https://blog.csdn.net/ahaotata/article/details/84027000
        # Author: ahaotata, CSDN.
        print('Current Epoch: {}, Image stitching...'.format(epoch))
        IMAGES_PATH = 'OTVAE_img'
        IMAGES_FORMAT = ['.png']
        IMAGE_SIZE = 28
        IMAGE_ROW = 10
        IMAGE_COLUMN =10
        IMAGE_SAVE_PATH = IMAGES_PATH + '/' + str(epoch) + '.jpg'

        image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
                       os.path.splitext(name)[1] == item]

        to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))

        for y in range(1, IMAGE_ROW + 1):
            for x in range(1, IMAGE_COLUMN + 1):
                from_image = Image.open(IMAGES_PATH + '/' + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))

        to_image.save(IMAGE_SAVE_PATH)

        for name in image_names:
            os.remove(IMAGES_PATH + '/' + name)

        print('Current Epoch: {} Done!'.format(epoch))

    for epoch in range(epoches):
        train(epoch)
        evaluate(epoch)
        torch.save(vae.state_dict(), 'checkpoint.pth')


# You can change the hyperparameters here.
VAE_OPTIMAL(pth_path='', k=2, epoches=100, dim_z=32)
