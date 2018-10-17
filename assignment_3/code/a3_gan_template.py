import argparse
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 784
        #   Output non-linearity
        
        self.linear1 = nn.Linear(args.latent_dim, 128)
        
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(128, 256)
        
        self.bn3 = nn.BatchNorm1d(256)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.linear3 = nn.Linear(256, 512)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.lrelu4 = nn.LeakyReLU(0.2)
        self.linear4 = nn.Linear(512, 1024)
        
        self.bn5 = nn.BatchNorm1d(1024)
        self.lrelu5 = nn.LeakyReLU(0.2)
        self.linear5 = nn.Linear(1024, 784)
    
    def forward(self, z):
        # Generate images from z
        out = self.linear1(z)
        out = self.linear2(self.lrelu2(out))
        out = self.linear3(self.lrelu3(self.bn3(out)))
        out = self.linear4(self.lrelu4(self.bn4(out)))
        out = self.linear5(self.lrelu5(self.bn5(out)))
        out = F.tanh(out)
        
        return out.view(out.shape[0], 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        
        self.linear1 = nn.Linear(784, 512)
        
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(512, 256)
        
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.linear3 = nn.Linear(256, 1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, img):
        # return discriminator score for img
        
        out = self.linear1(img.view(img.shape[0], 784))
        out = self.linear2(self.lrelu2(out))
        out = self.linear3(self.lrelu3(out))
        out = self.sigmoid(out)
        
        return out


def interpolations(n_samples=20):
    samples = []
    for _ in range(n_samples):
        z1 = torch.FloatTensor(np.random.normal(0, 1, (args.latent_dim))).to(args.device)
        z2 = torch.FloatTensor(np.random.normal(0, 1, (args.latent_dim))).to(args.device)
        z = torch.empty(args.latent_dim, 9).to(args.device)
        
        for i in range(args.latent_dim):
            r = torch.linspace(z1[i], z2[i], 9)
            z[i] = r
        z.transpose_(0, 1)
        samples.append(z)
        
    return samples


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    criterion = torch.nn.BCELoss()
    interpolation_samples = interpolations(n_samples=100)
    
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            
            real_imgs = imgs.cuda()
            
            real_labels = torch.ones(imgs.shape[0], dtype=torch.float, device=args.device)
            fake_labels = torch.zeros(imgs.shape[0], dtype=torch.float, device=args.device)
            
            # Train Generator
            # ---------------
            z = torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))).to(args.device)
            fake_imgs = generator(z)
            fake_scores = discriminator(fake_imgs)
            
            g_loss = criterion(fake_scores, real_labels)
            g_loss.backward()
            optimizer_G.step()
            
            # Train Discriminator
            # -------------------
            fake_scores = discriminator(fake_imgs.detach())
            real_scores = discriminator(real_imgs)
            
            d_loss_fake = criterion(fake_scores, fake_labels)
            d_loss_real = criterion(real_scores, real_labels)
            d_loss = (d_loss_fake + d_loss_real) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            
            if batches_done % 100 == 0:
                print("Epoch: {}/{} | Batch: {}/{} | G_Loss: {:.4f} | "
                      "D_Loss {:.4f} | D(x): {:.2f} | D(G(z)): {:.2f}|".format(
                    epoch, args.n_epochs, i, len(dataloader), g_loss, d_loss,
                    real_scores.data.mean(), fake_scores.data.mean()))
            
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(fake_imgs.unsqueeze(1)[:25],
                           'samples_gan/{}.png'.format(batches_done),
                           nrow=5, normalize=True)

        if epoch % 10 == 0:
            generator.eval()
            for i in range(len(interpolation_samples)):
                z = interpolation_samples[i]
                imgs = generator(z)
                save_image(imgs.unsqueeze(1),
                           'interpolations_gan/epoch{}_n{}.png'.format(epoch, i),
                           nrow=9, normalize=True)
            generator.train()
                


def main():
    # Create output image directory
    os.makedirs('samples_gan', exist_ok=True)
    os.makedirs('interpolations_gan', exist_ok=True)
    
    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)
    
    # Initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    generator, discriminator = generator.to(args.device), discriminator.to(args.device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    checkpoint = torch.load("mnist_generator1.pt")
    generator.load_state_dict(checkpoint)
    
    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)
    
    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', default="cuda:0", type=str,
                        help='training device')
    args = parser.parse_args()
    
    main()
