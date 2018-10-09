import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision.utils import make_grid
from torchvision.utils import save_image

import os

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=500, z_dim=20):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.to_hidden = nn.Linear(input_dim, hidden_dim)
        self.to_mean = nn.Linear(hidden_dim, z_dim)
        self.to_logvar = nn.Linear(hidden_dim, z_dim)
        
        # initialize at N(0, 0.01)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and logvar with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        
        h = F.relu(self.to_hidden(input))
        mean = self.to_mean(h)
        logvar = self.to_logvar(h)

        return mean, logvar


class Decoder(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=500, z_dim=20):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.to_hidden = nn.Linear(z_dim, hidden_dim)
        self.to_output = nn.Linear(hidden_dim, input_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns output with shape [batch_size, 784].
        """
        
        h = F.relu(self.to_hidden(input))
        output = self.to_output(h)

        return output


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim=hidden_dim, z_dim=z_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim, z_dim=z_dim)
        
    def reparameterize(self, mu, logvar):
        noise = torch.randn_like(mu)
        z = noise * logvar + mu
        return z
    
    def compute_elbo(self, input, output, mu, logvar):
        bce = F.binary_cross_entropy_with_logits(output, input, reduction='elementwise_sum')
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (bce + kl) / input.shape[0]

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        
        mu, logvar = self.encoder(input)
        z = self.reparameterize(mu, logvar)
        y = self.decoder(z)
        
        average_negative_elbo = self.compute_elbo(input, y, mu, logvar)
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        mu = torch.randn(n_samples, 20).to(device)
        std = torch.randn(n_samples, 20).to(device)
        z = self.reparameterize(mu, std)
        y = self.decoder(z)
        
        sampled_ims, im_means = torch.sigmoid(y.view(n_samples, 28, 28)), mu
        return sampled_ims, im_means
    
    def sample_manifold(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        mu = torch.randn(n_samples, 20).to(device)  # TODO uniform
        std = torch.randn(n_samples, 20).to(device)  # TODO zeros
        z = self.reparameterize(mu, std)
        y = self.decoder(z)

        sampled_ims, im_means = y.view(n_samples, 28, 28), mu
        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    elbos = []
    for i, batch in enumerate(data):
        optimizer.zero_grad()
        
        batch = batch.to(device)
        batch = batch.view(batch.shape[0], -1)
        elbo = model(batch)
        elbos.append(elbo.item())
        
        elbo.backward()
        optimizer.step()
        
        
    average_epoch_elbo = sum(elbos) / len(elbos)

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    if not os.path.exists("samples"):
        os.makedirs("samples")
        
    data = bmnist(batch_size=ARGS.batch_size)[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.lr_rate)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        samples, samples_mu = model.sample(64)
        grid = make_grid(samples.unsqueeze(1))
        save_image(grid, f"samples/{epoch}.png")
        kkkk = 0

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='size of each batch')
    parser.add_argument('--lr_rate', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--device', default="cuda:0", type=str,
                        help='training device')

    ARGS = parser.parse_args()

    device = torch.device(ARGS.device)
    
    main()
