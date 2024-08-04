import torch
import torch.nn.functional as F

from typing import *
from torch import nn
from torch import Tensor
from base import BaseVAE

class VanillaVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        modules = []

        for hidden_dim in hidden_dims:
            module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels=hidden_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(num_features=hidden_dim),
                nn.LeakyReLU()
            )
            modules.append(module)
            in_channels = hidden_dim
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(in_features=hidden_dims[-1]*4, out_features=latent_dim)
        self.fc_var = nn.Linear(in_features=hidden_dims[-1]*4, out_features=latent_dim)

        # Build Decoder
        modules = []

        self.decoer_input = nn.Linear(in_features=latent_dim, out_features=hidden_dims[-1]*4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            module = nn.Sequential(
                nn.ConvTranspose2d(in_channels=hidden_dims[i], out_channels=hidden_dims[i+1],
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU
            )
            modules.append(module)

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[-1], out_channels=hidden_dims[-1],
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encoder(self, x: Tensor) -> List[Tensor]:
        """
        parameters:
            x: (Tensor) Input tensor to encoder expressed as x in paper [B, C, H, W] 
        returns:
            mu, log_var: (Tensor) latent codes expressed as mu and log(sigma^2) in paper [N, latent_dim]
        """
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)

        mu = self.fc_mu(h)
        log_var = self.fc_var(h)

        return [mu, log_var]
    
    def decode(self, z: Tensor) -> Tensor:
        """
        parameters:
            z: (Tensor) [B, latent_dim]
        returns:
            x: (Tensor) [B, C, H, W]
        """
        h = self.decoder_input(z)
        h = h.view(-1, 512, 2, 2)
        h = self.decoder(h)
        x = self.final_layer(h)
        return x

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        parameters:
            mu: (Tensor) mean of latent gaussian [B, latent_dim]
            log_var: (Tensor) standard deviation of the latent gaussian [B, latent_dim]
        returns:
            z: (Tensor) latent variable [B, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return mu + std * eps

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Compute the VAE loss function
        KL(N(mu, sigma), N(0, 1)) = log(1/sigma) + (sigma^2 + mu^2)/2 - 1/2
        """
        reconstructions = args[0]
        x = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_n']
        reconstruction_loss = F.mse_loss(reconstructions, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = reconstruction_loss + kld_weight * kld_loss
        return {
            'loss': loss,
            'reconstruction_loss': reconstruction_loss.detach(),
            'kld_loss': -kld_loss.detach()
        }

    def sample(self, num_samples:int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and 
        return the corresponding image space map
        parameters:
            num_samples: (int) number of samples
            current_device: (int) device to run the model
        returns:
            samples: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, return the reconstructed image
        parameters:
            x: (Tensor) input image [B, C, H, W]
        returns:
            reconstructed image: (Tenosr) [B, C, H, W]
        """

        return self.forward(x)[0]