import torch
import torch.nn as nn


def vae_loss(x_hat, x, mu, logvar):
    """
        reconstruction loss between the x_hat and x
    """
    recon_loss = nn.functional.mse_loss(x_hat, x)
    KL_divergence = 0.5 * torch.sum(mu.pow(2) + torch.exp(logvar) - 1 - logvar)
    return recon_loss + KL_divergence