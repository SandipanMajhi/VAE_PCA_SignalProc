import torch
import torch.nn as nn

import numpy as np

from abc import ABC, abstractmethod


class BaseModel:
    
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def inverse_transform(self):
        pass
      


class PCA(BaseModel):
    def __init__(self, n_components, device = "cpu"):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.device = device

    def fit(self, x):
        """
            Expects x to be a Float Tensor
        """
        x = torch.FloatTensor(x)
        self.mean = torch.mean(x, dim = 0).to(self.device)
        x_centered = x - self.mean
        cov_mat = torch.cov(x_centered.T).to(self.device)

        eig_value, eig_vect = torch.linalg.eigh(cov_mat)

        sorted_idx = torch.argsort(eig_value, descending=True)
        eig_value = eig_value[sorted_idx]
        eig_vect = eig_vect[:, sorted_idx]

        self.components = eig_vect[:, :self.n_components]
        self.components = self.components.to(self.device)

    def transform(self, x):
        """
            Expects x to be a Float Tensor
        """
        x = torch.FloatTensor(x)
        x_centered = x - self.mean
        return torch.matmul(x_centered, self.components).to(self.device)
    
    def inverse_transform(self, x_reduced):
        """
            x_reduced : torch.FloatTensor
        """
        reconstructed_x =  torch.matmul(x_reduced, self.components.T) + self.mean
        reconstructed_x = reconstructed_x.to(self.device)
        return reconstructed_x
    


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        ### Encoder ###
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        ### Decoder ###
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparametarize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def decode(self, x):
        h  = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametarize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    

class RegressionNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        
        self.fc_1 = nn.Linear(in_features, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 8)
        self.fc_4 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.relu(self.fc_1(x))
        z = self.relu(self.fc_2(z))
        z = self.relu(self.fc_3(z))
        z = self.fc_4(z)
        z = self.sigmoid(z)
        return z





