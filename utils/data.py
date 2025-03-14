import torch
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Model.model import PCA

class SignalData:
    def __init__(self, csv_file = "Data/TASK-ML-INTERN.csv", device = "cpu", **kwargs):
        self.csv_file = csv_file
        self.x, self.y = self.load()

        self.device = device

    def project_data(self, mode, model = None, **kwargs):
        """
            If mode == "VAE, it assumes the VAE is trained already
        """
        if mode == "PCA":
            model = PCA(n_components=kwargs.get("n_components"))
            model.fit(self.x)
            return model.transform(self.x)
        
        else: 
            latent_vectors = []
            model.eval()
            with torch.no_grad():
                for x in self.x:
                    x = torch.FloatTensor(x)
                    x = x.unsqueeze(dim = 0).to(self.device)
                    mu_x, logvar_x = model.encode(x)
                    latent_vectors.append(mu_x)

            latent_vectors = torch.cat(latent_vectors, dim = 0)
            latent_vectors = latent_vectors.cpu().numpy()

            return latent_vectors


    def load(self):
        data = pd.read_csv(self.csv_file)
        data = data.drop(columns=["hsi_id"])
        x = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values
        self.target_scaling = MinMaxScaler()
        y = y.reshape(-1,1)
        # y = self.target_scaling.fit_transform(y)

        return x, y

    def split_data(self, split_size = 0.8):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, train_size=split_size)
        self.target_scaling.fit(y_train)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
        
        y_train = self.target_scaling.transform(y_train)
        y_val = self.target_scaling.transform(y_val)
        y_test = self.target_scaling.transform(y_test)

        self.x_train = torch.FloatTensor(x_train).to(self.device)
        self.y_train = torch.FloatTensor(y_train).to(self.device)

        self.x_val = torch.FloatTensor(x_val).to(self.device)
        self.y_val = torch.FloatTensor(y_val).to(self.device)

        self.x_test = torch.FloatTensor(x_test).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)
    

    def create_dataloader(self, batch_size = 16):
        train_dataset = TensorDataset(self.x_train, self.y_train)
        val_dataset = TensorDataset(self.x_val, self.y_val)
        test_dataset = TensorDataset(self.x_test, self.y_test)

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

        return trainloader, valloader, testloader
        


    




        


