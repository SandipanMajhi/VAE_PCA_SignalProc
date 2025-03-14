import torch
import torch.nn as nn

from tqdm import tqdm

class VAETrainer:
    def __init__(self, model, loss_fn, optimizer, epochs, model_name = "best_VAE.pt", checkpoint_path = "Checkpoints", device = "cpu"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.device = device
        self.model_name = model_name

        self.history = {
            "train" : {
                "loss" : [],
                "acc" : []
            },
            "val" : {
                "loss" : [],
                "acc" : []
            },
            "test" : {
                "loss" : [],
                "acc" : []
            }
        }


    def train(self, trainloader, valloader):
        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            trainloss = 0
            valloss = 0

            for batch in trainloader:
                self.optimizer.zero_grad()
                x, _ = batch[0].to(self.device), batch[1].to(self.device)
                x_recon, mu, logvar = self.model(x)
                loss = self.loss_fn(x_recon, x, mu, logvar)

                loss.backward()
                self.optimizer.step()
                trainloss += loss.item()

            self.model.eval()
            with torch.no_grad():
                for batch in valloader:
                    x, _ = batch[0].to(self.device), batch[1].to(self.device)
                    x_recon, mu, logvar = self.model(x)
                    loss = self.loss_fn(x_recon, x, mu, logvar)
                    valloss += loss.item()


            self.history["train"]["loss"].append(trainloss/len(trainloader))

            if len(self.history["val"]["loss"]) > 0:
                if valloss/len(valloader) < min(self.history["val"]["loss"]):
                    torch.save(self.model.state_dict(), f"{self.checkpoint_path}/{self.model_name}")

            self.history["val"]["loss"].append(valloss/len(valloader))

            print(f"Epoch {epoch}/{self.epochs} : trainloss = {self.history["train"]["loss"][-1]} valloss = {self.history["val"]["loss"][-1]}")


class RegressionTrainer:
    def __init__(self, model, loss_fn, optimizer, epochs, model_name = "best_regressor.pt", checkpoint_path="Checkpoints", device="cpu"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.device = device
        self.model_name = model_name

        self.history = {
            "train" : {
                "loss" : [],
                "acc" : []
            },
            "val" : {
                "loss" : [],
                "acc" : []
            },
            "test" : {
                "loss" : [],
                "acc" : []
            }
        }


    def train(self, trainloader, valloader):
        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            trainloss = 0
            valloss = 0

            for batch in trainloader:
                self.optimizer.zero_grad()
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                output = self.model(x)
                loss = self.loss_fn(output, y)

                loss.backward()
                self.optimizer.step()
                trainloss += loss.item()

            self.model.eval()
            with torch.no_grad():
                for batch in valloader:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    output = self.model(x)
                    loss = self.loss_fn(output, y)
                    valloss += loss.item()


            self.history["train"]["loss"].append(trainloss/len(trainloader))

            if len(self.history["val"]["loss"]) > 0:
                if valloss/len(valloader) < min(self.history["val"]["loss"]):
                    torch.save(self.model.state_dict(), f"{self.checkpoint_path}/{self.model_name}.pt")

            self.history["val"]["loss"].append(valloss/len(valloader))

            print(f"Epoch {epoch}/{self.epochs} : trainloss = {self.history["train"]["loss"][-1]} valloss = {self.history["val"]["loss"][-1]}")

    
    def predict(self, model, testloader):
        test_predictions = []
        testloss = 0
        model.eval()
        with torch.no_grad():
            for batch in testloader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                output = self.model(x)
                loss = self.loss_fn(output, y)
                testloss += loss.item()
                test_predictions.append(output)

        test_predictions = torch.cat(test_predictions, dim = 0)
        return test_predictions, testloss





                