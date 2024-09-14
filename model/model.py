import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from torch.nn import functional as F
from torch import nn



class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


    def forward(self, x):
        x = x.view(x.size(0), -1)
        embedding = self.encoder(x)
        x_hat = self.decoder(embedding)
        return x_hat
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x,y = train_batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x) ## loss from functional class
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x,y = val_batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat,x) ## you can define the number of ephocs
        self.log('val_loss', loss,  prog_bar=True, on_step=False, on_epoch=True)

    
    def test_step(self, val_batch, batch_idx):
        x,y = val_batch
        x = x.view(x.size(0), -1)
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat,x) ## you can define the number of ephocs
        self.log('test_loss', loss,  prog_bar=True, on_step=False, on_epoch=True)

    
    # def backward(self, trainer, loss, optimizer, optimizer_idx): ## if you need to
    #     loss.backward()
    
    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx): ## if you need to
    #     optimizer.step()




        

