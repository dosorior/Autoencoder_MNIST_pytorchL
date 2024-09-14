
import argparse
from model.model import LitAutoEncoder
import lightning as L
from torchvision.datasets import MNIST
import os
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Training of an Autoencoder on scratch using the database MINIST on pytorch lightning')

parser.add_argument('--batch',  '-b',
                    dest="limit_batch",
                    type=int,
                    help = 'batch limit on training',
                    default=100)

parser.add_argument('--check_val_e',  '-ch',
                    dest="check_val_every_n_epoch",
                    type=int,
                    help = 'val on training',
                    default=5)

parser.add_argument('--max_e',  '-e',
                    dest="max_epochs",
                    type=int,
                    help = 'number of epochs',
                    default=50)

parser.add_argument('--output_ckp',  '-o',
                    dest="root_dir",
                    type=str,
                    help = 'saving checkpoint',
                    default="./checkpoints/")

args = parser.parse_args()


model = LitAutoEncoder()
trainer = L.Trainer(limit_train_batches=args.limit_batch, max_epochs=args.max_epochs, check_val_every_n_epoch=args.check_val_every_n_epoch, logger=True, default_root_dir=args.root_dir) ## you can include accelerator="gpu"/"tpu", devices=4
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5)]) ##Normalize data
dataset = MNIST(os.getcwd(), download=True, transform=transform)
train_loader = DataLoader(dataset)
dataset_val = MNIST(os.getcwd(), train=False, transform=transform)
val_loader = DataLoader(dataset_val)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader) ## Needed of DataLoaders here
