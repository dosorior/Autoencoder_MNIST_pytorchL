from model.model import LitAutoEncoder
import lightning as L
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os
from callbacks.test_callback import MyTestCallback
import argparse

parser = argparse.ArgumentParser(description='Testing')

parser.add_argument('--output_ckp',  '-o',
                    dest="root_dir",
                    type=str,
                    help = 'dir to load the checkpoint',
                    default="./checkpoints/lightning_logs/version_0/checkpoints/epoch=49-step=5000.ckpt")

parser.add_argument('--output',  '-out',
                    dest="root_out",
                    type=str,
                    help = 'output to save the images',
                    default="test_images")

args = parser.parse_args()

# load checkpoint
checkpoint = args.root_dir
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5)]) ##Normalize data
dataset_val = MNIST(os.getcwd(), download=True, train=False, transform=transform)
val_loader = DataLoader(dataset_val, batch_size=25)

test_callback = MyTestCallback(val_loader,path_save=args.root_out,every_n_batches=1)
trainer = L.Trainer(callbacks=[test_callback])
trainer.test(autoencoder,dataloaders=val_loader)


