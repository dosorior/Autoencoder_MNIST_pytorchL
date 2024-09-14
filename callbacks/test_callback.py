
from lightning.pytorch.callbacks import Callback
import os
from torchvision.utils import make_grid, save_image
import random


class MyTestCallback(Callback):

    def __init__(self, dataset, path_save="test_images", every_n_batches=1) -> None:
        super().__init__()
        self.dataset = dataset
        self.path_save = path_save
        self.every_n_batches = every_n_batches
        os.makedirs(path_save, exist_ok=True)


    def on_test_start(self, trainer, pl_module):
        print("Testing has started!")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        print(f"Finished testing batch {batch_idx}.")

    def on_test_epoch_end(self, trainer, pl_module):
        batch_list = []
        pl_module.eval()
        for batch in self.dataset:
            x, _ = batch
            # print(f"Size of the tensor of input is{x.size()}")
            r = pl_module(x)
            batch_list.append(r)

        im = random.choice(batch_list)
        im = im.view(-1, 1, 28, 28)
        for t, m, s in zip(im.unbind(dim=1), [0.5], [0.5]):
            t.mul_(s).add_(m)
        
        test_grid = make_grid(im, nrow=5) 
        test_save_path = os.path.join(self.path_save, "test.png")
        save_image(test_grid, test_save_path)
        print(f"Saved batch images to {test_save_path}")
        print("Testing has finished!")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()


        pl_module.train()
        print("Testing has finished!")