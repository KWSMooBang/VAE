import os
import torch
import pytorch_lightning as pl

from typing import *
from torch import Tensor
from torch import optim
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision.transforms import v2
from models.base import BaseVAE

class VAE_experimenter(pl.LightningModule):
    def __init__(
            self,
            vae_model: BaseVAE,
            params: dict
    ) -> None:
        super(VAE_experimenter, self).__init__()

        self.model = vae_model
        self.params = params
        self.current_device = None
        self.hold_graph = False
        
        try: 
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_images, labels = batch
        self.current_device = real_images.device

        results = self.forward(real_images, labels=labels)
        train_loss = self.model.loss_function(*results, 
                                              M_N=self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)
        self.log_dict({key: val.item() for key, val in train_loss.items()},
                      sync_dist=True)
        
        return train_loss['loss']
    
    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_images, labels = batch
        self.current_device = real_images.device

        results = self.forward(real_images, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()},
                      sync_dist=True)
        
    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        test_images, test_labels = next(iter(self.trainer.datamodule.test_dataloader()))
        test_images = test_images.to(self.current_device)
        test_labels = test_labels.to(self.current_device)

        reconstructions = self.model.generate(test_images, labels=test_labels)
        utils.save_image(reconstructions.data,
                         os.path.join(self.logger.log_dir,
                                      'reconstructions',
                                      f'reconstructions_{self.logger.name}_epoch_{self.current_epoch}.png'),
                         normalizer=True,
                         nrow=12)
    
        try:
            samples = self.model.sample(144, self.current_device, labels=test_labels)
            utils.save_image(samples.cpu().data,
                             os.path.join(self.logger.log_dir,
                                          'samples',
                                          f'samples_{self.logger.name}_epoch_{self.current_device}.png'),
                             normalize=True,
                             nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):
        optimizers = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optimizers.append(optimizer)

        return optimizers

