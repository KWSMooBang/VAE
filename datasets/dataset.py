import os 
import torch

from typing import *
from torch import Tensor
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.datasets import CelebA

class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

class MyCelebA(CelebA):
    def _check_integrity(self) -> bool:
        return True

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """ 
    def __init__(self,
                data_path: str, 
                split: str,
                transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"
        self.transforms = transform
        images = sorted([f for f in self.data_dir.iterdir() if f.suffix == 'jpg'])

        self.images = images[:int(len(images) * 0.75)] if split == 'train' else \
            images[int(len(images) * 0.75):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = default_loader(self.images[index])

        if self.transforms is not None:
            image = self.transforms(image)

        return image, 0.0

class VAEDataset(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super(VAEDataset, self).__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.CenterCrop(148),
            v2.Resize(self.patch_size),
            v2.ToImage()
        ])

        val_transforms = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.CenterCrop(148),
            v2.Resize(self.patch_size),
            v2.ToImage()
        ])

        self.train_dataset = MyCelebA(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )

        self.val_dataset = MyCelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )