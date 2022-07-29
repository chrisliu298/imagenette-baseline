import os
from copy import deepcopy as c

import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_data_path = config.data_path + "train"
        self.test_data_path = config.data_path + "val"
        self.num_workers = os.cpu_count()
        # calculate mean and std
        train_dataset = ImageFolder(
            self.train_data_path, transform=transforms.CenterCrop(160)
        )
        train_images = np.stack(
            [np.array(train_dataset[i][0]) for i in range(len(train_dataset))]
        )
        means = (np.mean(train_images, axis=(0, 1, 2)) / 255.0).round(4).tolist()
        stds = (np.std(train_images, axis=(0, 1, 2)) / 255.0).round(4).tolist()
        # define transforms
        self.transforms_train = []
        self.transforms_test = []
        base_transfroms = [
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]
        if config.data_augmentation:
            self.transforms_train.append(transforms.RandomCrop(160, padding=20))
            self.transforms_train.append(transforms.RandomHorizontalFlip())
        self.transforms_train.extend(base_transfroms)
        self.transforms_test.extend(base_transfroms)
        del train_dataset

    def prepare_data(self):
        # download data
        self.train_dataset = ImageFolder(
            self.train_data_path, transform=transforms.Compose(self.transforms_train)
        )
        self.val_dataset = ImageFolder(
            self.train_data_path, transform=transforms.Compose(self.transforms_test)
        )
        self.test_dataset = ImageFolder(
            self.test_data_path, transform=transforms.Compose(self.transforms_test)
        )

    def setup(self, stage=None):
        self.split_data()

    def split_data(self, val_size=0.2):
        indices = np.arange(len(self.train_dataset))
        train_idx, val_idx = train_test_split(indices, test_size=val_size, shuffle=True)
        tmp_train_dataset = c(self.train_dataset)
        tmp_val_dataset = c(self.val_dataset)
        self.train_dataset = Subset(tmp_train_dataset, train_idx)
        self.val_dataset = Subset(tmp_val_dataset, val_idx)
        del tmp_train_dataset, tmp_val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
