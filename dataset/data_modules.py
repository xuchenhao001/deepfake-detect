from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset.nt_object import NTDetectionDataset


class NTObjectDataModule(L.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int, num_workers: int, image_size: int):
        super().__init__()
        root_dir = Path(root_dir)
        assert root_dir.exists(), f"provided path {root_dir} does not exist"

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.dataset_train, self.dataset_test = None, None

    def setup(self, stage: str):
        # https://pytorch.org/vision/stable/transforms.html
        transform_train = v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(size=(self.image_size, self.image_size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float, scale=True),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.ToDtype(torch.float, scale=True),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if stage == "fit":
            self.dataset_train = NTDetectionDataset(str(self.root_dir), is_train=True, transforms=transform_train)
        self.dataset_test = NTDetectionDataset(str(self.root_dir), is_train=False, transforms=transform_test)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
