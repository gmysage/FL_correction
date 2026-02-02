import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import io

class Dataset3DCNN(Dataset):
    """
    Paired noisy / ground-truth 3D volumes
    Shape per sample: (B, H, W)
    """

    def __init__(
        self,
        gt_dir,
        noisy_dir,
        file_ext=".tiff", # ".pt",   # or ".npy"
        transform=None
    ):
        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.transform = transform

        self.gt_files = sorted([
            f for f in os.listdir(gt_dir)
            if f.endswith(file_ext)
        ])

        self.noisy_files = sorted([
            f for f in os.listdir(noisy_dir)
            if f.endswith(file_ext)
        ])

        assert len(self.gt_files) > 0, "No gt data found"
        assert len(self.noisy_files) == len(self.gt_files), "num of gt_image != num of noisy image"

    def _load(self, path):
        if path.endswith(".pt"):
            return torch.load(path)
        elif path.endswith(".npy"):
            return torch.from_numpy(np.load(path))
        elif path.endswith(".tiff"):
            return torch.from_numpy(io.imread(path))
        else:
            raise ValueError("Unsupported format")

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        fname_gt = self.gt_files[idx]
        fname_noisy = self.noisy_files[idx]

        gt = self._load(os.path.join(self.gt_dir, fname_gt))
        noisy = self._load(os.path.join(self.noisy_dir, fname_noisy))

        # (B, H, W) → float32
        gt = gt.float()
        noisy = noisy.float()

        if self.transform is not None:
            noisy, gt = self.transform(noisy, gt)

        return {
            "noisy": noisy,   # (B, H, W)
            "gt": gt
        }


def dataloader_3DCNN(
    gt_dir,
    noisy_dir,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    file_ext=".tiff"
):
    dataset = Dataset3DCNN(
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
        file_ext=file_ext
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )