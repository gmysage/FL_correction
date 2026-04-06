import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
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

def dataloader_3DCNN_split(
    gt_dir,
    noisy_dir,
    batch_size=2,
    train_ratio=0.8,
    shuffle=True,
    num_workers=0,
    file_ext=".tiff",
    seed=42
):
    """
    Create train/validation loaders from paired dataset

    Returns:
        train_loader, valid_loader
    """

    # ----- create full dataset -----
    dataset = Dataset3DCNN(
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
        file_ext=file_ext
    )

    N = len(dataset)

    assert N > 1, "Need at least 2 samples to split"

    # ----- compute split sizes -----
    n_train = int(train_ratio * N)
    n_valid = N - n_train

    # ----- deterministic split -----
    generator = torch.Generator().manual_seed(seed)

    train_set, valid_set = random_split(
        dataset,
        [n_train, n_valid],
        generator=generator
    )

    # ----- build loaders -----
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,      # usually validation not shuffled
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Dataset split: {N} total → {n_train} train + {n_valid} valid")

    return train_loader, valid_loader