import os
import torch
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io
from skimage.transform import resize
import shutil

def split_dataset_to_dataloader(
        dataset,
        split_ratio=0.8,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        seed=42
):
    n = len(dataset)
    assert n > 1, "Need at least 2 samples to split"
    # ----- compute split sizes -----
    n_train = int(split_ratio * n)
    n_valid = n - n_train

    # ----- deterministic split -----
    generator = torch.Generator().manual_seed(seed)
    train_set, valid_set = random_split(dataset, [n_train, n_valid], generator=generator)
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
        shuffle=False,  # usually validation not shuffled
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    n_train = len(train_loader)
    n_valid = len(valid_loader)
    n = len(dataset)
    print(f"Dataset split: {n} total → {n_train} train + {n_valid} valid")
    return train_loader, valid_loader

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
    dataset = Dataset3DCNN(gt_dir=gt_dir, noisy_dir=noisy_dir, file_ext=file_ext)
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
    split_ratio=0.8,
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
    dataset = Dataset3DCNN(gt_dir=gt_dir, noisy_dir=noisy_dir, file_ext=file_ext)
    train_loader, valid_loader = split_dataset_to_dataloader(
        dataset,
        split_ratio,
        batch_size,
        shuffle,
        num_workers,
        seed=seed)

    return train_loader, valid_loader
###############################################

class Dataset_gt_blur(Dataset):
    def __init__(self, img_gt_dir, blur_dir, length=None, exclude_range=[], binning=1):
        super().__init__()
        self.fn_img = np.sort(glob.glob(f'{img_gt_dir}/*'))
        self.fn_blur = np.sort(glob.glob(f'{blur_dir}/*'))
        self.binning = binning
        assert len(self.fn_img) == len(self.fn_blur), "Mismatch: Number of GT and Blur files must equal!"

        if length is not None:
            self.fn_img = self.fn_img[:length]
            self.fn_blur = self.fn_blur[:length]

        if len(exclude_range) == 2:
            fs, fe = exclude_range
            t1 = self.fn_img[:fs]
            t2 = self.fn_img[fe:]
            self.fn_img = list(t1) + list(t2)

            t1 = self.fn_blur[:fs]
            t2 = self.fn_blur[fe:]
            self.fn_blur = list(t1) + list(t2)

    def __len__(self):
        return len(self.fn_img)

    def __getitem__(self, idx):
        img_gt = io.imread(self.fn_img[idx])  # (512, 512)
        img_blur = io.imread(self.fn_blur[idx])
        s = img_gt.shape
        if self.binning == 2:
            newshape = (s[0] // self.binning, s[1] // self.binning)
            img_gt = resize(img_gt, newshape, anti_aliasing=True, preserve_range=True)
            img_blur = resize(img_blur, newshape, anti_aliasing=True, preserve_range=True)

        img_gt = np.clip(img_gt, 0, None)
        img_blur = np.clip(img_blur, 0, None)

        img_gt = img_gt[np.newaxis]
        img_gt = torch.tensor(img_gt, dtype=torch.float)

        img_blur = img_blur[np.newaxis]
        img_blur = torch.tensor(img_blur, dtype=torch.float)
        return {
            "noisy": img_blur,
            "gt": img_gt
        }


def dataloader_RRDB(
    gt_dir,
    noisy_dir,
    batch_size=16,
    length=None,
    exclude_range=[],
    binning=1,
    shuffle=True,
    num_workers=0,
):
    dataset = Dataset_gt_blur(gt_dir, noisy_dir, length, exclude_range, binning)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

def dataloader_RRDB_split(
    gt_dir,
    noisy_dir,
    batch_size=16,
    length=None,
    exclude_range=[],
    binning=1,
    split_ratio=0.8,
    shuffle=True,
    num_workers=0,
):
    dataset = Dataset_gt_blur(gt_dir, noisy_dir, length, exclude_range, binning)
    train_loader, valid_loader = split_dataset_to_dataloader(
        dataset,
        split_ratio,
        batch_size,
        shuffle,
        num_workers,
        seed=42)
    return train_loader, valid_loader



############# RRDB for 3D denoise ############


### using img_stack for training
class Dataset_3D_gt_blur(Dataset):
    def __init__(self, img3D_gt, img3D_blur, length=None, binning=1):
        super().__init__()

        # Expecting shape (Depth, Height, Width)
        self.img3D_gt = np.array(img3D_gt)
        self.img3D_blur = np.array(img3D_blur)
        self.binning = binning

        # Apply maximum length limit if provided
        if length is not None:
            self.img3D_gt = self.img3D_gt[:length]
            self.img3D_blur = self.img3D_blur[:length]

    def __len__(self):
        return self.img3D_gt.shape[0]

    def __getitem__(self, idx):
        # Extract a single 2D slice from the 3D stack
        img_gt = self.img3D_gt[idx].copy()
        img_blur = self.img3D_blur[idx].copy()

        # Handle spatial downsampling (binning)
        if self.binning == 2:
            s = img_gt.shape
            newshape = (s[0] // self.binning, s[1] // self.binning)
            img_gt = resize(img_gt, newshape, anti_aliasing=True, preserve_range=True)
            img_blur = resize(img_blur, newshape, anti_aliasing=True, preserve_range=True)

        # Clip negative values to zero
        img_gt[img_gt < 0] = 0
        img_blur[img_blur < 0] = 0

        # Add channel dimension: (H, W) -> (1, H, W)
        img_gt = img_gt[np.newaxis]
        img_blur = img_blur[np.newaxis]

        # Convert to PyTorch floats
        img_gt_tensor = torch.tensor(img_gt, dtype=torch.float)
        img_blur_tensor = torch.tensor(img_blur, dtype=torch.float)

        return {
            "noisy": img_blur_tensor,
                "gt": img_gt_tensor
        }


def dataloader_3D_image_stack(img3D_gt, img3D_blur, batch_size=8, split_ratio=0.8, exclude_range=[], binning=1):
    # Instantiate the new 3D array-based dataset
    dataset = Dataset_3D_gt_blur(
        img3D_gt=img3D_gt,
        img3D_blur=img3D_blur,
        length=None,  # Fixed bug: was previously passing batch_size here
        exclude_range=exclude_range,
        binning=binning
    )
    train_loader, valid_loader = split_dataset_to_dataloader(dataset, split_ratio, batch_size)
    return train_loader, valid_loader

def dataloader_img3D_to_3D_stack(img3D_gt, img3D_blur, batch_size=2, n_sli=32, split_ratio=0.9):
    gt_dir, noisy_dir = save_img3D_to_3D_stack(img3D_gt, img3D_blur)
    train_loader, valid_loader = dataloader_3DCNN_split(gt_dir, noisy_dir, batch_size, split_ratio)
    return train_loader, valid_loader

def dataloader_img3D_to_2D_slice(img3D_gt, img3D_blur, batch_size=2, split_ratio=0.9):
    gt_dir, noisy_dir = save_img3D_to_2D_slice(img3D_gt, img3D_blur)
    train_loader, valid_loader = dataloader_RRDB_split(gt_dir, noisy_dir, batch_size, split_ratio=split_ratio)
    return train_loader, valid_loader

def save_img3D_to_2D_slice(img3D_gt, img3D_blur):
    cwd = os.getcwd()
    gt_dir = os.path.join(cwd, 'tmp_image_for_training', 'gt_2D_image')
    noisy_dir = os.path.join(cwd, 'tmp_image_for_training', 'noisy_2D_image')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)
    n = len(img3D_gt)
    for i in range(n):
        img_gt = img3D_gt[i]
        img_blur = img3D_blur[i]
        io.imsave(os.path.join(gt_dir, f'img_gt_{i:04d}.tiff'), img_gt.astype(np.float32))
        io.imsave(os.path.join(noisy_dir, f'img_noise_{i:04d}.tiff'), img_blur.astype(np.float32))
    return gt_dir, noisy_dir

def save_img3D_to_3D_stack(img3D_gt, img3D_blur, n_sli=32, split_ratio=0.9):
    cwd = os.getcwd()
    gt_dir = os.path.join(cwd, 'tmp_image_for_training', 'gt_3D_image')
    noisy_dir = os.path.join(cwd, 'tmp_image_for_training', 'noisy_3D_image')
    shutil.rmtree(gt_dir, ignore_errors=True)
    shutil.rmtree(noisy_dir, ignore_errors=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)
    n = len(img3D_gt)
    n_f = n // n_sli
    for i in range(n_f + 1):
        id_s = min(i * n_sli, n - n_sli)
        id_e = min((i + 1) * n_sli, n)
        img_gt = img3D_gt[id_s: id_e]
        img_blur = img3D_blur[id_s: id_e]
        io.imsave(os.path.join(gt_dir, f'img_gt_{i:04d}.tiff'), img_gt.astype(np.float32))
        io.imsave(os.path.join(noisy_dir, f'img_noise_{i:04d}.tiff'), img_blur.astype(np.float32))
    return gt_dir, noisy_dir