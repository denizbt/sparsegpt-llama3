"""ImageNet calibration and evaluation loaders for ViT SparseGPT."""

import random

import torch
from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader, Dataset

IMAGENET_DATASET = "ILSVRC/imagenet-1k"


class ProcessedImageDataset(Dataset):
    def __init__(self, dataset, processor, indices=None):
        self.dataset = dataset
        self.processor = processor
        self.indices = list(indices) if indices is not None else None

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.dataset)

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]
        example = self.dataset[index]
        image = example["image"].convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
        return pixel_values, int(example["label"])


def get_imagenet_loaders(
    root,
    processor,
    nsamples=512,
    seed=0,
    eval_batch_size=32,
    train_split="train",
    validation_split="validation",
    cache_dir=None,
    calibration_indices=None,
    eval_samples=None,
    use_cached_imagenet=False,
    require_validation=True,
    eval_seed=0,
):
    """Load previously downloaded ImageNet data and sample calibration images.

    Supply either a local ImageFolder ``root`` or ``use_cached_imagenet=True``
    for an existing Hugging Face ``ILSVRC/imagenet-1k`` cache. Network access
    is disabled here deliberately; downloading ImageNet is a prerequisite.
    """
    if int(root is not None) + int(use_cached_imagenet) != 1:
        raise ValueError("Supply either imagenet_root or --cached-imagenet, not both")
    download_config = DownloadConfig(local_files_only=True)
    if use_cached_imagenet:
        dataset = load_dataset(
            IMAGENET_DATASET, cache_dir=cache_dir, download_config=download_config
        )
    else:
        dataset = load_dataset(
            "imagefolder", data_dir=root, cache_dir=cache_dir,
            download_config=download_config,
        )
    required_splits = {train_split}
    if require_validation:
        required_splits.add(validation_split)
    missing = required_splits - set(dataset)
    if missing:
        raise ValueError(
            f"Dataset source is missing splits {sorted(missing)}; "
            f"found {sorted(dataset)}"
        )
    if nsamples > len(dataset[train_split]):
        raise ValueError(
            f"Requested {nsamples} calibration images, but the training split "
            f"contains only {len(dataset[train_split])}"
        )

    if calibration_indices is None:
        indices = list(range(len(dataset[train_split])))
        random.Random(seed).shuffle(indices)
    else:
        indices = list(calibration_indices)
    if len(indices) < nsamples:
        raise ValueError(
            f"Only {len(indices)} calibration indices were provided for {nsamples} samples"
        )
    indices = indices[:nsamples]
    calibration_dataset = ProcessedImageDataset(
        dataset[train_split], processor, indices=indices
    )
    calibration_loader = DataLoader(
        calibration_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    if not require_validation:
        return calibration_loader, None

    validation_indices = None
    if eval_samples is not None:
        if eval_samples <= 0:
            raise ValueError("eval_samples must be positive")
        validation_indices = list(range(len(dataset[validation_split])))
        random.Random(eval_seed).shuffle(validation_indices)
        validation_indices = validation_indices[:eval_samples]
    validation_dataset = ProcessedImageDataset(
        dataset[validation_split], processor, indices=validation_indices
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0
    )
    return calibration_loader, validation_loader
