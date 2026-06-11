"""
Data loader for the Open-I chest X-ray dataset.

Expected directory layout (after Kaggle download and extraction):
    data/open-i/
        train/
            images/   (*.png or *.jpg)
            reports/  (*.txt  — one report per file, same stem as image)
        val/
            images/
            reports/
        test/
            images/
            reports/

Each report file contains plain-text radiology findings + impression sections.
"""

import os
import glob
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class OpenIDataset(Dataset):
    """
    Dataset for Open-I chest X-ray image–report pairs.

    Args:
        split_dir: Directory containing 'images/' and 'reports/' subdirectories.
        transform: torchvision transforms applied to images.
        tokenizer: HuggingFace tokenizer for report text.
        max_seq_length: Maximum token length for reports.
        return_raw_text: If True, also return the raw report string alongside tokens.
    """

    def __init__(
        self,
        split_dir: str,
        transform=None,
        tokenizer=None,
        max_seq_length: int = 128,
        return_raw_text: bool = False,
    ):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.return_raw_text = return_raw_text

        self.samples: List[Tuple[Path, Path]] = self._scan_pairs()
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image–report pairs found in {split_dir}. "
                "Ensure 'images/' and 'reports/' subdirectories exist."
            )
        logger.info("Loaded %d pairs from %s", len(self.samples), split_dir)

    def _scan_pairs(self) -> List[Tuple[Path, Path]]:
        """Find (image_path, report_path) pairs with matching stems."""
        img_dir = self.split_dir / "images"
        rep_dir = self.split_dir / "reports"

        pairs = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img_path in sorted(img_dir.glob(ext)):
                rep_path = rep_dir / (img_path.stem + ".txt")
                if rep_path.exists():
                    pairs.append((img_path, rep_path))
                else:
                    logger.warning("No report found for image: %s", img_path.name)
        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, rep_path = self.samples[idx]

        # --- load image ---
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # --- load report ---
        report_text = rep_path.read_text(encoding="utf-8").strip()

        item = {"image": image, "report_text": report_text, "image_path": str(img_path)}

        # --- tokenize if tokenizer provided ---
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                report_text,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            # squeeze batch dim added by return_tensors='pt'
            item["input_ids"] = encoding["input_ids"].squeeze(0)
            item["attention_mask"] = encoding["attention_mask"].squeeze(0)

        if not self.return_raw_text:
            item.pop("report_text")

        return item


def build_transform(image_resolution: int = 224, split: str = "train"):
    """
    Build torchvision image transform compatible with BioMedCLIP preprocessing.

    ImageNet mean/std are used to match BioMedCLIP's pretraining normalisation.
    """
    from torchvision import transforms

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_resolution, image_resolution)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_resolution, image_resolution)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )


def get_dataloader(
    split_dir: str,
    tokenizer,
    batch_size: int = 16,
    image_resolution: int = 224,
    max_seq_length: int = 128,
    split: str = "train",
    num_workers: int = 4,
    return_raw_text: bool = False,
) -> DataLoader:
    """
    Convenience function to build a DataLoader for a dataset split.

    Args:
        split_dir: Path to the split directory (e.g. 'data/open-i/train').
        tokenizer: HuggingFace T5 tokenizer.
        batch_size: Batch size.
        image_resolution: Image side length (224 for BioMedCLIP).
        max_seq_length: Report token length.
        split: 'train', 'val', or 'test' — controls augmentation.
        num_workers: DataLoader workers.
        return_raw_text: Whether to include raw report strings in batches.

    Returns:
        Configured DataLoader.
    """
    transform = build_transform(image_resolution, split)
    dataset = OpenIDataset(
        split_dir=split_dir,
        transform=transform,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        return_raw_text=return_raw_text,
    )
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=split == "train",
    )
