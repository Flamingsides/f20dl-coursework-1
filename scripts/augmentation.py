"""
augmentation.py — data augmentation for the image models (R2 -> R4). 

Why: only ~30 TRAINING images per species (60 total, half are test), which is
very few for a CNN. Augmentation creates a slightly different version of every
training image each epoch, so the model cannot memorise the 30 photos.

Leakage rules followed here:
  1. Augment the TRAINING split only. Validation/test images get the same
     deterministic resize + normalise every time.
  2. Make the validation split from the TRAINING images BEFORE anything is
     augmented, so no augmented copy of a validation image is ever trained on.
  3. Augment on the fly (new random version per epoch), never save augmented
     copies to disk mixed with originals.

Bird-specific choices (justify these in the report):
  - Horizontal flip: YES (a bird facing left is the same species).
  - Vertical flip:   NO  (birds are almost never upside down; it creates
                          unrealistic images).
  - Hue jitter:      VERY SMALL or off. Colour is the main cue for many species
                     (eda.ipynb: Blue Grosbeak vs Indigo Bunting, Yellow vs
                     Wilson Warbler), so shifting hue can turn one species into
                     another. Brightness/contrast/saturation are safe.
  - Rotation:        small (+-15 degrees), like a tilted camera.

Needs (add to the dependency file): torch, torchvision (>= 0.16 for MixUp/CutMix).
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

RANDOM_STATE = 42
IMG_SIZE = 224
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
N_CLASSES = 200

# ---------------------------------------------------------------------------
# 1. Transforms
# ---------------------------------------------------------------------------
train_transform = T.Compose([
    T.ToImage(),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3), antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    T.RandomErasing(p=0.25, scale=(0.02, 0.15)),   # hides a patch -> model can't rely on one part
])

eval_transform = T.Compose([                        # validation AND test: no randomness
    T.ToImage(),
    T.Resize(IMG_SIZE + 32, antialias=True),       # 256
    T.CenterCrop(IMG_SIZE),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# 2. Dataset + leakage-safe splits
# ---------------------------------------------------------------------------
class BirdImages(Dataset):
    def __init__(self, table, image_root, transform):
        self.paths = [Path(image_root) / p for p in table["cropped_path"]]
        self.labels = (table["class_id"] - 1).tolist()        # 1..200 -> 0..199
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            img = img.convert("RGB")
        return self.transform(img), self.labels[i]


def make_splits(csv_path, val_size=0.2):
    """Official test split stays untouched; validation is carved out of TRAIN, stratified by species."""
    table = pd.read_csv(csv_path)
    train_full = table[table["is_train"] == 1]
    test = table[table["is_train"] == 0]
    train, val = train_test_split(train_full, test_size=val_size,
                                  stratify=train_full["class_id"], random_state=RANDOM_STATE)
    return train, val, test


def make_loaders(processed_dir, batch_size=32, num_workers=2):
    processed_dir = Path(processed_dir)
    train, val, test = make_splits(processed_dir / "cub_images_cropped.csv")
    g = torch.Generator().manual_seed(RANDOM_STATE)
    return (
        DataLoader(BirdImages(train, processed_dir, train_transform), batch_size,
                   shuffle=True, num_workers=num_workers, generator=g, drop_last=True),
        DataLoader(BirdImages(val, processed_dir, eval_transform), batch_size, num_workers=num_workers),
        DataLoader(BirdImages(test, processed_dir, eval_transform), batch_size, num_workers=num_workers),
    )


# ---------------------------------------------------------------------------
# 3. Optional batch-level augmentation: MixUp / CutMix (training batches only)
# ---------------------------------------------------------------------------
mix = T.RandomChoice([T.MixUp(alpha=0.2, num_classes=N_CLASSES),
                      T.CutMix(alpha=1.0, num_classes=N_CLASSES)])
# In the training loop:
#     images, labels = mix(images, labels)    # labels become soft (batch, 200)
#     loss = torch.nn.functional.cross_entropy(model(images), labels)
# Never apply `mix` to validation/test batches.
