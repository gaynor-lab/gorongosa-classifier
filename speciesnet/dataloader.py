from typing import Optional, List
import os
from PIL import Image, UnidentifiedImageError, ImageFile

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision import io as tvio

# tolerate partial/corrupt JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SpeciesImageDataset(Dataset):
    def __init__(
        self,
        df,
        image_dir,
        backbone: str = "resnet18",
        include_species: Optional[List[str]] = None,
        augment: bool = True,
        use_color_jitter: bool = False,
        read_backend: str = "pil",
        size: int = 224,
    ):
        self.image_dir = image_dir
        self.backbone = str(backbone).lower()
        self.read_backend = read_backend
        self.size = int(size)

        if self.backbone != "resnet18":
            raise ValueError("This dataloader only supports backbone='resnet18'")

        df = df.reset_index(drop=True).copy()
        df["species"] = df["species"].astype(str).str.strip().str.lower()

        if include_species is not None:
            classes = [s.strip().lower() for s in include_species]
            df = df[df["species"].isin(classes)].reset_index(drop=True)
        else:
            classes = sorted(df["species"].unique())

        self.classes = classes
        self.class_to_idx = {s: i for i, s in enumerate(self.classes)}
        df["ground_truth_index"] = df["species"].map(self.class_to_idx).astype(int)

        if "filename_crop" not in df.columns:
            raise ValueError("DataFrame must contain 'filename_crop'")

        df = df[df["filename_crop"].notna()].reset_index(drop=True)

        self._filenames = df["filename_crop"].astype(str).tolist()
        self._labels = df["ground_truth_index"].tolist()
        self._bboxes = df["bbox"].tolist() if "bbox" in df.columns else None

        base = [
            T.Resize(
                (self.size, self.size),
                interpolation=InterpolationMode.BILINEAR,
            ),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]

        aug = [
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
        ]

        if use_color_jitter:
            aug.append(T.ColorJitter(brightness=0.1, contrast=0.1))

        self.transform = T.Compose((aug + base) if augment else base)

    def __len__(self):
        return len(self._filenames)

    def _load_image_pil(self, path: str):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise UnidentifiedImageError(f"Missing/zero-byte: {path}")

        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            raise UnidentifiedImageError(f"Unreadable: {path} ({e})")

    def _load_image_torch(self, path: str):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise UnidentifiedImageError(f"Missing/zero-byte: {path}")

        try:
            img = tvio.read_image(path)

            if img.ndim != 3 or img.size(0) != 3:
                pil = self._load_image_pil(path)
                return T.ToTensor()(pil) * 255.0

            return img

        except Exception as e:
            raise UnidentifiedImageError(f"Unreadable: {path} ({e})")

    def __getitem__(self, idx: int):
        fname = self._filenames[idx]
        label = int(self._labels[idx])
        path = os.path.join(self.image_dir, fname)

        try:
            if self.read_backend == "torch":
                img = self._load_image_torch(path)
                img = T.functional.resize(
                    img,
                    [self.size, self.size],
                    interpolation=InterpolationMode.BILINEAR,
                )
                img = img.float().div_(255.0)
                img = T.functional.normalize(
                    img,
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                )
                return img, label, fname

            img = self._load_image_pil(path)
            img = self.transform(img)
            return img, label, fname

        except UnidentifiedImageError:
            return None


def collate_keep_good(batch):
    """
    Supports items of shape:
      (img, label)                or
      (img, label, filename)

    - Drops None items (bad/corrupt samples).
    - Converts uint8 -> float32/255 only when needed.
    - Ensures 4D NCHW float32 contiguous batch.
    - Returns filenames list if provided, else [].
    """
    good = [b for b in batch if b is not None]

    if not good:
        return torch.empty(0), torch.empty(0, dtype=torch.long), []

    if len(good[0]) == 3:
        imgs, labels, names = zip(*good)
    else:
        imgs, labels = zip(*good)
        names = [None] * len(imgs)

    processed_imgs = []

    for img in imgs:
        if not torch.is_tensor(img):
            raise TypeError(f"Expected torch.Tensor, got {type(img)}")

        if img.ndim != 3:
            raise RuntimeError(f"Expected 3D CHW image, got shape {tuple(img.shape)}")

        if img.dtype == torch.uint8:
            img = img.float().div_(255.0)
        elif not img.is_floating_point():
            img = img.float()

        processed_imgs.append(img.contiguous())

    batch_imgs = torch.stack(processed_imgs, dim=0)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    names = list(names)

    try:
        batch_imgs = batch_imgs.contiguous(memory_format=torch.channels_last)
    except Exception:
        pass

    return batch_imgs, batch_labels, names