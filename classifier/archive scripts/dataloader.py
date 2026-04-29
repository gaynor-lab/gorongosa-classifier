# dataloader.py

from typing import Optional, List, Tuple
import os
from PIL import Image, UnidentifiedImageError, ImageFile

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision import io as tvio

# tolerate partial/corrupt JPEGs (avoids slow exception paths)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SpeciesImageDataset(Dataset):
    def __init__(
        self,
        df,
        image_dir,
        classifier=None,
        backbone: str = "resnet18",
        top_n_species: Optional[int] = None,
        include_species: Optional[List[str]] = None,
        augment: bool = True,
        use_color_jitter: bool = False,     # <- off by default (speed)
        read_backend: str = "pil",          # "pil" or "torch"
        size: int = 224                     # 224/256 are much faster than 384
    ):
        self.image_dir = image_dir
        self.classifier = classifier
        self.backbone = str(backbone).lower()
        self.read_backend = read_backend
        self.size = int(size)

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

        # Pre-cache columns to avoid df.iloc (noticeably faster)
        self._filenames = df["filename"].tolist()
        self._labels = df["ground_truth_index"].tolist()
        # Optional: fast bbox access if present
        self._bboxes = df["bbox"].tolist() if "bbox" in df.columns else None

        # Build transforms
        if self.backbone == "resnet18":
            # Cheaper resize path (bilinear + no antialias)
            base = [
                T.Resize((self.size, self.size), interpolation=InterpolationMode.BILINEAR, antialias=False),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            aug = [T.RandomHorizontalFlip(), T.RandomRotation(10)]
            if use_color_jitter:
                aug.append(T.ColorJitter(brightness=0.1, contrast=0.1))
            self.transform = T.Compose((aug + base) if augment else base)
        elif self.backbone == "speciesnet":
            # SpeciesNet handles preprocess itself
            self.transform = None
        else:
            raise ValueError("backbone must be 'resnet18' or 'speciesnet'")

    def __len__(self) -> int:
        return len(self._filenames)

    def _load_image_pil(self, path: str) -> Image.Image:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise UnidentifiedImageError(f"Missing/zero-byte: {path}")
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            raise UnidentifiedImageError(f"Unreadable: {path} ({e})")

    def _load_image_torch(self, path: str) -> torch.Tensor:
        # Returns CHW uint8 tensor [3,H,W] without PIL; faster in many cases.
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise UnidentifiedImageError(f"Missing/zero-byte: {path}")
        try:
            img = tvio.read_image(path)  # uint8 [C,H,W]
            if img.ndim != 3 or img.size(0) != 3:
                # force RGB if odd mode
                # fallback to PIL to convert (rare)
                pil = self._load_image_pil(path)
                return T.ToTensor()(pil) * 255.0
            return img
        except Exception as e:
            raise UnidentifiedImageError(f"Unreadable: {path} ({e})")

    # --- inside SpeciesImageDataset.__getitem__ ---
    def __getitem__(self, idx: int):
        fname = self._filenames[idx]
        label = int(self._labels[idx])
        path = os.path.join(self.image_dir, fname)

        try:
            if self.backbone == "resnet18":
                if self.read_backend == "torch":
                    img = self._load_image_torch(path)         # [C,H,W] uint8
                    img = T.functional.resize(
                        img, [self.size, self.size],
                        interpolation=InterpolationMode.BILINEAR, antialias=False
                    )
                    img = img.float().div_(255.0)
                    img = T.functional.normalize(img, [0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])
                    return img, label, fname                   # <- add fname
                else:
                    img = self._load_image_pil(path)
                    img = self.transform(img)
                    return img, label, fname                   # <- add fname

            # speciesnet path
            if self.classifier is None or not hasattr(self.classifier, "preprocess"):
                raise RuntimeError("SpeciesNet backbone requires classifier.preprocess")

            bboxes = None
            if self._bboxes is not None:
                bb = self._bboxes[idx]
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    try:
                        from speciesnet.utils import BBox
                        bboxes = [BBox(*bb)]
                    except Exception:
                        bboxes = None

            pre = self.classifier.preprocess(self._load_image_pil(path), bboxes=bboxes)
            arr = torch.as_tensor(pre.arr).permute(2, 0, 1).float().div_(255.0)
            return arr, label, fname                            # <- add fname

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
    # Drop bad samples
    good = [b for b in batch if b is not None]
    if not good:
        # keep return arity stable: (imgs, labels, names)
        return torch.empty(0), torch.empty(0, dtype=torch.long), []

    # Unpack with/without filenames
    if len(good[0]) == 3:
        imgs, labels, names = zip(*good)
    else:
        imgs, labels = zip(*good)
        names = [None] * len(imgs)

    proc = []
    for img in imgs:
        if not torch.is_tensor(img):
            raise TypeError(f"Expected torch.Tensor, got {type(img)}")

        # Require CHW
        if img.ndim != 3:
            raise RuntimeError(f"Expected 3D CHW image, got shape {tuple(img.shape)}")

        # Convert only if uint8; avoid double-dividing already-normalized float tensors
        if img.dtype == torch.uint8:
            img = img.float().div_(255.0)
        elif not img.is_floating_point():
            img = img.float()

        proc.append(img.contiguous())

    # Stack -> NCHW
    batch_imgs = torch.stack(proc, dim=0)  # [N, C, H, W]
    batch_labels = torch.tensor(labels, dtype=torch.long)
    names = list(names)

    # Optional: channels_last is valid only for 4D tensors
    try:
        batch_imgs = batch_imgs.contiguous(memory_format=torch.channels_last)
    except Exception:
        pass  # silently skip if not supported

    return batch_imgs, batch_labels, names


