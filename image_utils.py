"""
image_utils.py - Image validation and preprocessing for MediScan AI.

This module provides safe, in-memory validation and preprocessing for medical chest
X-ray images, adhering to the documented input requirements of TorchXRayVision
DenseNet models (single-channel grayscale, 224x224 spatial resolution, [-1024, 1024]
intensity normalization).
"""

from __future__ import annotations

import io
from typing import Tuple, Dict, Any
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torchvision
import torchxrayvision as xrv

# Configuration Constants
MAX_IMAGE_SIZE_BYTES: int = 8 * 1024 * 1024  # 8 Megabytes
SUPPORTED_EXTENSIONS: set[str] = {"png", "jpg", "jpeg"}
SUPPORTED_FORMATS: set[str] = {"PNG", "JPEG"}
MIN_DIMENSION: int = 32
MAX_DIMENSION: int = 8192
TARGET_RESOLUTION: int = 224


class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation criteria."""
    pass


def validate_image_file(
    file_bytes: bytes,
    filename: str = "uploaded_image.png",
    max_size_bytes: int = MAX_IMAGE_SIZE_BYTES
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Validates the raw file bytes for file size, extension, format, and decodability.

    Args:
        file_bytes: In-memory byte array of the image file.
        filename: Original name of the uploaded file for extension checking.
        max_size_bytes: Maximum permitted file size in bytes.

    Returns:
        A tuple of (Pillow Image object, metadata dictionary).

    Raises:
        ImageValidationError: If the file is empty, oversized, unsupported, or corrupted.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise ImageValidationError("The provided file is empty (0 bytes).")

    if len(file_bytes) > max_size_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        raise ImageValidationError(
            f"File size ({size_mb:.2f} MB) exceeds the maximum allowed limit of {max_mb:.1f} MB."
        )

    # Check file extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file extension '.{ext}'. Supported extensions are: "
            f"{', '.join(sorted('.' + e for e in SUPPORTED_EXTENSIONS))}."
        )

    # Decode image in memory using PIL
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()  # Force decoding to detect truncation or corruption
    except (UnidentifiedImageError, OSError, ValueError) as err:
        raise ImageValidationError(
            f"Image file is corrupted or cannot be decoded: {str(err)}"
        ) from err

    # Verify decoded format
    if image.format not in SUPPORTED_FORMATS:
        raise ImageValidationError(
            f"Unsupported internal image format '{image.format}'. Supported formats: "
            f"{', '.join(sorted(SUPPORTED_FORMATS))}."
        )

    width, height = image.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) are too small. "
            f"Minimum dimensions required: {MIN_DIMENSION}x{MIN_DIMENSION}."
        )

    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed dimension of {MAX_DIMENSION}px."
        )

    metadata: Dict[str, Any] = {
        "filename": filename,
        "format": image.format,
        "mode": image.mode,
        "width": width,
        "height": height,
        "size_bytes": len(file_bytes),
    }

    return image, metadata


def preprocess_chest_xray(
    image: Image.Image,
    target_res: int = TARGET_RESOLUTION
) -> torch.Tensor:
    """
    Preprocesses a validated PIL Image into the exact tensor format expected by
    TorchXRayVision DenseNet models.

    Steps:
        1. Converts image to 8-bit single-channel grayscale ('L' mode).
        2. Converts to NumPy array with shape (H, W) and range [0, 255].
        3. Applies TorchXRayVision intensity normalization to [-1024, 1024].
        4. Expands channel dimension to (1, H, W).
        5. Applies XRayCenterCrop and XRayResizer(target_res).
        6. Converts to PyTorch FloatTensor and adds batch dimension (1, 1, 224, 224).

    Args:
        image: Decoded PIL Image object.
        target_res: Target spatial resolution (default 224).

    Returns:
        PyTorch float32 tensor of shape (1, 1, target_res, target_res) normalized to [-1024, 1024].
    """
    # 1. Convert to grayscale ('L' mode: 8-bit pixels, black and white)
    gray_image = image.convert("L")
    img_arr = np.array(gray_image, dtype=np.float32)

    # 2. Normalize intensity from [0, 255] to [-1024, 1024]
    # Standard TorchXRayVision normalization formula: (img / maxval) * 2048 - 1024
    norm_arr = xrv.datasets.normalize(img_arr, 255)

    # 3. Add single channel dimension -> shape: (1, H, W)
    if norm_arr.ndim == 2:
        norm_arr = norm_arr[None, ...]
    elif norm_arr.ndim == 3 and norm_arr.shape[0] != 1:
        # If multi-channel reached here, average channels
        norm_arr = norm_arr.mean(axis=0, keepdims=True)

    # 4. Apply TorchXRayVision aspect-ratio preserving CenterCrop and Resizer
    transforms = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(target_res)
    ])
    transformed_arr = transforms(norm_arr)

    # 5. Convert to PyTorch Tensor of shape (1, 1, target_res, target_res)
    tensor = torch.from_numpy(transformed_arr).float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # Add batch dimension -> (1, 1, 224, 224)

    return tensor
