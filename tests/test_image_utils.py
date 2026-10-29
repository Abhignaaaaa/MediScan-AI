"""
test_image_utils.py - Comprehensive unit tests for image validation and preprocessing.
"""

import io
import pytest
import numpy as np
from PIL import Image
import torch

from image_utils import (
    validate_image_file,
    preprocess_chest_xray,
    ImageValidationError,
    MAX_IMAGE_SIZE_BYTES,
    TARGET_RESOLUTION,
)


def create_test_image_bytes(
    format: str = "PNG",
    size: tuple[int, int] = (256, 256),
    color: tuple[int, int, int] | int = 128,
    mode: str = "RGB"
) -> bytes:
    """Helper to generate in-memory image bytes."""
    img = Image.new(mode, size, color=color)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


class TestImageValidation:
    """Tests for validate_image_file function."""

    def test_valid_png_image(self):
        """Verifies that a standard valid PNG image passes validation."""
        img_bytes = create_test_image_bytes(format="PNG", size=(256, 256), mode="L")
        image, meta = validate_image_file(img_bytes, filename="xray.png")

        assert image is not None
        assert meta["filename"] == "xray.png"
        assert meta["format"] == "PNG"
        assert meta["width"] == 256
        assert meta["height"] == 256
        assert meta["size_bytes"] == len(img_bytes)

    def test_valid_jpeg_image(self):
        """Verifies that a valid JPEG image passes validation."""
        img_bytes = create_test_image_bytes(format="JPEG", size=(512, 512), mode="RGB")
        image, meta = validate_image_file(img_bytes, filename="chest.jpg")

        assert image is not None
        assert meta["filename"] == "chest.jpg"
        assert meta["format"] == "JPEG"
        assert meta["width"] == 512

    def test_empty_input_rejected(self):
        """Verifies that empty bytes raise an ImageValidationError."""
        with pytest.raises(ImageValidationError, match="empty"):
            validate_image_file(b"", filename="empty.png")

    def test_oversized_input_rejected(self):
        """Verifies that files exceeding maximum byte size are rejected."""
        fake_oversized_bytes = b"0" * (MAX_IMAGE_SIZE_BYTES + 1024)
        with pytest.raises(ImageValidationError, match="exceeds the maximum allowed limit"):
            validate_image_file(fake_oversized_bytes, filename="huge.png")

    def test_unsupported_extension_rejected(self):
        """Verifies that unsupported file extensions are rejected."""
        img_bytes = create_test_image_bytes(format="PNG")
        with pytest.raises(ImageValidationError, match="Unsupported file extension"):
            validate_image_file(img_bytes, filename="scan.gif")

    def test_corrupted_image_rejected(self):
        """Verifies that malformed/corrupted byte streams are rejected."""
        corrupted_bytes = b"\x89PNG\r\n\x1a\n" + b"random_corrupted_garbage_bytes_12345"
        with pytest.raises(ImageValidationError, match="corrupted or cannot be decoded"):
            validate_image_file(corrupted_bytes, filename="corrupted.png")

    def test_invalid_dimensions_too_small(self):
        """Verifies that tiny images below minimum threshold are rejected."""
        tiny_bytes = create_test_image_bytes(format="PNG", size=(16, 16))
        with pytest.raises(ImageValidationError, match="too small"):
            validate_image_file(tiny_bytes, filename="tiny.png")

    def test_case_insensitive_extension(self):
        """Verifies that extensions like .PNG and .JPEG are accepted."""
        img_bytes = create_test_image_bytes(format="PNG", size=(100, 100))
        _, meta = validate_image_file(img_bytes, filename="TEST_SCAN.PNG")
        assert meta["format"] == "PNG"


class TestImagePreprocessing:
    """Tests for preprocess_chest_xray function."""

    def test_preprocessing_output_shape(self):
        """Verifies that preprocessing yields a 4D tensor with shape (1, 1, 224, 224)."""
        img = Image.new("L", (300, 400), color=150)
        tensor = preprocess_chest_xray(img, target_res=TARGET_RESOLUTION)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 4
        assert tensor.shape == (1, 1, TARGET_RESOLUTION, TARGET_RESOLUTION)
        assert tensor.dtype == torch.float32

    def test_preprocessing_rgb_to_single_channel(self):
        """Verifies that a 3-channel RGB image is correctly converted to single-channel."""
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        tensor = preprocess_chest_xray(img, target_res=224)

        assert tensor.shape == (1, 1, 224, 224)

    def test_preprocessing_normalization_range(self):
        """Verifies that pixel values are normalized to [-1024, 1024]."""
        # Create an image with black (0) and white (255) pixels
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[:50, :] = 255
        img = Image.fromarray(arr, mode="L")

        tensor = preprocess_chest_xray(img, target_res=224)

        # Min should be approximately -1024, max should be approximately 1024
        assert tensor.min().item() >= -1025.0
        assert tensor.max().item() <= 1025.0
        assert tensor.min().item() < 0
        assert tensor.max().item() > 0
