"""
test_model_utils.py - Comprehensive tests for model loading, inference, and error handling.
"""

from unittest.mock import MagicMock
import pytest
import torch

from model_utils import (
    load_model,
    validate_input_tensor,
    run_inference,
    ModelInferenceError,
    NUM_CLASSES,
    MODEL_NAME,
    EXPECTED_CHANNELS,
    EXPECTED_SPATIAL_RES,
)


class MockDenseNet(torch.nn.Module):
    """Mock model simulating TorchXRayVision DenseNet-121 without loading full weights."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.pathologies = [
            "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
            "Edema", "Emphysema", "Fibrosis", "Effusion",
            "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule",
            "Mass", "Hernia", "Lung Lesion", "Fracture",
            "Lung Opacity", "Enlarged Cardiomediastinum"
        ]
        self.op_threshs = torch.full((num_classes,), 0.05)
        self.apply_sigmoid = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns normalized scores where 0.6 is elevated (>0.5) and 0.2 is below threshold
        batch_size = x.shape[0]
        out = torch.full((batch_size, len(self.pathologies)), 0.2)
        out[:, 0] = 0.65  # Atelectasis elevated
        out[:, 10] = 0.72  # Cardiomegaly elevated
        return out

    def features2(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 1024))

    def classifier(self, f: torch.Tensor) -> torch.Tensor:
        # Raw logits
        return torch.zeros((f.shape[0], len(self.pathologies)))


class TestModelValidationAndInference:
    """Unit tests for tensor validation and inference logic using mock objects."""

    def test_validate_input_tensor_valid(self):
        """Verifies that a well-formed (1, 1, 224, 224) float tensor passes validation."""
        tensor = torch.zeros((1, EXPECTED_CHANNELS, EXPECTED_SPATIAL_RES, EXPECTED_SPATIAL_RES))
        # Should not raise
        validate_input_tensor(tensor)

    def test_validate_input_tensor_non_tensor(self):
        """Verifies that non-tensor inputs raise ModelInferenceError."""
        with pytest.raises(ModelInferenceError, match="Expected torch.Tensor"):
            validate_input_tensor([[1, 2], [3, 4]])  # type: ignore

    def test_validate_input_tensor_invalid_dimensions(self):
        """Verifies that non-4D tensors are rejected."""
        tensor_3d = torch.zeros((1, 224, 224))
        with pytest.raises(ModelInferenceError, match="Expected 4D input tensor"):
            validate_input_tensor(tensor_3d)

    def test_validate_input_tensor_invalid_channels(self):
        """Verifies that 3-channel RGB tensors are rejected at model boundary."""
        tensor_rgb = torch.zeros((1, 3, 224, 224))
        with pytest.raises(ModelInferenceError, match="Model requires 1 channel"):
            validate_input_tensor(tensor_rgb)

    def test_validate_input_tensor_invalid_spatial_resolution(self):
        """Verifies that tensors with mismatched spatial resolutions are rejected."""
        tensor_wrong_res = torch.zeros((1, 1, 128, 128))
        with pytest.raises(ModelInferenceError, match="Expected spatial resolution"):
            validate_input_tensor(tensor_wrong_res)

    def test_run_inference_with_mock_model(self):
        """Tests end-to-end inference output formatting using a mock model."""
        mock_model = MockDenseNet()
        tensor = torch.zeros((1, 1, 224, 224))

        results = run_inference(mock_model, tensor)

        assert results["model_name"] == MODEL_NAME
        assert results["total_classes"] == NUM_CLASSES
        assert results["elevated_count"] == 2
        assert len(results["all_findings"]) == NUM_CLASSES

        # Verify sorted order (highest normalized score first)
        top_finding = results["all_findings"][0]
        assert top_finding["pathology"] == "Cardiomegaly"
        assert top_finding["normalized_score"] == 0.72
        assert top_finding["is_elevated"] is True
        assert "Cardiomegaly" in results["elevated_findings"][0]["pathology"]

    def test_inference_exception_handling(self):
        """Verifies that model runtime errors raise ModelInferenceError."""
        faulty_model = MagicMock()
        faulty_model.eval = MagicMock()
        faulty_model.side_effect = RuntimeError("GPU Out of Memory")

        tensor = torch.zeros((1, 1, 224, 224))
        with pytest.raises(ModelInferenceError, match="Inference execution failed"):
            run_inference(faulty_model, tensor)

    def test_unexpected_output_shape_handling(self):
        """Verifies that non-(1, 18) output shapes are caught."""
        broken_model = MagicMock()
        broken_model.eval = MagicMock()
        broken_model.return_value = torch.zeros((1, 5))  # Only 5 classes instead of 18
        broken_model.features2 = MagicMock(return_value=torch.zeros((1, 1024)))
        broken_model.classifier = MagicMock(return_value=torch.zeros((1, 5)))

        tensor = torch.zeros((1, 1, 224, 224))
        with pytest.raises(ModelInferenceError, match="Unexpected model output shape"):
            run_inference(broken_model, tensor)


class TestRealPretrainedModelInference:
    """Smoke tests using the real pretrained weights."""

    def test_real_model_loading_and_inference(self):
        """Smoke test verifying that real pretrained weights load and produce valid predictions."""
        try:
            model = load_model()
        except ModelInferenceError as e:
            pytest.skip(f"Pretrained weights not available in current environment: {e}")

        assert model is not None
        assert len(model.pathologies) == NUM_CLASSES

        # Run inference on a synthetic valid normalized input
        dummy_input = torch.zeros((1, 1, 224, 224), dtype=torch.float32)
        results = run_inference(model, dummy_input)

        assert results["total_classes"] == NUM_CLASSES
        assert len(results["all_findings"]) == NUM_CLASSES
        for finding in results["all_findings"]:
            assert 0.0 <= finding["normalized_score"] <= 1.0
            assert 0.0 <= finding["raw_probability"] <= 1.0
            assert finding["pathology"] in model.pathologies
