"""
model_utils.py - Pretrained model loading, inference, and output interpretation for MediScan AI.

This module encapsulates the TorchXRayVision DenseNet-121 architecture ('densenet121-res224-all')
trained across multiple clinical chest radiograph datasets (NIH, CheXpert, MIMIC-CXR, PadChest,
Google OpenI, RSNA). It guarantees strict evaluation mode, disables gradient tracking,
validates tensor input/output shapes, and formats predictions with clinical transparency.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Dict, List, Any, Optional, Tuple
import torch
import torchxrayvision as xrv

MODEL_NAME: str = "densenet121-res224-all"
EXPECTED_CHANNELS: int = 1
EXPECTED_SPATIAL_RES: int = 224
NUM_CLASSES: int = 18

# Official weight download URL from TorchXRayVision releases
WEIGHTS_URL: str = (
    "https://github.com/mlmed/torchxrayvision/releases/download/v1/"
    "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
)
WEIGHTS_FILENAME: str = "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"

# Plain-language educational descriptions of radiographic pathologies
PATHOLOGY_DESCRIPTIONS: Dict[str, str] = {
    "Atelectasis": "Partial or complete collapse of a lung or lobe, often visible as volume loss or opacity.",
    "Consolidation": "Alveolar spaces filled with fluid, pus, or cellular debris, appearing as dense white areas.",
    "Infiltration": "Accumulation of abnormal substance (fluid, inflammatory cells) in pulmonary parenchyma.",
    "Pneumothorax": "Abnormal collection of air in the pleural space between lung and chest wall.",
    "Edema": "Accumulation of fluid in lung interstitium and alveoli, commonly associated with cardiac failure.",
    "Emphysema": "Permanent enlargement of airspaces distal to terminal bronchioles with wall destruction.",
    "Fibrosis": "Formation of excess fibrous connective tissue in the lung, appearing as reticular markings.",
    "Effusion": "Excess fluid accumulation between the layers of the pleura outside the lungs.",
    "Pneumonia": "Inflammatory condition of the lung primarily affecting the microscopic air sacs (alveoli).",
    "Pleural_Thickening": "Fibrous thickening or scarring of the pleural membranes surrounding the lungs.",
    "Cardiomegaly": "Enlargement of the cardiac silhouette (cardiothoracic ratio > 0.50 on PA projection).",
    "Nodule": "Discrete, well-defined pulmonary lesion up to 30 mm in diameter.",
    "Mass": "Circumscribed pulmonary lesion greater than 30 mm in diameter.",
    "Hernia": "Protrusion of abdominal structures through diaphragm into thoracic cavity.",
    "Lung Lesion": "Any localized abnormal tissue change or density detected within the lung field.",
    "Fracture": "Break or discontinuity in thoracic skeletal structures (ribs, clavicle, spine).",
    "Lung Opacity": "Non-specific radiographic density attenuation visible in the lung parenchyma.",
    "Enlarged Cardiomediastinum": "Widened central compartment of the thoracic cavity (heart, great vessels, trachea).",
}


class ModelInferenceError(Exception):
    """Raised when model loading, validation, or inference encounters an error."""
    pass


def get_default_weights_path() -> str:
    """Returns the local path where torchxrayvision stores its model cache."""
    cache_dir = os.path.expanduser(os.path.join("~", ".torchxrayvision", "models_data"))
    return os.path.join(cache_dir, WEIGHTS_FILENAME)


def ensure_weights_available(target_path: Optional[str] = None) -> str:
    """
    Checks if model weights exist locally. If missing, downloads them safely.

    Args:
        target_path: Specific destination filepath. Defaults to torchxrayvision cache directory.

    Returns:
        Absolute filepath to the verified weights file.

    Raises:
        ModelInferenceError: If weight download fails or file is empty.
    """
    if target_path is None:
        target_path = get_default_weights_path()

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
        return target_path

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, target_path)
    except Exception as err:
        raise ModelInferenceError(
            f"Failed to download model weights from {WEIGHTS_URL}: {str(err)}. "
            "Please verify network access or manually place the weights file."
        ) from err

    if not os.path.exists(target_path) or os.path.getsize(target_path) < 1000000:
        raise ModelInferenceError(
            f"Downloaded weights file at '{target_path}' is invalid or incomplete."
        )

    return target_path


def load_model(weights_name: str = MODEL_NAME) -> xrv.models.DenseNet:
    """
    Loads and initializes the verified pretrained TorchXRayVision DenseNet-121 model.
    Guarantees that the model is in eval mode with pretrained weights loaded.

    Args:
        weights_name: Identifier of the pretrained weight set (default: 'densenet121-res224-all').

    Returns:
        Configured PyTorch DenseNet model in eval mode.

    Raises:
        ModelInferenceError: If weights cannot be found or loaded.
    """
    try:
        ensure_weights_available()
        model = xrv.models.DenseNet(weights=weights_name)
        model.eval()
        return model
    except Exception as err:
        raise ModelInferenceError(
            f"Failed to initialize TorchXRayVision DenseNet model '{weights_name}': {str(err)}"
        ) from err


def validate_input_tensor(tensor: torch.Tensor) -> None:
    """
    Validates the input tensor shape, dimensions, and data type.

    Args:
        tensor: PyTorch tensor to be verified.

    Raises:
        ModelInferenceError: If tensor shape or type does not match model expectations.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ModelInferenceError(f"Expected torch.Tensor, but received {type(tensor).__name__}.")

    if tensor.ndim != 4:
        raise ModelInferenceError(
            f"Expected 4D input tensor (batch, channels, height, width), but received shape {tuple(tensor.shape)}."
        )

    batch, channels, height, width = tensor.shape

    if channels != EXPECTED_CHANNELS:
        raise ModelInferenceError(
            f"Model requires {EXPECTED_CHANNELS} channel (grayscale), but received {channels} channels."
        )

    if height != EXPECTED_SPATIAL_RES or width != EXPECTED_SPATIAL_RES:
        raise ModelInferenceError(
            f"Expected spatial resolution of {EXPECTED_SPATIAL_RES}x{EXPECTED_SPATIAL_RES}, "
            f"but received {height}x{width}."
        )


def run_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor
) -> Dict[str, Any]:
    """
    Executes real inference on a preprocessed chest X-ray tensor without gradient computation.

    Returns structured results including:
        - pathology: Name of radiographic finding.
        - normalized_score: Score normalized against the dataset's calibrated operating point [0, 1],
          where >= 0.50 indicates the model's calibrated decision threshold was reached.
        - raw_probability: The raw sigmoid probability output.
        - operating_threshold: The dataset-calibrated operating threshold.
        - description: Plain-language educational explanation of the finding.
        - status: Qualitative indicator based on the operating threshold.

    Args:
        model: Pretrained TorchXRayVision model.
        tensor: Validated 4D input tensor of shape (1, 1, 224, 224).

    Returns:
        Dictionary containing model metadata, top findings, and complete class breakdown.

    Raises:
        ModelInferenceError: If inference fails or output dimensions are unexpected.
    """
    validate_input_tensor(tensor)

    try:
        model.eval()
        with torch.no_grad():
            # 1. Forward pass with default operating threshold normalization
            # DenseNet with op_threshs outputs normalized values where 0.5 is the operating point
            norm_outputs = model(tensor)  # Shape: (1, 18)

            # 2. Compute raw sigmoid probabilities from features and classifier
            features = model.features2(tensor)
            raw_logits = model.classifier(features)
            raw_probs = torch.sigmoid(raw_logits)

    except Exception as err:
        raise ModelInferenceError(f"Inference execution failed: {str(err)}") from err

    if norm_outputs.ndim != 2 or norm_outputs.shape[1] != NUM_CLASSES:
        raise ModelInferenceError(
            f"Unexpected model output shape {tuple(norm_outputs.shape)}. Expected (1, {NUM_CLASSES})."
        )

    pathologies: List[str] = list(model.pathologies)
    op_threshs: List[float] = [
        float(t) for t in getattr(model, "op_threshs", torch.zeros(NUM_CLASSES))
    ]

    findings: List[Dict[str, Any]] = []
    for idx, name in enumerate(pathologies):
        norm_score = float(norm_outputs[0, idx].item())
        raw_prob = float(raw_probs[0, idx].item())
        thresh = op_threshs[idx] if idx < len(op_threshs) else 0.5
        is_elevated = norm_score >= 0.50

        findings.append({
            "pathology": name,
            "normalized_score": round(norm_score, 4),
            "raw_probability": round(raw_prob, 4),
            "operating_threshold": round(thresh, 4),
            "status": "Elevated Radiographic Likelihood" if is_elevated else "Below Operating Threshold",
            "is_elevated": is_elevated,
            "description": PATHOLOGY_DESCRIPTIONS.get(name, "Radiographic finding on chest X-ray."),
        })

    # Sort findings by normalized score descending
    findings_sorted = sorted(findings, key=lambda x: x["normalized_score"], reverse=True)

    # Filter findings that exceed the calibrated operating threshold (score >= 0.50)
    elevated_findings = [f for f in findings_sorted if f["is_elevated"]]

    return {
        "model_name": MODEL_NAME,
        "total_classes": NUM_CLASSES,
        "elevated_count": len(elevated_findings),
        "elevated_findings": elevated_findings,
        "all_findings": findings_sorted,
    }
