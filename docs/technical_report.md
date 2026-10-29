# Technical Development Report — MediScan AI

**Project Name:** MediScan AI — AI-Powered Medical Image Analysis Assistant  
**Assessment Context:** Trialshopy Technical Development Intern Assessment (AI/ML, Generative AI, Computer Vision, AR/VR)  
**Submission Target:** 30 October 2026  
**Engineer / Author:** Antigravity AI Engineering Assistant on behalf of Candidate  

---

## 1. Problem Statement & Objectives

Modern clinical imaging environments generate millions of radiographs annually. Medical computer vision holds immense promise for assisting radiologists, accelerating preliminary triaging, and providing educational platforms for students and biomedical engineers. However, developing genuine medical AI tools requires overcoming critical pitfalls:
- Over-reliance on generic natural image classifiers (ImageNet) that lack radiographic understanding.
- Fabrication of mock predictions or uncalibrated confidence percentages.
- Neglect of domain-specific preprocessing pipelines (e.g., radiographic intensity normalization and aspect ratio handling).
- Failure to transparently communicate that statistical model likelihood is fundamentally distinct from a confirmed medical diagnosis.

### Core Objectives
1. Build a working educational prototype in Streamlit that accepts frontal chest radiographs.
2. Integrate a verified, peer-reviewed pretrained medical computer vision model with verifiable open weights and licensing.
3. Implement the exact preprocessing pipeline mandated by the selected model.
4. Execute genuine PyTorch inference without mock fallbacks or fabricated confidence numbers.
5. Provide clear, calibrated output interpretation across multiple radiographic pathologies.
6. Enforce strict in-memory privacy and comprehensive automated testing.

---

## 2. Requirements & Model Selection Rationale

### Evaluation of Candidate Models
During Phase 1 of development, three potential options were evaluated:

| Candidate Architecture | Weight Source & License | Task / Resolution | Evaluation Outcome |
| :--- | :--- | :--- | :--- |
| **Torchvision ResNet50 (ImageNet)** | PyTorch (BSD-3) | Natural objects (1000 classes, 224x224) | **Rejected:** Generic natural classifier; completely inappropriate for medical pathology detection. |
| **MedMNIST v2 (PneumoniaMNIST ResNet-18)** | Zenodo / Apache-2.0 | Pediatric pneumonia binary (28x28) | **Considered:** Highly robust benchmark, but low spatial resolution (28x28) limits realistic radiograph display. |
| **TorchXRayVision DenseNet-121 (`densenet121-res224-all`)** | GitHub / Hugging Face (Apache-2.0) | 18 chest pathologies (224x224) | **Selected:** Gold-standard academic benchmark, trained on 6 major clinical datasets (NIH, CheXpert, MIMIC, PadChest, OpenI, RSNA), lightweight (28 MB), verified PyTorch CPU/GPU inference. |

### Selected Architecture Justification
`densenet121-res224-all` from TorchXRayVision was selected because:
- **Verified Provenance:** Developed by Cohen et al. (MIDL 2022) with open-access weights hosted on GitHub Releases and Hugging Face Hub.
- **Permissive Open-Source Licensing:** Apache 2.0 license covers both library code and pretrained model weights.
- **Multitask Radiographic Coverage:** Predicts 18 distinct radiological classes covering parenchymal, pleural, and mediastinal abnormalities.
- **Empirical Operating Thresholds:** Includes precomputed operating-point decision thresholds evaluated across multi-center validation cohorts.
- **Modest Computational Footprint:** 28.38 MB checkpoint executes seamlessly on standard consumer CPUs without requiring a dedicated GPU.

---

## 3. Preprocessing & Inference Implementation

### Preprocessing Architecture (`image_utils.py`)
Chest radiographs differ substantially from standard consumer photographs. They represent radiographic attenuation rather than surface reflectance:
1. **Validation:** Checks byte size ($\le 8\text{ MB}$), file extensions (`.png`, `.jpg`, `.jpeg`), and Pillow image integrity via in-memory buffer decoding. Rejects images smaller than $32 \times 32$ or corrupted streams.
2. **Grayscale Standardization:** Converts RGB/RGBA uploads to single 8-bit channel mode (`L`).
3. **Dynamic Range Transformation:** Maps standard $[0, 255]$ pixel values into the dynamic range $[-1024, 1024]$ required by TorchXRayVision:
   $$x_{\text{normalized}} = \left(\frac{x}{255}\right) \times 2048 - 1024$$
4. **Spatial Cropping & Interpolation:** Applies `XRayCenterCrop()` to preserve diagnostic lung field aspect ratios, followed by `XRayResizer(224)` for bilinear spatial resampling to $224 \times 224$ pixels.
5. **Tensor Assembly:** Outputs a float32 tensor of shape `(1, 1, 224, 224)`.

### Inference Architecture (`model_utils.py`)
- **Evaluation Discipline:** Model is explicitly set to `model.eval()`, and all forward passes are wrapped in `torch.no_grad()`.
- **Validation:** Enforces 4D tensor constraints `(batch, 1, 224, 224)` prior to calling the network.
- **Dual-Metric Output Generation:**
  - **Operating-Point Normalized Score:** Piecewise linear scaling around the empirical operating point $\theta_i$, mapping to $[0, 1]$ where $\ge 0.50$ signifies meeting or exceeding the operating threshold. This is an uncalibrated model score, not a clinical probability.
  - **Raw Sigmoid Output:** $\sigma(z_i) = \frac{1}{1 + e^{-z_i}}$. Uncalibrated neural network output.
- **Error Handling:** Gracefully handles missing weights, network timeouts, invalid shapes, and hardware exceptions.

---

## 4. Automated Testing Methodology & Actual Results

Automated unit and integration tests were developed using `pytest`. Tests are segregated into:
1. **Mock Unit Tests:** Fast, isolated validation of image bounds, invalid formats, corrupted bytes, channel dimensions, and model error handlers without requiring large file downloads or GPU resources.
2. **Real-Inference Smoke Test:** End-to-end verification of actual pretrained DenseNet-121 weight loading, tensor forward pass, and output validation on a real sample.

### Actual Automated Test Results
```text
============================= test session starts =============================
platform win32 -- Python 3.13.1, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\bolla\Downloads\MediScan-AI
collected 21 items

tests/test_image_utils.py::TestImageValidation::test_valid_png_image PASSED          [  4%]
tests/test_image_utils.py::TestImageValidation::test_valid_jpeg_image PASSED         [  9%]
tests/test_image_utils.py::TestImageValidation::test_empty_input_rejected PASSED     [ 14%]
tests/test_image_utils.py::TestImageValidation::test_oversized_input_rejected PASSED [ 19%]
tests/test_image_utils.py::TestImageValidation::test_unsupported_extension_rejected PASSED [ 23%]
tests/test_image_utils.py::TestImageValidation::test_corrupted_image_rejected PASSED [ 28%]
tests/test_image_utils.py::TestImageValidation::test_invalid_dimensions_too_small PASSED [ 33%]
tests/test_image_utils.py::TestImageValidation::test_case_insensitive_extension PASSED [ 38%]
tests/test_image_utils.py::TestImagePreprocessing::test_preprocessing_output_shape PASSED [ 42%]
tests/test_image_utils.py::TestImagePreprocessing::test_preprocessing_rgb_to_single_channel PASSED [ 47%]
tests/test_image_utils.py::TestImagePreprocessing::test_preprocessing_normalization_range PASSED [ 52%]
tests/test_model_utils.py::TestModelValidationAndInference::test_validate_input_tensor_valid PASSED [ 57%]
tests/test_model_utils.py::TestModelValidationAndInference::test_validate_input_tensor_non_tensor PASSED [ 61%]
tests/test_model_utils.py::TestModelValidationAndInference::test_validate_input_tensor_invalid_dimensions PASSED [ 66%]
tests/test_model_utils.py::TestModelValidationAndInference::test_validate_input_tensor_invalid_channels PASSED [ 71%]
tests/test_model_utils.py::TestModelValidationAndInference::test_validate_input_tensor_invalid_spatial_resolution PASSED [ 76%]
tests/test_model_utils.py::TestModelValidationAndInference::test_run_inference_with_mock_model PASSED [ 80%]
tests/test_model_utils.py::TestModelValidationAndInference::test_score_interpretation_and_threshold_logic PASSED [ 85%]
tests/test_model_utils.py::TestModelValidationAndInference::test_inference_exception_handling PASSED [ 90%]
tests/test_model_utils.py::TestModelValidationAndInference::test_unexpected_output_shape_handling PASSED [ 95%]
tests/test_model_utils.py::TestRealPretrainedModelInference::test_real_model_loading_and_inference PASSED [100%]

======================= 21 passed, 8 warnings in 31.63s =======================
```
**Outcome:** 21 out of 21 tests passed successfully (100% pass rate).

---

## 5. Engineering Challenges & Debugging Decisions

### Challenge 1: Windows Console Encoding in TorchXRayVision
- **Issue:** During the initial download of weights via `xrv.models.DenseNet(weights="...")`, the library's utility attempted to print a progress bar with a unicode block character `\u2588`, which caused a `UnicodeEncodeError: 'charmap' codec can't encode character` under the default Windows `cp1252` encoding.
- **Resolution:** Implemented an independent, resilient weight downloader in `model_utils.py` (`ensure_weights_available`) utilizing Python's `urllib.request` that bypasses console encoding issues and verifies file size integrity before model instantiation.

### Challenge 2: Distinction Between Model Scores and Calibrated Clinical Probabilities
- **Issue:** Neither raw sigmoid outputs nor `op_norm` rescaled outputs are calibrated probabilities of patient disease. Raw sigmoid outputs from binary cross-entropy on imbalanced data represent raw network activations, while `op_norm` is a piecewise linear rescaling designed solely to center operating thresholds at 0.50. Labeling either as "calibrated disease probabilities" is scientifically inaccurate and clinically misleading.
- **Resolution:** Refactored the output interpretation schema across `model_utils.py`, `app.py`, and test suites to strictly use neutral mathematical terminology: "Normalized Model Score" (or "Operating Score") and "Raw Sigmoid Output". Added clear educational notices documenting that these values are uncalibrated model scores and cannot be interpreted as posterior probabilities of disease.

---

## 6. Known Limitations & Future Improvements

### Limitations
1. **Lack of Clinical Clearance:** Strictly an educational demonstration.
2. **Label Noise:** Training datasets utilized automated report parsing with known label noise.
3. **No Multimodal Context:** Scans are evaluated without clinical history, symptoms, or comparative historical radiographs.

### Future Improvements
1. **Explainability & Saliency Maps:** Integration of Grad-CAM or integrated gradients to visualize anatomical regions influencing each prediction.
2. **DICOM Ingestion Support:** Support for native `.dcm` files with direct extraction of 12-bit/16-bit Hounsfield or optical density units and clinical windowing tags.
3. **Multi-View Integration:** Simultaneous analysis of frontal (PA/AP) and lateral views for improved 3D spatial reasoning.
