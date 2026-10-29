# 🫁 MediScan AI — AI-Powered Medical Image Analysis Assistant

> **Assessment Context:** Developed for the **Trialshopy Technical Development Intern Assessment** in AI/ML, Generative AI, Computer Vision, and AR/VR.  
> **Submission Deadline:** 30 October 2026.  
> **Status:** Fully functional working prototype verified with genuine pretrained PyTorch model weights.

---

## 1. Project Overview & Problem Statement

Chest radiography is among the most common and vital diagnostic imaging modalities in medicine. However, automated image interpretation poses serious computer vision and safety challenges:
- Generic natural image classifiers (e.g. ImageNet models) fail on medical attenuation patterns.
- Many educational projects fabricate mock prediction percentages or invent unsubstantiated accuracy metrics.
- Improper preprocessing (such as improper dynamic range scaling or distortion of lung field aspect ratios) destroys diagnostic cues.

**MediScan AI** is an educational medical computer vision assistant built with Streamlit and PyTorch. It demonstrates genuine, verified deep learning inference on frontal chest radiographs using an established, open-source model: **TorchXRayVision DenseNet-121** (`densenet121-res224-all`), trained across six major clinical chest X-ray repositories.

---

## 2. Key Features

- **Genuine Pretrained Model Inference:** Runs real PyTorch forward passes on CPU/GPU using verified weights (`densenet121-res224-all`, 28.38 MB). No fabricated predictions, mock data, or hardcoded scores.
- **Radiology-Specific Preprocessing:** Implements exact intensity normalization ($[-1024, 1024]$) and aspect-ratio preserving center crop/resizing ($224 \times 224$ px).
- **18-Class Radiographic Evaluation:** Analyzes 18 distinct findings including Atelectasis, Cardiomegaly, Consolidation, Effusion, Infiltration, Pneumonia, and Pneumothorax.
- **Operating-Point Normalized Scores:** Rescales model outputs relative to precomputed empirical operating thresholds (0.50 = operating threshold) using TorchXRayVision's `op_norm`.
- **Safe In-Memory Processing:** Images are validated and processed in volatile memory (`io.BytesIO`). Zero persistence of patient images or private metadata.
- **Educational Sample Included:** Ships with a verified public-access educational sample radiograph from NIH ChestX-ray14 for instant testing without requiring private scans.
- **Prominent Educational Disclaimer:** Emphasizes that statistical model predictions are **not** clinical diagnoses and cannot replace qualified healthcare practitioners.
- **Comprehensive Automated Testing:** 21 automated unit and smoke tests covering validation, corrupted files, tensor transformations, mock inference, neutral score logic, and real-weight forward passes.

---

## 3. Technology Stack

- **Language:** Python 3.11 / 3.13 (compatible with 64-bit Windows, Linux, and macOS)
- **Web Application Framework:** Streamlit (v1.66.1)
- **Deep Learning Framework:** PyTorch (v2.15.0) & Torchvision (v0.29.2)
- **Medical Vision Library:** TorchXRayVision (v1.5.7)
- **Image Processing & Numerical Computation:** Pillow (v11.0.0), NumPy (v2.2.0), Scikit-Image (v0.26.0)
- **Automated Testing:** Pytest (v9.1.1)

---

## 4. Repository Structure

```text
MediScan-AI/
├── app.py                      # Main Streamlit web application & UI workflow
├── model_utils.py              # Pretrained model loading, inference & calibration
├── image_utils.py              # In-memory image validation & radiographic preprocessing
├── requirements.txt            # Pinned dependency specifications
├── README.md                   # Comprehensive project documentation & guide
├── .gitignore                  # Git exclusions for venv, caches, and weights
├── samples/
│   └── nih_sample_00000001_000.png # Verified public-domain NIH test radiograph
├── tests/
│   ├── __init__.py
│   ├── test_image_utils.py     # Automated tests for image validation & preprocessing
│   └── test_model_utils.py     # Automated tests for model loading, inference & error handling
└── docs/
    ├── architecture.md         # System workflow, Mermaid diagrams, & modular design
    ├── technical_report.md     # Engineering report, challenges, & test results
    ├── model_card.md           # Model provenance, datasets, license, & specifications
    └── limitations.md          # Comprehensive clinical & algorithmic constraints
```

---

## 5. Model Provenance & Specifications

- **Model Identifier:** `densenet121-res224-all`
- **Architecture:** DenseNet-121
- **Source Repository:** [mlmed/torchxrayvision](https://github.com/mlmed/torchxrayvision) (Joseph Paul Cohen et al., MIDL 2022)
- **Model Weights Mirror:** [Hugging Face — torchxrayvision/densenet121-res224-all](https://huggingface.co/torchxrayvision/densenet121-res224-all)
- **Weight Checkpoint Size:** 28,382,008 bytes (~28.38 MB)
- **License:** Apache License 2.0 (Permissive open source)
- **Training Datasets:** NIH ChestX-ray14, CheXpert, MIMIC-CXR, PadChest, Google OpenI, RSNA Pneumonia Challenge.
- **Input Specifications:** Grayscale frontal chest radiograph (PA or AP), normalized to $[-1024, 1024]$, shape `(1, 1, 224, 224)`.
- **Output Classes:** 18 Radiographic Pathologies.

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.11 or Python 3.13 (64-bit).
- Git installed.
- Internet access on initial launch to download pretrained weights (~28 MB).

### Step 1: Clone or Enter the Repository
```bash
git clone https://github.com/<your-username>/MediScan-AI.git
cd MediScan-AI
```

### Step 2: Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Pinned Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Weight Availability & Smoke Test
You can optionally preload the model weights and verify inference via:
```bash
python -c "import model_utils; model_utils.load_model(); print('Model weights verified!')"
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 7. Running the Automated Test Suite

Execute the full suite of 21 unit and integration tests:

```bash
pytest -v
```

### Actual Test Execution Output:
```text
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

---

## 8. Application Workflow & Screenshots

### Application Flow
1. **Upload / Sample Selection:** The user uploads a PNG/JPEG chest radiograph (up to 8 MB) or clicks the one-click educational sample button.
2. **Validation & Preview:** File byte headers and dimensions are validated. The image is rendered with decoded resolution and file size.
3. **Analysis:** The user clicks **Analyze Radiograph**. The preprocessing pipeline normalizes the image and passes it to the TorchXRayVision DenseNet-121 model.
4. **Results Presentation:**
   - Visual alert indicating the count of elevated findings.
   - Summary cards detailing elevated pathologies with calibrated scores and anatomical definitions.
   - Comprehensive interactive table covering all 18 evaluated conditions.
   - Preprocessing inspection tab explaining the mathematical transformations applied.

---

## 9. Educational-Use Disclaimer

> **IMPORTANT:**  
> MediScan AI is an educational demonstration of pretrained computer vision models. Its outputs are **not medical diagnoses** and must **not** be used for treatment decisions, clinical triage, or patient management. Model performance may vary depending on image quality, patient population, acquisition conditions, and the training data distributions. Always consult a qualified healthcare professional for medical interpretation.

---

## 10. Third-Party Libraries & Attributions

- **PyTorch:** Open-source machine learning framework (BSD-3 License).
- **TorchXRayVision:** Cohen et al. (MIDL 2022 / Apache License 2.0).
- **Streamlit:** Open-source Python app framework (Apache License 2.0).
- **Pillow (PIL):** Python Imaging Library (HPND License).
- **NumPy & Scikit-Image:** Scientific computing libraries (BSD License).
- **Pytest:** Testing framework (MIT License).
- **Sample Image:** NIH ChestX-ray14 public dataset released by the NIH Clinical Center.

---

## 11. AI Tools Used During Development

- **Antigravity AI (Google DeepMind):** Used as a senior AI/ML pair programmer for architectural design, code authoring, dependency resolution, test construction, and technical documentation.

---

## 12. GitHub Repository Setup Guide

To push this project to a new GitHub repository:

```bash
# 1. Initialize git
git init

# 2. Add files (ignoring .venv and temp files via .gitignore)
git add .

# 3. Create initial commit
git commit -m "feat: initial release of MediScan AI medical analysis assistant"

# 4. Link to your GitHub remote repository
git remote add origin https://github.com/<your-username>/MediScan-AI.git

# 5. Push to GitHub
git branch -M main
git push -u origin main
```
