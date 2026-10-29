# Model Card — TorchXRayVision DenseNet-121 (`densenet121-res224-all`)

## 1. Model Details
- **Model Name:** TorchXRayVision DenseNet-121 (`densenet121-res224-all`)
- **Architecture:** Dense Convolutional Network (DenseNet-121; Huang et al., CVPR 2017)
- **Primary Authors/Maintainers:** Joseph Paul Cohen, Joseph D. Viviano, Paul Bertin, et al.
- **Academic Publication:** Cohen et al., *"TorchXRayVision: A library of chest X-ray datasets and models"*, Proceedings of Medical Imaging with Deep Learning (MIDL), PMLR 172:231-249, 2022.
- **Repository:** [https://github.com/mlmed/torchxrayvision](https://github.com/mlmed/torchxrayvision)
- **Model Weights Mirror:** [https://huggingface.co/torchxrayvision/densenet121-res224-all](https://huggingface.co/torchxrayvision/densenet121-res224-all)
- **Direct Weight URL:** `https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt`
- **Model Weight File Size:** 28,382,008 bytes (~28.38 MB)
- **License:** Apache License 2.0 (open source, permissive educational and research use).

---

## 2. Intended Use & Target Task
- **Intended Purpose:** Educational demonstration, academic research, benchmarking, and study of convolutional neural network feature representations for chest radiographs.
- **Target Task:** Multi-label classification of 18 radiographic pathologies and anatomical findings from 2D frontal chest X-ray images (posteroanterior [PA] and anteroposterior [AP] views).
- **Out-of-Scope & Prohibited Uses:**
  - Clinical diagnostic decision-making.
  - Triage or primary screening in healthcare facilities.
  - Autonomous treatment recommendations or patient management.
  - Evaluation of non-chest radiographs (e.g., bone fractures outside the thorax, abdominal X-rays, mammography, dental CT, MRI).

---

## 3. Training Datasets & Provenance
The `"all"` weight variant was trained across multiple major multi-center radiological datasets:
1. **NIH ChestX-ray14 (NIH Clinical Center):** 112,120 frontal chest X-rays from 30,805 unique patients labeled using NLP mined from radiological text reports.
2. **CheXpert (Stanford ML Group):** 224,316 chest radiographs from 65,240 patients collected at Stanford Hospital.
3. **MIMIC-CXR (Beth Israel Deaconess Medical Center):** 377,110 chest X-rays associated with 227,835 imaging studies.
4. **PadChest (University of Alicante / BIMCV):** 160,000 chest radiographs from 67,000 patients with detailed Spanish radiologist reports.
5. **Google OpenI (Indiana University):** Open-access chest radiograph collection from the Indiana Network for Patient Care.
6. **RSNA Pneumonia Challenge (Kaggle/RSNA):** Annotated subset of NIH ChestX-ray14 with radiologist-verified lung opacities.

---

## 4. Input Specifications & Preprocessing
- **Modality:** Frontal projection Chest Radiograph (PA or AP view).
- **Color Format:** Grayscale (single channel).
- **Intensity Normalization:** Scaled from 8-bit dynamic range $[0, 255]$ to the range $[-1024, 1024]$ using:
  $$x_{\text{normalized}} = \left(\frac{x}{255}\right) \times 2048 - 1024$$
- **Spatial Transformation:** Aspect-ratio preserving center crop followed by bilinear interpolation to $224 \times 224$ pixels.
- **Input Tensor Dimensions:** `(batch_size, 1, 224, 224)` of data type `torch.float32`.

---

## 5. Output Pathologies (18 Classes)
The model outputs simultaneous predictions for 18 distinct radiological classes:
1. `Atelectasis`
2. `Consolidation`
3. `Infiltration`
4. `Pneumothorax`
5. `Edema`
6. `Emphysema`
7. `Fibrosis`
8. `Effusion`
9. `Pneumonia`
10. `Pleural_Thickening`
11. `Cardiomegaly`
12. `Nodule`
13. `Mass`
14. `Hernia`
15. `Lung Lesion`
16. `Fracture`
17. `Lung Opacity`
18. `Enlarged Cardiomediastinum`

---

## 6. Output Calibration & Operating Scores
- **Raw Output:** Raw linear logits $z_i \in (-\infty, \infty)$ converted to sigmoid probabilities $\sigma(z_i) \in [0, 1]$.
- **Operating Point Normalization:** Because individual pathologies occur at vastly different natural prevalences (e.g., Pneumothorax $\approx 1\%$, Cardiomegaly $\approx 8\%$), raw probabilities cannot be thresholded at a naive 0.50. TorchXRayVision computes empirical operating thresholds $\theta_i$ for each pathology based on validation F1/ROC performance.
- **MediScan AI Normalization:** Outputs are mapped into a standardized $[0.0, 1.0]$ score where $0.50$ represents the calibrated decision threshold:
  - $\text{Score} \ge 0.50$: Model detects radiographic patterns exceeding the calibrated sensitivity/specificity operating threshold for that pathology.
  - $\text{Score} < 0.50$: Radiographic features remain below the operating threshold.

---

## 7. Known Limitations & Potential Biases
1. **Report Extraction Noise:** Ground truth labels in source datasets (NIH, CheXpert, MIMIC) were largely derived by automated NLP parsing of unstructured radiologist free-text reports, introducing label noise (estimated 5–15% in published literature).
2. **Confounding Artifacts:** Chest radiographs frequently contain support devices (pacemakers, catheters, chest tubes) and physical positioning markers ('PA', 'AP', 'PORTABLE') that models can unintentionally correlate with disease presence (shortcut learning).
3. **Demographic & Equipment Domain Shift:** Performance varies across scanner manufacturers, patient age groups, body habitus, exposure parameters, and radiological projection angles.
4. **Pediatric Inapplicability:** The training datasets consist predominantly of adult patients. Performance on pediatric patients is unverified.
5. **No Clinical Context:** Radiologists interpret chest X-rays in combination with patient history, physical examination, laboratory values, and previous imaging. An image-only classifier lacks this essential clinical context.

---

## 8. Non-Diagnostic Medical Disclaimer
Outputs generated by this model are solely for educational exploration of computer vision methodologies. They do not constitute diagnostic medical advice, clinical impressions, or treatment recommendations.
