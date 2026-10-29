# System Limitations & Clinical Constraints — MediScan AI

MediScan AI is strictly an educational research prototype. Understanding the fundamental limitations of automated medical computer vision systems is critical for engineers, data scientists, and healthcare observers.

---

## 1. Educational Purpose & Absence of Clinical Validation
- **Non-Diagnostic:** MediScan AI has not been evaluated, approved, or cleared by any regulatory body (such as the US FDA, European CE-MDR, or CDSCO).
- **No Diagnostic Intent:** The application is built to demonstrate deep learning inference pipelines, tensor transformations, and UI integration. It must not be deployed in clinical environments or used to guide patient triage, diagnosis, or therapy.

---

## 2. Radiographic Label Noise & Dataset Artifacts
- **Automated Label Mining:** The training corpora for multi-dataset chest X-ray models (such as NIH ChestX-ray14 and MIMIC-CXR) relied on natural language processing (NLP) algorithms applied to historical radiology text reports. These mining tools have an inherent error rate (reported between 5% and 15% across different pathology classes).
- **Shortcut Learning:** Deep convolutional networks are known to identify non-anatomical visual shortcuts, including:
  - Hospital-specific orientation markers (e.g., lead 'L' or 'R' markers, 'PORTABLE' stamps).
  - Radiographic artifacts caused by chest tubes, central lines, ECG leads, or pacemaker hardware.
  - Scanner-specific contrast profiles that correlate with patient acuity rather than pathology.

---

## 3. Domain Shift & Generalization Limits
- **Population Variance:** Model performance degrades when presented with demographic distributions, age brackets (e.g., pediatric or neonatal anatomy), or comorbidities not represented in the training corpora.
- **Acquisition Variability:** Image acquisition factors significantly alter feature representation:
  - Patient positioning (Posteroanterior [PA] vs. Anteroposterior [AP] vs. Lateral decubitus).
  - Degree of inspiration (under-inflated lungs mimic diffuse infiltration or basilar atelectasis).
  - Exposure factors (over-penetrated vs. under-penetrated X-rays distort bone and soft-tissue contrast).

---

## 4. Probabilistic Confidence vs. Clinical Certainty
- **Confidence Calibration:** Neural network output values ($\sigma(z)$) represent statistical likelihoods under a specific mathematical optimization function (Binary Cross-Entropy), **not** the clinical probability that a patient has a condition.
- **Threshold Sensitivity:** Radiographic findings operate at varied empirical thresholds. A high numerical probability can occur in an ambiguous scan due to over-confidence, while genuine subtle pathology (e.g., early-stage pneumonia or small pneumothorax) may produce a low score.
- **Independence Assumption:** In multi-label sigmoid classification, output classes are treated as mathematically independent, whereas anatomical pathologies in human physiology are heavily interrelated (e.g., congestive heart failure simultaneously manifesting as cardiomegaly, pulmonary edema, and pleural effusion).

---

## 5. False Positives and False Negatives
- **False Positives:** May induce unnecessary anxiety, costly redundant imaging, or inappropriate interventions if acted upon clinically.
- **False Negatives:** In a clinical context, a false negative represents a missed finding (e.g., missed pneumothorax or malignancy), with severe potential consequences.
- **Missing Multimodal Data:** Human radiologists evaluate medical imaging alongside vital signs, patient history, blood chemistry, and chronological comparison with prior scans. An isolated 2D classifier lacks this holistic context.

---

## 6. Technical, Hardware, and Privacy Constraints
- **Hardware Footprint:** Inference requires PyTorch and TorchXRayVision. On CPU-only environments, forward passes take approximately 0.5 to 1.5 seconds per 224x224 image.
- **Privacy Assurance:** MediScan AI processes images entirely in-memory (`io.BytesIO`). It does not persist uploaded scans to disk, transmit patient metadata to external third-party servers, or maintain patient logs.

---

## 7. Prerequisites for Clinical AI Translation
Before any medical vision algorithm can transition toward clinical trials or decision support, the following rigorous milestones are required:
1. Multi-center prospective clinical trials evaluating diagnostic accuracy against expert consensus panels.
2. Robust out-of-distribution (OOD) detection mechanisms to flag unsupported or non-chest inputs.
3. Spatial localization and explainability verification (e.g., CheXlocalize, Grad-CAM, pixel-level bounding boxes).
4. Full compliance with medical software standards (IEC 62304, ISO 14971, HIPAA, GDPR).
