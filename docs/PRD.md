# Product Requirements Document (PRD)

**Project Title:** I-JEPA-AD: Decoder-Free Latent Anomaly Detection and Localization Engine  
**Document Version:** 1.0.0  
**Target Environment:** Local Edge-Workstation (NVIDIA RTX 4060, 8 GB VRAM, PyTorch 2.x)

---

## 1. Executive Summary & Product Vision
I-JEPA-AD is an unsupervised visual anomaly detection and spatial localization framework engineered for industrial quality assurance. By replacing compute-heavy pixel-level generative decoders with a pre-trained, frozen Joint Embedding Predictive Architecture (I-JEPA) and a lightweight Transformer Predictor, the system predicts normal latent states from contextual image patches. 

To eliminate the inference bottleneck of dense overlapping sliding masks, I-JEPA-AD introduces a deterministic **Coarse-to-Fine Adaptive Masking Strategy**. The system achieves high-precision defect localization while minimizing forward latent evaluations and running fully within an 8 GB VRAM hardware budget.

---

## 2. Target User & Operational Environment
* **Primary Persona:** Computer Vision / QA Engineer deploying inspection models on factory-floor workstations or embedded edge GPUs.
* **Operating Constraints:**
  * Strict memory envelope: $\le 8\text{ GB VRAM}$ (NVIDIA RTX 4060).
  * Zero reliance on external cloud APIs.
  * Unsupervised regime: Training data contains **zero defect samples**.

---

## 3. Core Research Questions (Verification Goals)
* **RQ1:** Can I-JEPA latent contextual prediction detect industrial anomalies without decoding to RGB pixel space?
* **RQ2:** Can latent prediction discrepancy ($1 - \cos(z^{\text{pred}}, z^{\text{actual}})$) provide fine spatial localization?
* **RQ3:** Does coarse-to-fine adaptive masking measurably reduce computational cost compared to dense sliding masks?
* **RQ4:** What is the empirical trade-off between localization accuracy (Pixel AUROC/PRO) and latency across refinement thresholds $\tau$?
* **RQ5:** How does the coarse-to-fine pipeline perform across varying defect sizes (small pinholes vs. large structural tears)?

---

## 4. System Architecture & Component Specifications

                     +---------------------------------------+
                     |         Input Image (224x224)         |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |       Frozen I-JEPA ViT Encoder       |
                     +---------------------------------------+
                                         |
                       [Stage 1: Coarse Grid Extraction]
                                         |
                                         v
                     +---------------------------------------+
                     |    Lightweight Transformer Predictor  |
                     |       (2-4 Layers, 4-8 Heads)         |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |    Coarse Discrepancy Map Compute     |
                     |      S_i = 1 - cos(z_hat, z)          |
                     +---------------------------------------+
                                         |
                                   [Threshold tau]
                                         |
                     +-------------------+-------------------+
                     |                                       |
             (S_i <= tau: Normal)                    (S_i > tau: Defect)
                     |                                       |
              Bypass Stage 2                         Stage 2 Fine Masking
                     |                                       |
                     |                               [Run Predictor]
                     |                                       |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |   Multi-Scale Score Map Fusion        |
                     |   S_final = S_coarse + lambda*S_fine  |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |      Bilinear Spatial Heatmap         |
                     +---------------------------------------+


### 4.1. Frozen I-JEPA Encoder
* **Model Checkpoint:** Official Meta AI I-JEPA ViT-B/16 or ViT-L/16 (ImageNet-1K pre-trained).
* **Execution State:** Strictly frozen (`eval()` mode, `requires_grad = False`).
* **Input Resolution:** $224 \times 224 \times 3$.
* **Output:** Patch latent representations $Z \in \mathbb{R}^{N \times D}$, where $N = (224/16)^2 = 196$ tokens, and $D$ is the embedding dimension.

### 4.2. Offline Latent Caching Subsystem
* **Purpose:** Eliminate repetitive encoder forward passes during training on nominal datasets.
* **Functionality:** 
  * Pass all nominal training images through the frozen encoder once.
  * Store resulting tensor dictionaries `(image_id, patch_latents)` to disk (HDF5 or PyTorch `.pt` format).
  * During training, load cached embeddings directly into memory, bypassing encoder execution entirely.

### 4.3. Lightweight Transformer Predictor
* **Architecture:**
  * Transformer Encoder Layers: 2 to 4 layers.
  * Multi-Head Attention: 4 to 8 attention heads.
  * Hidden Dimension: Matched or linearly projected to I-JEPA latent dimension $D$.
  * Positional Encoding: Learnable target mask position tokens.
* **Training Loss:**
  * Primary: Cosine Distance Loss $L_{\cos} = 1 - \frac{z^{\text{pred}} \cdot z^{\text{target}}}{\|z^{\text{pred}}\|_2 \|z^{\text{target}}\|_2}$
  * Ablation Alternative: Mean Squared Error / $L_2$ Loss $L_{L2} = \|z^{\text{pred}} - z^{\text{target}}\|_2^2$

---

## 5. Functional Requirements: Adaptive Masking Engine

### 5.1. Stage 1 — Coarse Latent Scan
* **Operation:** Partition the image into coarse spatial blocks (e.g., $4 \times 4$ patch blocks).
* **Prediction:** Context patches are passed to the predictor with masked target positions to compute $\hat{z}_i^{\text{coarse}}$.
* **Anomaly Map Calculation:**
  $$S_i = 1 - \cos(\hat{z}_i^{\text{coarse}}, z_i^{\text{actual}})$$

### 5.2. Stage 2 — Fine Adaptive Refinement
* **Gating Condition:** Evaluate all coarse regions $i$. If $S_i > \tau$, mark region $i$ as "Suspicious".
* **Dynamic Mask Generation:** Generate overlapping fine-scale masks (e.g., $1 \times 1$ or $2 \times 2$ patch resolution) exclusively over the suspicious spatial coordinates.
* **Selective Prediction:** Run the lightweight predictor only on the generated fine masks to calculate $S_j^{\text{fine}}$.

### 5.3. Anomaly Score Fusion & Upsampling
* **Score Integration:**
  $$S_{\text{final}}(x, y) = S_{\text{coarse}}(x, y) + \lambda \cdot S_{\text{fine}}(x, y)$$
  *(Default hyperparameter initialization: $\lambda = 1.0$, refined via ablation).*
* **Spatial Heatmap Construction:** Bilinearly interpolate the discrete 2D patch score matrix to the original $224 \times 224$ image space.

---

## 6. Non-Functional Requirements & Performance Constraints

| Constraint | Specification | Verification Method |
| :--- | :--- | :--- |
| **GPU VRAM Limit** | $\le 8.0\text{ GB}$ peak allocation during training and inference. | Continuous logging via `torch.cuda.max_memory_allocated()`. |
| **Precision** | Automatic Mixed Precision (AMP / FP16). | `torch.cuda.amp.autocast()` enabled across all modules. |
| **Inference Latency** | Measurably lower per-image latency than dense sliding-window masking. | Wall-clock execution profiling (ms/image) averaged over 100 runs. |
| **Evaluation Overhead** | $\ge 40\%\text{ reduction}$ in total mask predictions per image vs. dense baseline. | Explicit forward pass counter logging. |

---

## 7. Experimental Setup & Benchmarking Protocols

### 7.1. Datasets
* **Primary:** MVTec Anomaly Detection (MVTec AD) — 15 categories (5 textures, 10 objects).
  * **Pilot / Debug Set:** `bottle`, `capsule`, `hazelnut`.
  * **Full Rollout:** Remaining 12 categories after pipeline validation.
  * **Optional Generalization:** MVTec AD 2.

### 7.2. Baseline Comparisons

+-------------------------------------------------------------------------------+
|                             EXPERIMENTAL MATRIX                               |
+----------------------+--------------------------------------------------------+
| Baseline 1           | PatchCore (Memory Bank / k-NN standard)                |
| Baseline 2           | MAE Reconstruction Baseline (Pixel Decoder)            |
| Baseline 3           | Fixed-Mask I-JEPA Latent Detector                      |
| Baseline 4           | Dense Overlapping Masked I-JEPA                        |
| Proposed Model       | Coarse-to-Fine Adaptive Masked I-JEPA                  |
+----------------------+--------------------------------------------------------+

### 7.3. Primary Evaluation Metrics
* **Detection Performance:** Image-level AUROC ($\%$).
* **Localization Performance:** Pixel-level AUROC ($\%$) and Per-Region Overlap (PRO / AUPRO).
* **Efficiency Profile:** 
  * Average masks evaluated per test image.
  * Average inference latency per image (ms).
  * Peak GPU VRAM footprint (MB).

### 7.4. Ablation Matrix
* **Refinement Threshold:** $\tau \in \{0.2, 0.3, 0.4, 0.5, 0.6, 0.7\}$.
* **Loss Objective:** $L_{\cos}$ (Cosine distance) vs. $L_{L2}$ ($L_2$ Euclidean distance).
* **Predictor Depth:** 2 layers vs. 4 layers vs. 6 layers.
* **Fusion Weight:** $\lambda \in \{0.0, 0.5, 1.0, 1.5, 2.0\}$.

---

## 8. Risk Management Matrix

| Risk ID | Identified Risk Description | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **R-01** | Missing small/subtle defects during coarse Stage 1 scan. | High | Implement overlap in coarse masks; perform ablation over threshold $\tau \in [0.2, 0.7]$. |
| **R-02** | Adaptive overhead negates inference latency savings. | High | Utilize vectorized tensor masking; if savings are insufficient, report dense vs. adaptive as an ablation. |
| **R-03** | 8 GB GPU VRAM out-of-memory (OOM) on large batches. | Critical | Maintain frozen encoder, enforce offline latent caching, and use batch size $\le 16$ with AMP. |
| **R-04** | Predictor fails to learn normal distribution from frozen latents. | Medium | Validate loss convergence on 3 pilot categories (`bottle`, `capsule`, `hazelnut`) prior to scaling. |

---

## 9. 16-Week Implementation Roadmap

* **Weeks 1–2:** Ingest MVTec AD; configure PyTorch / Anomalib environment.
* **Weeks 3–4:** Integrate frozen I-JEPA ViT checkpoint; construct offline latent caching pipeline.
* **Weeks 5–6:** Build lightweight Transformer predictor; train on cached embeddings (`bottle`, `capsule`, `hazelnut`).
* **Weeks 7–8:** Implement fixed-mask and dense sliding-mask inference baselines.
* **Weeks 9–10:** Implement Stage 1 + Stage 2 Coarse-to-Fine adaptive masking algorithm.
* **Weeks 11–12:** Benchmark dense vs. adaptive masking (measure latency, FLOPs, mask counts).
* **Weeks 13–14:** Execute full benchmark across all 15 MVTec AD categories; conduct threshold ($\tau$) ablations.
* **Weeks 15–16:** Generate qualitative localization heatmaps; complete final thesis and code repository documentation.
    