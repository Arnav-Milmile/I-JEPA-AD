# Project Synopsis
**Project Title:** I-JEPA-AD: Decoder-Free Latent Anomaly Detection and Localization for Industrial Inspection
**Academic Level:** B.Tech Final Year Capstone Project (Computer Science & Engineering)
**Target Hardware:** Single NVIDIA RTX 4060 (8 GB VRAM)

---

## 1. Background and Motivation
Automated surface anomaly detection and localization are vital for industrial quality control, manufacturing reliability, and high-throughput production lines. Conventional Unsupervised Anomaly Detection (UAD) frameworks rely primarily on pixel-reconstruction methods, including Autoencoders (AEs), Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Masked Autoencoders (MAEs). These systems train solely on defect-free nominal images under the assumption that anomalous patterns cannot be accurately reconstructed. 

However, generative pixel-space reconstruction suffers from major practical bottlenecks:
1. **Computational Inefficiency:** Generative decoders require massive parameter overhead and high floating-point operations (FLOPs), severely limiting deployment on resource-constrained factory edge devices.
2. **High-Frequency Vulnerability:** Pixel decoders spend substantial representational capacity reconstructing irrelevant high-frequency textures, illumination shifts, and background noise rather than structural semantics.
3. **The Identity Trap:** Overly expressive generative decoders frequently generalize to out-of-distribution patterns, successfully reconstructing the anomaly and thereby failing detection.

Yann LeCun’s Joint Embedding Predictive Architecture (I-JEPA) eliminates generative pixel decoders by predicting missing image information strictly within an abstract latent representation space. This project investigates the application of frozen I-JEPA representations to industrial visual anomaly detection. By replacing pixel reconstruction with contextual latent prediction and introducing an adaptive coarse-to-fine masking mechanism, the proposed framework aims to deliver fast, highly accurate spatial defect localization on consumer-grade hardware (NVIDIA RTX 4060 8 GB).

---

## 2. Problem Statement
Current industrial anomaly localization systems either rely on heavy generative decoders that cannot sustain edge-level inference speeds or depend on memory-heavy nearest-neighbor feature banks (e.g., PatchCore) that scale poorly with dataset size. Furthermore, attempting dense sliding-window masked inference across latent representations introduces severe computational redundancies, executing unnecessary forward evaluations on large nominal image regions. 

The primary research objective is to answer:
> *Can a decoder-free, coarse-to-fine adaptive latent prediction framework achieve comparable or superior anomaly localization accuracy while significantly reducing the number of I-JEPA forward evaluations compared to dense overlapping masked inference on an 8 GB VRAM compute budget?*

---

## 3. Objectives
* **Develop a Decoder-Free Anomaly Engine:** Implement a latent anomaly scoring pipeline utilizing a frozen pre-trained I-JEPA encoder and a lightweight Transformer predictor trained exclusively on nominal industrial images.
* **Implement Offline Latent Caching:** Build an offline embedding cache for nominal training data, decoupling frozen encoder passes from predictor optimization to fit training strictly within an 8 GB VRAM envelope.
* **Design a Coarse-to-Fine Adaptive Masking Algorithm:** Engineer a two-stage deterministic inference mechanism that applies coarse evaluation across the image and conditionally deploys fine-grained masks only over suspicious anomalous candidate regions ($S_i > \tau$).
* **Perform Multi-Scale Score Fusion:** Formulate an anomaly map fusion strategy:
  $$S_{\text{final}} = S_{\text{coarse}} + \lambda S_{\text{fine}}$$
  to balance coarse context capture and fine boundary delineation.
* **Benchmark Localization vs. Computational Cost:** Rigorously evaluate Image AUROC, Pixel AUROC, PRO/AUPRO, average mask counts, and inference latency on the MVTec AD dataset against PatchCore, MAE baselines, and dense-masking variants.

---

## 4. Literature Review and Research Gap

| Sr No | Authors & Year | Paper Title | Methodology / Key Contribution | Limitation / Research Gap |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Assran et al., 2023 | Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture | Proposes I-JEPA; predicts representations of target blocks from context blocks without pixel decoding. | Foundational self-supervised work; does not address anomaly detection or spatial localization. |
| 2 | He et al., 2022 | Masked Autoencoders Are Scalable Vision Learners | Reconstructs missing image patches in pixel space via an asymmetric encoder-decoder. | Heavy pixel decoder introduces substantial latency and models non-essential high-frequency noise. |
| 3 | Roth et al., 2022 | Towards Total Recall in Industrial Anomaly Detection (PatchCore) | Employs neighborhood patch memory banks with greedy coreset subsampling for fast nearest-neighbor scoring. | Memory consumption scales with dataset size; lacks explicit contextual predictive structure. |
| 4 | Bergmann et al., 2019 | MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection | Establishes the standard benchmark dataset and pixel-level ground truth evaluation protocols. | Dataset benchmark paper; does not propose predictive architectures. |
| 5 | Akcay et al., 2022 | Anomalib: A Deep Learning Library for Anomaly Detection | Standardized benchmarking library for unsupervised anomaly detection algorithms. | Standard repository lacks native support for JEPA-based latent predictive masking pipelines. |
| 6 | Batzner et al., 2024 | Efficient Anomaly Detection via Masked Vision Transformers | Explores masked representations for fast anomaly detection by pruning decoder blocks. | Still retains RGB reconstruction elements rather than pure latent-space discrepancy modeling. |

**Identified Research Gap:** While masked modeling is widely recognized for feature learning, existing anomaly detection literature remains divided between memory-bank nearest neighbors (PatchCore) and compute-heavy generative pixel decoders (MAE-AD). Purely latent predictive architectures (I-JEPA) remain unoptimized for spatial anomaly localization, and uniform dense masking creates significant inference latency. A dedicated coarse-to-fine adaptive masking strategy that preserves localization fidelity while minimizing latent forward passes has not been systematically investigated.

---

## 5. Proposed Methodology / Plan of Work

### System Architecture
The proposed **I-JEPA-AD** framework decouples image encoding from predictive evaluation:
1. **Pre-computation & Offline Caching:** Nominal training images from MVTec AD are processed through the frozen Meta I-JEPA ViT-g/16 Context and Target Encoders. The resulting 1408-dimensional spatial patch embeddings are cached to disk to allow lightweight predictor training without repeated encoder forward passes.
2. **Predictor Optimization:** A lightweight Transformer Predictor (2–4 layers, 4–8 heads) is trained on cached nominal embeddings. Given context representations and target positional mask tokens, it learns to predict the expected nominal target representation using an $L_2$ or Cosine loss:
   $$L_{\text{pred}} = 1 - \cos(z^{\text{pred}}, z^{\text{target}})$$
3. **Stage 1 (Coarse Inference):** The test image is passed through the frozen I-JEPA encoder. A grid of non-overlapping coarse masks is applied. The predictor forecasts the coarse latent representation $\hat{z}^{\text{coarse}}$, and the coarse anomaly score is calculated:
   $$S_i = 1 - \cos(\hat{z}_i^{\text{coarse}}, z_i^{\text{actual}})$$
4. **Stage 2 (Adaptive Fine Inference):** For spatial regions where $S_i > \tau$ (predefined threshold), localized, higher-resolution fine masks are dynamically instantiated. Fine predictions $\hat{z}^{\text{fine}}$ are computed strictly for these candidate defect zones.
5. **Score Fusion & Heatmap Generation:** The anomaly map is fused via $S_{\text{final}} = S_{\text{coarse}} + \lambda S_{\text{fine}}$ and bilinearly upsampled to the original image dimensions for pixel-level evaluation.

    +-----------------------------------------------------------------------------+
    |                               TRAINING PHASE                                |
    | Nominal Image -> [Frozen I-JEPA Encoder] -> Latent Cache -> [Train Predictor]
    +-----------------------------------------------------------------------------+

                                   INFERENCE PHASE
                                   
                                   Input Test Image
                                          |
                               [Frozen I-JEPA Encoder]
                                          |
                                 Coarse Grid Masking
                                          |
                             [Lightweight Predictor]
                                          |
                          Calculate Coarse Discrepancy Map
                          S_i = 1 - cos(z_pred, z_actual)
                                          |
                         +----------------+----------------+
                         |                                 |
                  (S_i <= tau)                        (S_i > tau)
                         |                                 |
                  Nominal Region                   Suspicious Region
                 (Retain Coarse)                           |
                         |                        Generate Fine Masks
                         |                                 |
                         |                       [Lightweight Predictor]
                         |                                 |
                         |                         Fine Discrepancy
                         |                                 |
                         +----------------+----------------+
                                          |
                              Multi-Scale Score Fusion
                         S_final = S_coarse + lambda * S_fine
                                          |
                              Bilinear Spatial Upsampling
                                          |
                            Final Anomaly Heatmap / BBox

---

## 6. Technology, Tools, and Platforms
* **Programming Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch 2.2+, PyTorch Lightning, Automatic Mixed Precision (AMP / FP16)
* **Foundation Backbone:** Meta Open-Source I-JEPA ViT-g/16 ImageNet-22K checkpoint, frozen
* **Evaluation Framework:** Intel Anomalib integration (for AUROC, PRO, and benchmarking metrics)
* **Primary Dataset:** MVTec Anomaly Detection (MVTec AD) benchmark (15 industrial categories)
  * *Pilot Phase:* `bottle`, `capsule`, `hazelnut`
  * *Full Phase:* Complete 15 industrial texture & object classes
  * *Optional Robustness:* MVTec AD 2
* **Hardware Constraint:** Single NVIDIA RTX 4060 GPU (8 GB VRAM), local workstation

---

## 7. Expected Outcomes and Deliverables
* **Functional Software Artifact:** A complete, modular PyTorch pipeline featuring offline latent caching, predictor training, and adaptive coarse-to-fine inference modules.
* **Quantitative Performance:** Achieve competitive anomaly detection ($>95\%$ Image AUROC) and localization ($>90\%$ Pixel AUROC / PRO) on MVTec AD while reducing the number of I-JEPA forward passes by $\ge 50\%$ relative to dense sliding-window masking.
* **Empirical Trade-off Analysis:** Thorough ablation data documenting inference latency (ms), GPU memory usage (MB), and AUROC across varying refinement thresholds $\tau \in \{0.2, 0.3, 0.4, 0.5, 0.6, 0.7\}$.

---

## 8. Project Scope and Limitations
* **In Scope:** 2D industrial RGB images, unsupervised nominal-only training, structural and surface defect localization, and single-GPU optimization on 8 GB VRAM.
* **Out of Scope:** Multi-camera 3D point-cloud inspection, video stream analytics, and full end-to-end backpropagation through the heavy I-JEPA encoder backbone.
* **Identified Risks & Mitigations:**
  * *Small Defect Misses:* Addressed by tuning $\tau$ and evaluating multi-scale overlap ratios.
  * *VRAM Overflow:* Mitigated by caching latent tensors offline and using 224x224 input resolutions with FP16 precision.

---

## 9. Project Timeline (16-Week Schedule)

| Phase | Weeks | Milestone / Deliverables |
| :--- | :--- | :--- |
| **Phase 1: Setup & Data** | Weeks 1–2 | Environment configuration, Anomalib setup, MVTec AD data ingestion. |
| **Phase 2: Encoder & Caching** | Weeks 3–4 | Integrate I-JEPA checkpoints, verify tensor dimensions, build latent cache pipeline. |
| **Phase 3: Predictor Training** | Weeks 5–6 | Implement lightweight Transformer Predictor; train on nominal cache (`bottle`, `capsule`, `hazelnut`). |
| **Phase 4: Baselines Setup** | Weeks 7–8 | Implement Baseline A (Fixed-Mask) and Baseline B (Dense Sliding-Mask) anomaly heatmaps. |
| **Phase 5: Adaptive Masking** | Weeks 9–10 | Implement Stage 1 Coarse + Stage 2 Fine adaptive logic and thresholding engine. |
| **Phase 6: Comparative Benchmarking** | Weeks 11–12 | Run dense vs. adaptive comparisons; record latency, memory, and mask evaluation counts. |
| **Phase 7: Full Dataset Expansion** | Weeks 13–14 | Scale experiments across all 15 MVTec AD categories; conduct threshold ($\tau$) ablations. |
| **Phase 8: Documentation & Thesis** | Weeks 15–16 | Qualitative visualization generation, final report writing, code release prep. |

---

## 10. References
1. M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas, "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 15619–15629.
2. K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, "Masked Autoencoders Are Scalable Vision Learners," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 16000–16009.
3. P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 9592–9600.
4. K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, and P. Gehler, "Towards Total Recall in Industrial Anomaly Detection," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 14318–14328.
5. I. Akcay, D. Ameln, A. Vaidya, B. Lakshmanan, N. Ahuja, and U. Genc, "Anomalib: A Deep Learning Library for Anomaly Detection," in *Proc. IEEE Int. Conf. Data Eng. (ICDE)*, 2022, pp. 1706–1711.
6. T. Defard, A. Setkov, A. Loesch, and R. Romagnoli, "PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization," in *Int. Conf. Pattern Recognit. (ICPR) Workshops*, 2021, pp. 475–489.
7. R. Zavrtanik, M. Kristan, and D. Skočaj, "DRAEM — A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 8330–8339.
8. Y. LeCun, "A Path Towards Autonomous Machine Intelligence (Version 0.9.2)," *OpenReview*, Tech. Rep., 2022.
