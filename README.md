# Transfer Learning Benchmarks for Medical Image Classification

This repository benchmarks three modern parameter-efficient transfer learning techniques for medical image classification using a modular ResNet-50 backbone. These methods serve as baselines for evaluating a future General-Purpose Activation Function (GPAF) designed to improve CNN generalizability.

## Goal

- Develop a General-Purpose Activation Function (GPAF) to enhance the generalizability of pre-trained CNNs.
- Benchmark GPAF against state-of-the-art transfer learning strategies on diverse medical imaging tasks.
- Ensure efficient use of compute and memory, with minimal parameter overhead and strong performance on small datasets.

---

## Benchmark Methods

### 1. [LoRA-C: Parameter-Efficient Fine-Tuning of Robust CNN for IoT Devices](https://arxiv.org/abs/2410.16954)

- **Citation**: arXiv:2410.16954v1 (Oct 2024)
- **Goal**: Efficient adaptation of CNNs for edge deployment under compute/memory constraints.
- **Method**:
  - Inserts low-rank trainable matrices into convolution layers.
  - Freezes main model weights; trains a lightweight subspace.
  - Uses structured dropout, normalization, and adversarial robustness techniques.
- **Advantages**: Competitive accuracy with only ~0.1% trainable parameters.
- **Implementation**: Modular PyTorch wrappers around `Conv2d`.

---

### 2. [MetaLR: Meta-Tuning of Learning Rates for Transfer Learning](https://arxiv.org/abs/2206.01408)

- **Citation**: MICCAI 2023
- **Goal**: Improve convergence and generalization by learning per-parameter learning rates.
- **Method**:
  - Two-stage training: warm-up (head-only) → meta-tuning (adaptive learning rates).
  - Uses meta-gradients to adjust learning rates based on gradient trajectory.
- **Advantages**: More stable and efficient optimization in domain shift and low-data regimes.
- **Implementation**: Nested optimization loops; integrates with PyTorch optimizers.

---

### 3. [Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets](https://arxiv.org/abs/2208.07463)

- **Citation**: arXiv:2208.07463v4 (Aug 2022)
- **Goal**: Transfer learning via lightweight task-specific modules.
- **Method**:
  - Adds 3-layer bottleneck adapters (1×1 → 3×3 depthwise → 1×1) in parallel to convolution layers.
  - Keeps the ResNet backbone frozen; outputs added to residuals.
- **Advantages**: Competitive performance with only ~0.2% extra parameters.
- **Implementation**: Adapter modules are inserted into blocks with `stride=1`.

---

## 🧠 Datasets

| Dataset           | Domain        | Task                      | Classes | Modality          | Train / Val / Test       |
|------------------|---------------|---------------------------|---------|-------------------|--------------------------|
| ISIC 2018        | Dermatology   | Skin lesion classification| 7       | Dermoscopy (RGB)  | 6409 / 1602 / 2004       |
| Kaggle Brain MRI | Neuroimaging  | Tumor type classification | 4       | T1-weighted MRI   | 4494 / 1123 / 1406       |
| PathMNIST        | Histopathology| Tissue type classification| 9       | Microscopic Patches| 68,595 / 17,148 / 21,437 |

