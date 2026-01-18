# CNN-CIFAR10 - Research Notebook  
Focus: Efficient Deep Learning Systems, LLM Inference, Multimodal Models

This notebook records my **thinking process, experiments, bugs, ablations, and design decisions**.
The emphasis is on **clarity of reasoning and experimental discipline**, not presentation polish.

---

## Entry 1 — CNN on CIFAR-10  
**Theme:** Foundations, Optimization, and Generalization

### Date  
Day 1

---

### Motivation

I started with a CNN on CIFAR-10 to:

- Revisit core vision fundamentals
- Study training dynamics and generalization behavior
- Build a clean, reproducible baseline for later systems-level analysis

This experiment is meant as a **controlled reference point**, not a SOTA attempt.

---

### Dataset

- CIFAR-10  
- 50,000 training images, 10,000 test images  
- 32×32 RGB images  
- 10 classes (animals + vehicles)

---

### Architecture

Intentionally simple CNN:

Conv(3 → 32) → BatchNorm → ReLU → MaxPool  
Conv(32 → 64) → BatchNorm → ReLU → MaxPool  
Flatten (64 × 8 × 8 → 4096)  
FC(4096 → 128) → ReLU → Dropout  
FC(128 → 10)

**Design rationale:**
- BatchNorm stabilizes early optimization
- Pooling reduces spatial resolution progressively
- Dropout controls overfitting in dense layers
- No residuals or pretrained backbones to keep behavior interpretable

---

### Training Setup

- Optimizer: Adam  
- Learning rate: 1e-3  
- Batch size: 128  
- Loss: Cross-entropy  
- Epochs: 30  
- Deterministic seeding enabled  
- Metrics logged with Weights & Biases  

The goal here is **stability and reproducibility**, not aggressive tuning.

---

### Data Augmentation

Two regimes were evaluated:

**No augmentation**
- Raw CIFAR-10 images

**With augmentation**
- RandomCrop(32, padding=4)
- RandomHorizontalFlip

Motivation:
- CIFAR-10 is small
- Augmentation increases effective data diversity
- Acts as implicit regularization
- Encourages translation and reflection invariance

A strict ablation was performed:
- Same model
- Same seed
- Same optimizer
- Same epochs
- Only augmentation changed

---

### Results — Augmentation Ablation

**Final epoch (30) comparison:**

| Setup | Train Acc | Test Acc |
|-----|----------|----------|
| No Augmentation | 88.65% | 74.18% |
| With Augmentation | 69.11% | 76.25% |

---

### Observations

- No-augmentation model fits training data aggressively
- Large train–test gap indicates overfitting
- Augmented model trains more slowly
- Augmentation reduces memorization and improves test accuracy

Despite lower training accuracy, the augmented model generalizes better.

---

### Generalization Gap

| Setup | Train–Test Gap |
|------|---------------|
| No Augmentation | ~14% |
| With Augmentation | ~6% |

This gap reduction is the strongest evidence that augmentation is working as intended.

---

### Confusion Matrix Analysis

Observed trends:
- Vehicle classes (airplane, ship, truck) achieve higher accuracy
- Animal classes (cat, dog, deer, bird) are frequently confused

Interpretation:
- Vehicles are shape-dominant and consistent
- Animals show higher intra-class variation and texture dependence

Augmentation improves:
- cat
- deer
- frog
- airplane
- ship

Some classes (e.g., dog, horse) show limited gains, suggesting **model capacity limits rather than augmentation failure**.

---

### Debugging Notes

A tensor shape mismatch occurred when transitioning from convolutional layers to the first fully connected layer.

This reinforced an important lesson:

> Most deep learning bugs arise from implicit shape assumptions, not from algorithms.

Explicit tensor-shape reasoning significantly reduced debugging time.

---

### Key Insight

Accuracy alone is misleading.

What matters is:
- Generalization behavior
- Train–test gap
- Controlled ablations

Augmentation improved robustness, not memorization.

---

### Limitations (Intentional)

- No residual connections
- No learning-rate scheduling
- No weight decay
- No test-time augmentation

These omissions keep the baseline interpretable and analyzable.

---

## Entry 2 — Autograd & Gradient Flow  
**Theme:** PyTorch Internals

### Date  
Day 1

---

### Goal

To understand how PyTorch autograd works internally, instead of treating `.backward()` as a black box.

---

### Experiment

Implemented a 2-layer neural network **without `nn.Module`**:

- Manual tensor creation
- `requires_grad=True`
- Matrix multiplication + ReLU
- Scalar loss
- Manual `.backward()` call

---

### Observations

- Autograd builds a dynamic computation graph during forward pass
- Each tensor stores a reference to the operation that created it
- Gradients propagate backward via the chain rule
- `.grad` is populated only for leaf tensors

Clarified:
- Why in-place ops are dangerous
- Why `.detach()` breaks gradient flow
- Why gradients accumulate unless zeroed

---

### Why This Matters

Autograd understanding is essential for:
- Debugging unstable training
- Writing custom training loops
- Memory optimization (checkpointing)
- Reasoning about large-model efficiency

Many high-level bugs are actually autograd misunderstandings.

---

### Reflection

Framework familiarity without systems understanding leads to fragile models.

---

### Planned Extensions

- Visualize computation graphs
- Experiment with gradient clipping
- Explore gradient accumulation for large batch simulation

---

## Entry 3 — Error Analysis & Failure Modes  
**Theme:** Interpretability and Generalization

### Date  
Day 3

---

### Motivation

Aggregate accuracy metrics do not explain *why* a model fails.
To understand generalization behavior more deeply, I analyzed misclassified
test samples from both the augmented and non-augmented models.

---

### Method

- Collected misclassified images from the CIFAR-10 test set
- Visualized predictions alongside ground-truth labels
- Compared failure patterns between:
  - No-augmentation model
  - Augmentation-trained model

The analysis is qualitative and focused on identifying structured failure modes.

---

### Observations — Common Failure Patterns

Across both models, errors are **not random** and exhibit clear structure:

- **Class confusion among animals**
  - cat ↔ dog
  - deer ↔ horse  
  Likely due to similar body shapes and coarse texture cues at 32×32 resolution.

- **Background sensitivity**
  - birds against sky occasionally misclassified as airplanes
  - ships in water sometimes confused with airplanes  
  Indicates partial reliance on background context rather than object shape.

- **Pose and framing sensitivity**
  - Unusual poses
  - Partial occlusion
  - Off-center objects  
  These cases are more frequent in misclassifications.

---

### Augmentation vs No-Augmentation Behavior

Comparative visual inspection reveals:

- The **no-augmentation model** shows stronger sensitivity to:
  - Object centering
  - Fixed spatial positioning
  - Background cues

- The **augmented model** exhibits:
  - Improved robustness to translation and framing
  - Reduced dependence on background context
  - Fewer errors caused purely by object displacement

However, fine-grained texture confusions persist, suggesting
**model capacity limitations rather than data insufficiency**.

---

### Key Insight

Most model errors are **structured, not stochastic**.

Augmentation mitigates spatial and positional biases, but does not fully
resolve fine-grained semantic confusions, highlighting the distinction between:
- Data-centric improvements
- Architecture / capacity constraints

Understanding *how* a model fails provides more actionable insight
than marginal accuracy improvements.

---

### Updated Next Steps

- Cluster misclassifications by failure type
- Explore modest capacity increases (deeper CNN)
- Study whether additional invariances improve animal-class performance
- Use this baseline as a reference for sample-efficiency experiments

---

## Meta Notes

This notebook is intentionally raw.

Failures, bugs, and confusion are recorded because:
- Research is non-linear
- Insight often comes from mistakes
- Future work builds on these details

---

## Closing Reflection

Even simple CNN experiments highlight foundational principles:

- Generalization > memorization
- Controlled ablations are non-negotiable
- Error analysis reveals more than aggregate metrics
- Systems-level understanding compounds over time

This baseline now serves as a stable reference point for future work in
**efficient deep learning systems and large-scale models**.
