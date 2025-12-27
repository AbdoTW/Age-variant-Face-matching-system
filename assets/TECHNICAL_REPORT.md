# Technical Report: Age-Invariant Face Recognition System

## Table of Contents
1. [Dataset Choice and Rationale](#1-dataset-choice-and-rationale)
2. [Age Prediction Model Architecture](#2-age-prediction-model-architecture)
3. [Face Matching Model Explanation](#3-face-matching-model-explanation)
4. [Loss Function Selection and Rationale](#4-loss-function-selection-and-rationale)
5. [Performance Analysis and Evaluation Metrics](#5-performance-analysis-and-evaluation-metrics)

---

## 1. Dataset Choice and Rationale

### Datasets Used
This project utilizes four major age-invariant face recognition datasets for training and threshold tuning:

1. **MORPH (Craniofacial Longitudinal Morphological Face Database)**
   - Large-scale longitudinal aging dataset
   - Contains multiple age samples per subject
   - Age range: 16-77 years
   - Provides realistic aging progression patterns

2. **CACD (Cross-Age Celebrity Dataset)**
   - Celebrity faces across different ages
   - Real-world unconstrained images
   - Age range: 16-62 years
   - Captures natural aging variations in diverse conditions

3. **AgeDB (Age Database)**
   - Specifically designed for age-invariant face verification
   - Large age gaps between image pairs
   - Contains both in-the-wild and controlled images
   - Ideal for evaluating cross-age matching performance

4. **FG-NET (Face and Gesture Recognition Network Aging Database)**
   - Longitudinal aging database
   - Multiple images per subject across ages
   - Age range: 0-69 years
   - High-quality age progression sequences

### Rationale for Dataset Selection

#### Why These Datasets?
- **Complementary Coverage**: Each dataset provides unique characteristics:
  - MORPH: Controlled, high-quality aging samples
  - CACD: Celebrity faces with diverse poses and expressions
  - AgeDB: Extreme age gaps for robust evaluation
  - FG-NET: Complete aging trajectories

- **Age-Invariance Focus**: All four datasets are specifically designed for studying age variations in face recognition, making them ideal for training age-adaptive thresholds

- **Real-World Applicability**: Combination of controlled (MORPH, FG-NET) and in-the-wild (CACD, AgeDB) images ensures model generalization

- **Benchmark Standards**: These datasets are widely used in academic research, allowing direct comparison with state-of-the-art methods

#### Dataset Preprocessing
Preprocessing pipeline (detailed in `assets/1-morph-cacd-agedb-fgnet-dataset-preprocessing.ipynb`):
1. Face detection and alignment
2. Quality filtering
3. Age label verification
4. Resolution normalization
5. Train/validation/test splitting

Pair generation (detailed in `assets/2-create-pairs-dataset-processing.ipynb`):
1. Positive pairs: Same identity, different ages
2. Negative pairs: Different identities, controlled age distribution
3. Age-gap stratification for threshold optimization

---

## 2. Age Prediction Model Architecture

### Model Selection
**Vision Transformer (ViT)** from [HuggingFace: abhilash88/age-gender-prediction](https://huggingface.co/abhilash88/age-gender-prediction)

### Architecture Overview

![ViT Architecture Diagram](assets/vit_architecture.png)
*Figure 1: Vision Transformer Architecture for Age Prediction*

#### Key Components

1. **Patch Embedding Layer**
   - Input image size: 224×224×3
   - Patch size: 16×16
   - Number of patches: (224/16)² = 196 patches
   - Each patch is linearly embedded to D-dimensional space

2. **Transformer Encoder**
   ```
   Input: Sequence of embedded patches + [CLS] token
   ↓
   Multi-Head Self-Attention (MSA)
   ↓
   Layer Normalization
   ↓
   MLP (Feed-Forward Network)
   ↓
   Layer Normalization
   ↓
   Repeat for N layers
   ```

3. **Classification Head**
   - Extracts [CLS] token representation
   - Fully connected layer for age regression
   - Fully connected layer for gender classification

#### Model Specifications
```python
Model: Vision Transformer (ViT)
Parameters: ~86M
Input Resolution: 224×224
Patch Size: 16×16
Number of Transformer Layers: 12
Hidden Dimension: 768
MLP Dimension: 3072
Attention Heads: 12
Output: Age (regression) + Gender (classification)
```

### Why Vision Transformer for Age Prediction?

#### 1. **Global Context Understanding**
- Unlike CNNs that process local features, ViT captures global facial relationships
- Aging affects multiple facial regions simultaneously (wrinkles, face shape, skin texture)
- Self-attention mechanism allows the model to learn age-relevant correlations across the entire face

#### 2. **Superior Feature Representation**
- Transformer architecture excels at learning hierarchical representations
- Better at capturing subtle aging patterns compared to traditional CNNs
- Pre-trained on large-scale datasets, providing robust initial features

#### 3. **Robustness to Variations**
- Handles pose variations, lighting conditions, and partial occlusions effectively
- Age estimation remains consistent across different image qualities

#### 4. **State-of-the-Art Performance**
- ViT-based models achieve competitive results on age estimation benchmarks
- Lower Mean Absolute Error (MAE) compared to CNN-based approaches

### Integration in the System

```python
from model import predict_age_gender

# Usage in app.py
age_result1 = predict_age_gender(face_crop1)
age_result2 = predict_age_gender(face_crop2)

# Returns: {'age': predicted_age, 'gender': predicted_gender}
```

The predicted ages are used to:
1. **Select Age-Adaptive Thresholds**: Age gap determines which EER-optimized threshold to use
2. **Display Annotations**: Show predicted age on detected faces
3. **Performance Analysis**: Evaluate system accuracy across different age ranges

---

## 3. Face Matching Model Explanation

### Model Selection: InsightFace (buffalo_l)

**InsightFace** is a state-of-the-art face recognition framework that provides highly accurate face detection and recognition capabilities.


### Buffalo_l Model Specifications

```python
Model Name: buffalo_l
Framework: InsightFace
Backend: ONNX Runtime
Detection Model: RetinaFace
Recognition Model: ArcFace (ResNet-based)
Embedding Dimension: 512
Input Size: 112×112
Normalization: Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5]
```

### Architecture Components

#### 1. **Face Detection (RetinaFace)**
```
Input Image
↓
Feature Pyramid Network (FPN)
↓
Multi-scale Feature Extraction
↓
Classification + Bounding Box Regression
↓
5-Point Facial Landmarks Detection
↓
Detected Face + Landmarks
```

**Key Features:**
- Multi-scale detection (handles various face sizes)
- Accurate bounding box localization
- Facial landmark detection for alignment
- High recall rate even on small faces

#### 2. **Face Recognition (ArcFace Backbone)**


```
Detected Face (112×112)
↓
ResNet-100 Backbone
  ├─ Conv Layers (Feature Extraction)
  ├─ Residual Blocks (Deep Feature Learning)
  └─ Global Average Pooling
↓
Embedding Layer (512-dim)
↓
L2 Normalization (Unit Hypersphere)
↓
Normalized Embedding Vector
```


### Why InsightFace (buffalo_l)?

#### 1. **Unified Pipeline**
- Single framework handles both detection and recognition
- Seamless integration between components
- Optimized end-to-end performance

#### 2. **State-of-the-Art Accuracy**
- Buffalo_l is one of InsightFace's most accurate models
- Trained on massive datasets (millions of identities)
- Achieves >99% accuracy on LFW benchmark



#### 3. **Efficient Inference**
- ONNX-optimized models for fast inference
- CPU-compatible (important for Streamlit deployment)
- Low memory footprint


### Face Verification Process

#### Step 1: Face Detection
```python
# Detect single face using InsightFace
faces = app.get(image_bgr)

# Validation
if len(faces) == 0:
    return "No face detected"
if len(faces) > 1:
    return "Multiple faces detected"

face = faces[0]
bbox = face.bbox  # Bounding box [x1, y1, x2, y2]
```

#### Step 2: Face Cropping and Preprocessing
```python
# Add padding for better crop
padding = int(0.2 * max(x2 - x1, y2 - y1))
face_crop = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
```

#### Step 3: Embedding Extraction
```python
# Extract normalized embedding from InsightFace
embedding = face.embedding  # 512-dimensional vector
embedding = embedding / np.linalg.norm(embedding)  # L2 normalization
```

#### Step 4: Similarity Computation
```python
# Compute cosine similarity between embeddings
similarity = np.dot(embedding1, embedding2)
# Range: [-1, 1], where 1 = identical, -1 = opposite
```

#### Step 5: Age-Adaptive Decision
```python
# Select threshold based on age gap
threshold = get_adaptive_threshold(age1, age2)

# Make verification decision
is_same_person = similarity >= threshold
```



### Comparison with Alternative Models (17 version notebook[private])

Our experiments evaluated multiple face recognition models to determine the optimal choice for age-invariant verification:

| Model | Embedding Dim | FG-NET ROC AUC | Training Approach | Age Robustness |
|-------|---------------|----------------|-------------------|----------------|
| **InsightFace (buffalo_l)** | 512 | **~0.9800** | Pretrained (Multi-dataset) | Excellent |
| R100 (Glint360K) | 512 | 0.9444 | Pretrained | Excellent |
| R100 Fine-tuned (AgeDB+MORPH+CACD) | 512 | 0.8503 | Fine-tuned with ArcFace | Good |
| R100 Fine-tuned (AgeDB+MORPH) | 512 | 0.8482 | Fine-tuned with ArcFace | Good |
| FaceNet (Base) | 512 | 0.9031 | Pretrained (VGGFace2) | Good |
| FaceNet Fine-tuned (MORPH+AgeDB) | 512 | 0.7514 | Fine-tuned with Triplet Loss | Moderate |

**Key Observations:**
- **InsightFace (buffalo_l) achieved the highest ROC AUC (0.98)** on FG-NET validation, making it the clear choice for deployment
- Pretrained models consistently outperformed fine-tuned variants, suggesting strong baseline features learned from large-scale training
- Fine-tuning on age-specific datasets (AgeDB, MORPH, CACD) degraded performance, likely due to:
  - Smaller dataset size compared to original training data
  - Risk of overfitting to specific age patterns
  - Disruption of well-learned embedding geometry

**Why buffalo_l was selected for final deployment:**
- Superior accuracy across all age gaps
- Best balance of detection and recognition performance
- Unified pipeline reduces complexity
- Production-ready with minimal setup required

---

## 4. Loss Function Selection and Rationale

### Overview of Experimental Loss Functions

During our experiments, we evaluated two prominent loss functions for age-invariant face recognition:

1. **ArcFace Loss** (Additive Angular Margin Loss) - Used for R100 fine-tuning
2. **Age-Aware Triplet Loss** - Used for FaceNet fine-tuning

---

### 4.1 ArcFace Loss (R100 Experiments)

#### Mathematical Formulation

**Standard Softmax Loss (Baseline):**
```
L_softmax = -log( e^(W_y^T * f) / Σ_j e^(W_j^T * f) )
```

**ArcFace Loss (Our Implementation):**
```
L_ArcFace = -log( e^(s * cos(θ_y + m)) / (e^(s * cos(θ_y + m)) + Σ_{j≠y} e^(s * cos(θ_j))) )
```

Where:
- `θ_y = arccos(W_y^T * f)`: angle between embedding and true class weight
- `m = 0.5`: additive angular margin (in radians, ~28.6°)
- `s = 30`: feature scale (controls gradient magnitude)
- `f`: L2-normalized embedding
- `W_y`: L2-normalized weight vector for true class

![ArcFace Margin Visualization](assets/arcface_margin.png)
*Figure 5: Angular margin in ArcFace forces greater separation between identities*

#### Implementation in R100 Fine-tuning

```python
# Loss criterion
criterion = torch.nn.CrossEntropyLoss()

# Model architecture with ArcFace
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        self.in_features = in_features  # 512
        self.out_features = out_features  # Number of identities
        self.s = s  # Scale factor
        self.m = m  # Angular margin
        
    def forward(self, embedding, label):
        # Normalize features and weights
        embedding = F.normalize(embedding)
        W = F.normalize(self.weight)
        
        # Compute cosine similarity
        cosine = F.linear(embedding, W)
        
        # Compute angular margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # Add margin to target class
        target_logits = torch.cos(theta + self.m)
        
        # Apply margin only to target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * target_logits) + ((1.0 - one_hot) * cosine)
        output *= self.s  # Scale
        
        return output

# Training pipeline
embedding = model(img)  # R100 backbone
output = model.module.metric_fc(embedding, subject_id)  # ArcMarginProduct
loss = criterion(output, subject_id)  # CrossEntropyLoss
```

#### Why ArcFace for R100?

**1. Identity-Discriminative Embeddings**
- Enforces angular margin between different identities
- Same identity → smaller angular distance
- Different identities → larger angular distance
- Creates well-separated clusters in embedding space

**2. Alignment with Cosine Similarity**
- ArcFace explicitly optimizes angular margins
- Cosine similarity directly measures angular distance
- Perfect match for verification tasks using cosine thresholds

**3. Proven for Age-Invariant Recognition**
- Standard loss function for age-invariant benchmarks
- Used in state-of-the-art systems (AgeDB, MORPH, CACD)
- Robust to appearance changes over time

**4. Optimal Hyperparameters**
- `s = 30.0`: Standard scale factor for face recognition
- `m = 0.5`: Empirically validated angular margin (28.6°)

---

### 4.2 Age-Aware Triplet Loss (FaceNet Experiments)

#### Mathematical Formulation

**Base Triplet Loss:**
```
L_triplet = max(0, ||f_a - f_p||² - ||f_a - f_n||² + margin)
```

**Age-Aware Triplet Loss (Our Enhancement):**
```
L_age_aware = L_triplet + α * (age_gap / 60) * ||f_a - f_p||²
```

Where:
- `f_a, f_p, f_n`: Anchor, positive, negative embeddings (L2-normalized)
- `margin`: Distance margin between positive and negative pairs
- `α = 0.1`: Age penalty weight
- `age_gap`: Absolute age difference between anchor and positive
- `60`: Normalization constant (max expected age gap)

<!-- #### Implementation in FaceNet Fine-tuning

```python
class AgeAwareTripletLoss(nn.Module):
    def __init__(self, margin=0.5, age_weight=0.1):
        super(AgeAwareTripletLoss, self).__init__()
        self.margin = margin
        self.age_weight = age_weight
    
    def forward(self, anchor, positive, negative, 
                anchor_age=None, pos_age=None, neg_age=None):
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        # Base triplet loss
        triplet_loss = F.relu(pos_dist - neg_dist + self.margin)
        
        # Add age penalty
        if anchor_age is not None and pos_age is not None:
            age_gap = torch.abs(anchor_age - pos_age).float()
            normalized_age_gap = torch.clamp(age_gap / 60.0, 0, 1)
            age_penalty = self.age_weight * normalized_age_gap * pos_dist
            triplet_loss = triplet_loss + age_penalty
        
        return triplet_loss.mean(), pos_dist.mean(), neg_dist.mean()
```

#### Dynamic Margin Scheduling

```python
def get_margin(epoch):
    """Progressively increase margin for harder training"""
    if epoch < 10:
        return 0.3
    elif epoch < 20:
        return 0.5
    elif epoch < 35:
        return 0.7
    else:
        return 1.0
``` -->

#### Triplet Mining Strategy

<!-- **Age-Aware Positive Selection:**
```python
# Prefer larger age gaps within same identity
pos_ages_batch = ages[pos_indices]
age_gaps = torch.abs(pos_ages_batch - age)

# 70% probability: select from top 50% age gaps
if torch.rand(1).item() < 0.7:
    top_gap_threshold = torch.quantile(age_gaps.float(), 0.5)
    large_gap_indices = pos_indices[age_gaps >= top_gap_threshold]
    if len(large_gap_indices) > 0:
        pos_idx = random.choice(large_gap_indices)
```

**Hard Negative Mining:**
```python
# Gradually increase hard negative ratio
hard_ratio = min(0.5 + (epoch * 0.01), 0.9)

if torch.rand(1).item() < hard_ratio:
    # Select hardest negative (closest to anchor)
    neg_idx = neg_indices[torch.argmin(neg_dists)]
else:
    # Select semi-hard negative (pos_dist < neg_dist < pos_dist + margin)
    semi_hard_mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + margin)
    semi_hard_indices = neg_indices[semi_hard_mask]
    if len(semi_hard_indices) > 0:
        neg_idx = random.choice(semi_hard_indices)
``` -->

#### Why Age-Aware Triplet Loss for FaceNet?

**1. Explicit Age-Gap Modeling**
- Age penalty term directly addresses age variation
- Encourages model to maintain small distances despite large age gaps
- More intuitive for age-invariant tasks than standard triplet loss

**2. Flexible Training**
- No need for fixed class labels (unlike ArcFace)
- Works with online triplet mining
- Adapts to dataset characteristics during training

**3. Hard Example Mining**
- Dynamic selection of challenging positive pairs (large age gaps)
- Progressive hard negative mining
- Accelerates convergence on difficult cases

---

### 4.3 Loss Function Comparison

| Aspect | ArcFace Loss | Age-Aware Triplet Loss |
|--------|--------------|------------------------|
| **Type** | Classification-based | Metric learning |
| **Margin Type** | Angular (additive) | Euclidean (distance) |
| **Training Complexity** | Medium | High (requires triplet mining) |
| **Age Modeling** | Implicit (through margin) | Explicit (age penalty term) |
| **Stability** | High | Medium (sensitive to mining) |
| **Best Use Case** | Large-scale identity classification | Online metric learning |
| **Our Results (FG-NET)** | R100 Base: 0.9444 → Fine-tuned: 0.8503 | FaceNet Base: 0.9031 → Fine-tuned: 0.7514 |

---

### 4.4 Why Fine-Tuning Degraded Performance

#### Observation from Experiments

**R100 (ArcFace) Results:**
```
Base Model (Glint360K):          ROC AUC = 0.9444
After Epoch 1 (AgeDB):           ROC AUC = 0.7880
After Epoch 2 (AgeDB):           ROC AUC = 0.7864
After Epoch 3 (AgeDB):           ROC AUC = 0.7879
After Epoch 4 (AgeDB+MORPH+CACD): ROC AUC = 0.8503
```

**FaceNet (Triplet) Results:**
```
Base Model (VGGFace2):           ROC AUC = 0.9031
After Epoch 2 (MORPH+AgeDB):     ROC AUC = 0.7514
After Epoch 4 (MORPH+AgeDB):     ROC AUC = 0.7031
```

#### Root Causes

**1. Strong Pretrained Features**
- Base models trained on massive datasets (Glint360K: 17M images, VGGFace2: 3.3M images)
- Already learned robust identity-discriminative representations
- Fine-tuning datasets (MORPH+AgeDB+CACD: ~250K pairs) much smaller

**2. Distribution Shift**
- Pretrained: Identity classification on diverse faces
- Fine-tuning: Pair-based verification on age-specific datasets

**3. Overfitting Risk**
- Age-specific datasets smaller than pretraining data
- Model may overfit to specific aging patterns
- Loses generalization to unseen age variations


#### Expected Behavior in Metric Learning

This performance degradation is **normal and documented** in transfer learning literature:
- Early fine-tuning often decreases validation metrics
- Network needs time to adapt without destroying pretrained structure
- Small fine-tuning datasets rarely improve upon large-scale pretraining

---

### 4.5 Final System Choice: InsightFace (buffalo_l)

Given the experimental results, we chose **InsightFace (buffalo_l)** for the deployed system because:

**1. Superior Baseline Performance**
- ROC AUC: 0.98 on FG-NET (highest among all models)
- No fine-tuning required
- Immediately production-ready

**2. Pretrained on Massive Scale**
- Trained on millions of identities
- Robust to age, pose, expression variations
- Generalizes well to unseen data

**3. Unified Detection + Recognition**
- RetinaFace detection + ArcFace recognition in single framework
- Optimized end-to-end pipeline
- Fewer failure points

**4. Practical Advantages**
- No training infrastructure needed
- Lower deployment complexity
- Consistent performance across datasets

---

## 5. Performance Analysis and Evaluation Metrics

### 5.1 Evaluation Protocol

All models were evaluated on the **FG-NET dataset** to ensure fair comparison:

**FG-NET Test Set:**
- 82 unique identities
- 1,002 total images
- Age range: 0-69 years
- Genuine pairs: 5,808 (same identity, different ages)
- Impostor pairs: 495,693 (different identities)

**Evaluation Metrics:**
1. **ROC AUC** (Receiver Operating Characteristic - Area Under Curve)
2. **EER** (Equal Error Rate)
3. **TPR @ FPR** (True Positive Rate at fixed False Positive Rates)
   - TPR @ FPR = 0.01% (high security)
   - TPR @ FPR = 0.1% (balanced)
   - TPR @ FPR = 1% (high recall)
4. **Verification Accuracy** at optimal threshold
5. **Age-Gap Specific Performance** (0-5, 5-10, 10-20, 20-30, 30+ years)

---

### 5.2 Model Comparison Results

#### Overall Performance Summary

| Model | ROC AUC | EER (%) | Accuracy (%) | TPR@FPR=0.01% | TPR@FPR=0.1% | TPR@FPR=1% | Status |
|-------|---------|---------|--------------|---------------|--------------|------------|--------|
| **InsightFace (buffalo_l)** | **0.9835** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | ✅ Deployed |
| R100 (Glint360K Base) | 0.9444 | N/A | 87.97 | 28.34 | 43.10 | 67.67 | Baseline |
| R100 Fine-tuned (AgeDB+MORPH+CACD) | 0.8523 | N/A | 78.74 | 15.69 | 24.83 | 45.04 | Best Fine-tuned |
| R100 Fine-tuned (AgeDB+MORPH) | 0.8482 | N/A | 78.30 | 15.08 | 23.24 | 44.97 | Intermediate |
| R100 Fine-tuned (AgeDB only) | 0.7880 | N/A | 73.36 | 7.15 | 14.45 | 32.92 | Early Fine-tune |
| FaceNet (VGGFace2 Base) | 0.9031 | 17.65 | 83.69 | 9.88 | 19.99 | 41.86 | Baseline |
| FaceNet Fine-tuned (MORPH+AgeDB, Epoch 2) | 0.7514 | 31.32 | 70.35 | 0.12 | 2.26 | 11.79 | Fine-tuned |
| FaceNet Fine-tuned (MORPH+AgeDB, Epoch 4) | 0.7031 | 35.55 | 68.06 | 0.02 | 1.46 | 11.11 | Fine-tuned |

---

### 5.3 Age-Gap Performance Analysis

#### InsightFace (buffalo_l) - Best Overall

```
Performance by Age Gap on FG-NET:
- 0-5 years:   ROC AUC = 0.9963  (Excellent)
- 5-10 years:  ROC AUC = 0.9954  (Excellent)
- 10-20 years: ROC AUC = 0.9846  (Excellent)
- 20-30 years: ROC AUC = 0.9714  (Excellent)
- 30+ years:   ROC AUC = 0.9360  (Excellent)
```

#### R100 Base Model (Glint360K)

```
Performance by Age Gap on FG-NET:
Optimal Threshold: 0.1998

Overall: Accuracy = 87.97%
```

#### FaceNet Base Model (VGGFace2)

```
Performance by Age Gap on FG-NET:
Optimal Threshold: 0.3350

Age Gap 0-5 years:
  Pairs: 3,041 (Genuine: 1,655, Impostor: 1,386)
  ROC AUC: 0.9576
  Accuracy: 89.02%

Age Gap 5-10 years:
  Pairs: 2,822 (Genuine: 1,590, Impostor: 1,232)
  ROC AUC: 0.9333
  Accuracy: 85.44%

Age Gap 10-15 years:
  Pairs: 2,118 (Genuine: 1,082, Impostor: 1,036)
  ROC AUC: 0.8976
  Accuracy: 82.72%

Age Gap 15-20 years:
  Pairs: 1,319 (Genuine: 618, Impostor: 701)
  ROC AUC: 0.8642
  Accuracy: 80.14%

Age Gap 20+ years:
  Pairs: 2,316 (Genuine: 863, Impostor: 1,453)
  ROC AUC: 0.8775
  Accuracy: 82.60%
```

**Key Observation:** FaceNet shows gradual performance degradation as age gap increases, typical of age-invariant systems.

#### FaceNet Fine-tuned (MORPH+AgeDB, Epoch 2)

```
Performance by Age Gap on FG-NET:
Optimal Threshold: Not specified

Age Gap 0-5 years:
  Pairs: 4,015
  Accuracy: 70.04%
  Genuine Accuracy: 85.20%
  Impostor Accuracy: 59.41%

Age Gap 5-10 years:
  Pairs: 3,718
  Accuracy: 70.47%
  Genuine Accuracy: 70.06%
  Impostor Accuracy: 70.77%

Age Gap 10-20 years:
  Pairs: 4,676
  Accuracy: 69.14%
  Genuine Accuracy: 50.41%
  Impostor Accuracy: 79.84%

Age Gap 20-100 years:
  Pairs: 3,399
  Accuracy: 72.26%
  Genuine Accuracy: 40.67%
  Impostor Accuracy: 83.00%
```

---

### 5.4 Age-Adaptive Threshold Optimization

To address performance degradation at large age gaps, we developed **age-adaptive thresholds** optimized for InsightFace (buffalo_l).

#### Optimization Methodology

**Dataset:** FG-NET validation set
- 82 unique identities
- 1,002 images
- 5,808 genuine pairs (same person, different ages)
- 495,693 impostor pairs (different people)

**Target Metric:** False Accept Rate (FAR) = 5%

**Process:**
1. Extract InsightFace embeddings for all images
2. Compute cosine similarities for all pairs (501,501 valid comparisons)
3. Group pairs by age gap bins: 0-5, 5-10, 10-20, 20-30, 30+ years
4. For each bin, find threshold where exactly 5% of impostor pairs are accepted
5. Validate thresholds ensure age-invariant performance

#### Optimized Thresholds (Target FAR = 5%)

```python
age_adaptive_thresholds = {
    '0-5 years':   0.3083,  # Small age gap → high threshold
    '5-10 years':  0.2565,  # Medium age gap → moderate threshold
    '10-20 years': 0.2130,  # Large age gap → lower threshold
    '20-30 years': 0.1394,  # Very large age gap → low threshold
    '30+ years':   0.1381   # Extreme age gap → lowest threshold
}
```

#### Rationale Behind Decreasing Thresholds

As age gap increases:
1. **Facial appearance changes more dramatically**
   - Skin texture (wrinkles, age spots)
   - Facial structure (bone density, muscle tone)
   - Overall proportions shift

2. **Embedding similarity naturally decreases**
   - Same person at ages 20 and 70 has lower cosine similarity
   - Than same person at ages 20 and 22

3. **Lower threshold compensates for appearance drift**
   - Maintains consistent verification performance
   - Prevents false rejections for legitimate same-person pairs



### 5.5 Training Repository Reference

All fine-tuning experiments were conducted using the **AQUAFace** framework:

**Repository:** [https://github.com/sadiqebrahim/AQUAFace](https://github.com/sadiqebrahim/AQUAFace)

**Key Features:**
- ArcFace loss implementation for R100
- Age-aware triplet loss for FaceNet
- Validation on FG-NET, AgeDB, MORPH, CACD
- ROC curve generation and threshold optimization
- Comprehensive logging and visualization

**Training Notebooks:**
- `1-morph-cacd-agedb-fgnet-dataset-preprocessing.ipynb`: Dataset preparation
- `2-create-pairs-dataset-processing.ipynb`: Pair generation for verification
- `3-AQUAFace-Training.ipynb`: Model training and evaluation

---

### 5.6 Key Findings

#### ✅ What Worked

1. **Pretrained Models Outperformed Fine-tuned Variants**
   - InsightFace (buffalo_l): ROC AUC 0.98
   - R100 Base: ROC AUC 0.9444
   - Large-scale pretraining provides robust age-invariant features

2. **Age-Adaptive Thresholds Crucial**
   - Fixed thresholds fail at large age gaps
   - Dynamic adjustment maintains consistent FAR
   - Simple yet effective approach

3. **ArcFace Loss Superior for Face Recognition**
   - Better than triplet loss for identity discrimination
   - Stable training on large-scale datasets
   - Aligns with cosine similarity verification
---
### 5.7 Ablation Studies

#### Impact of Training Dataset Size

| Training Dataset | Pairs | Identities | R100 AUC | FaceNet AUC |
|------------------|-------|------------|----------|-------------|
| AgeDB only | 44,763 | 567 | 0.7880 | N/A |
| AgeDB + MORPH | 158,126 | 2,770 | 0.8482 | 0.7514 |
| AgeDB + MORPH + CACD | 318,126 | 3,791 | **0.8523** | N/A |

**Observation:** Larger datasets improved R100 performance but still below base model (0.9444).

#### Impact of Loss Function

| Model | Loss Function | Base AUC | Fine-tuned AUC | Δ AUC |
|-------|---------------|----------|----------------|-------|
| R100 | ArcFace (m=0.5, s=30) | 0.9444 | 0.8523 | -0.0921 |
| FaceNet | Age-Aware Triplet (m=0.3→1.0) | 0.9031 | 0.7514 | -0.1517 |

**Observation:** ArcFace more stable than triplet loss during fine-tuning.

#### Impact of Age Prediction Accuracy

Age prediction accuracy directly affects threshold selection:

- **Accurate age prediction** → Correct age bin → Optimal threshold
- **Inaccurate age prediction** → Wrong age bin → Suboptimal threshold

**ViT Age Prediction Performance** (on FG-NET):
- Mean Absolute Error (MAE): ~3-5 years
- Impact on threshold selection: Minimal for adjacent bins
- Critical for extreme age gaps (0-5 vs 30+)

---

## 6. Conclusion

This technical report presented a comprehensive analysis of the **Age-Invariant Face Recognition System**, covering:

### Key Contributions

1. **Systematic Model Comparison**
   - Evaluated 3 architectures (InsightFace, R100, FaceNet)
   - Tested 2 loss functions (ArcFace, Age-Aware Triplet)
   - Demonstrated pretrained models outperform fine-tuned variants

2. **Age-Adaptive Threshold Optimization**
   - Developed age-gap-specific thresholds (0-5 to 30+ years)
   - Maintained consistent 5% FAR across all age bins
   - Simple yet effective approach for age-invariant verification

3. **Production-Ready Deployment**
   - Selected InsightFace (buffalo_l) for superior performance (AUC 0.98)
   - Integrated Vision Transformer for age prediction
   - Real-time CPU inference with unified detection + recognition

### Reproducibility

All experiments, code, and notebooks are available:
- **Training Framework**: [AQUAFace Repository](https://github.com/sadiqebrahim/AQUAFace)
- **Deployment Code**: Streamlit application in `streamlit_app/`
- **Preprocessing Notebooks**: `assets/*.ipynb`

