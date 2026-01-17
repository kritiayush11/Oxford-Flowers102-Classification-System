# Oxford Flowers102 Classification (Levels 1–3)
Transfer Learning + Augmentation + Attention + Interpretability (Grad-CAM)

## 📌 Project Overview
This repository contains a complete image classification pipeline built on the **Oxford Flowers 102** dataset using **TensorFlow / Keras**.

The project is implemented in three progressive levels:

- ✅ **Level 1:** Baseline transfer learning model (ResNet50)
- ✅ **Level 2:** Intermediate improvements (augmentation, regularization, fine-tuning + ablation study)
- ✅ **Level 3:** Advanced architecture design (custom attention mechanism + interpretability with Grad-CAM)

---

## 🎯 Dataset
- **Dataset Name:** Oxford Flowers 102
- **Source:** TensorFlow Datasets (TFDS)
- **Type:** Multi-class image classification
- **Classes:** 102 flower categories
- **Splits:** Train / Validation / Test

---

## ✅ Level-wise Requirements

### ✅ LEVEL 1: Baseline Model
**Goal:** Build baseline classifier using transfer learning  
**Approach:** ResNet50 (ImageNet pretrained)

**Deliverables Included**
- Dataset loading (TFDS)
- Baseline model training
- Test accuracy metric
- Training curves (accuracy & loss)

---

### ✅ LEVEL 2: Intermediate Techniques
**Goal:** Improve performance with advanced techniques  
**Approach:** Augmentation + regularization + hyperparameter tuning + fine-tuning

**Deliverables Included**
- Augmentation pipeline
- Ablation study:
  - Level 1 (without augmentation)
  - Level 2 (with augmentation)
- Accuracy comparison table
- Analysis document (included in notebook)

---

### ✅ LEVEL 3: Advanced Architecture Design
**Goal:** Design custom/advanced architecture  
**Approach:** ResNet50 + Custom Attention (SE Block) + Grad-CAM interpretability

**Deliverables Included**
- Architecture design document
- Custom model implementation (Attention-based network)
- Per-class evaluation (classification report)
- Confusion matrix
- Interpretability (Grad-CAM)
- Insights & findings

---

## 🏗️ Model Architectures

### 🔹 Level 1 Model
- **Backbone:** ResNet50 (frozen)
- **Head:** GAP → Dense(256) → Dropout → Dense(102 softmax)

### 🔹 Level 2 Model
- Adds:
  - Strong **data augmentation**
  - More dropout regularization
  - Fine-tuning last layers of ResNet50

### 🔹 Level 3 Model (Custom Architecture)
- **ResNet50 + SE Attention Block**
- Fine-tuned training
- Grad-CAM visualization

---

## ⚙️ Tech Stack
- Python 3.x
- TensorFlow / Keras
- TensorFlow Datasets (TFDS)
- Matplotlib
- Pandas
- NumPy
- Scikit-learn (for evaluation metrics)
- (Optional) Seaborn for confusion matrix

---

## 🚀 How to Run

### 1) Install Dependencies
```bash
pip install tensorflow tensorflow-datasets numpy pandas matplotlib scikit-learn seaborn
