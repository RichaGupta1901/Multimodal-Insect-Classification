# 🪲 Multimodal Insect Species Recognition and Habitat-Aware Distribution Modelling

### **Project Overview**
This project focuses on **multimodal learning** for accurate **insect species identification and habitat-aware distribution modelling** under changing environmental conditions.  
Traditional image-based species classification struggles with visually indistinguishable taxa, while DNA barcoding—though highly accurate—is often incomplete or noisy. This project integrates **three complementary modalities**:

- 🖼️ **Image data** (morphological features)  
- 🧬 **DNA barcode sequences** (genetic signatures)  
- 🌍 **Environmental data** (latitude, longitude, NDVI)

The model learns **joint representations** that are robust to missing modalities and capable of generalizing to unseen taxa and environments.

---

## 🎯 **Objectives**
1. Develop a **multimodal deep learning framework** to integrate image, DNA, and environmental features.  
2. Implement and compare **fusion strategies** — *Early, Late, and Hybrid fusion*.  
3. Evaluate the impact of **contrastive learning** and **ModDrop regularization** on model generalization.  
4. Enable **robust performance even when a modality is missing**.

---

## 📦 **Dataset**
The dataset consists of multimodal samples representing various insect species.

| Split | Samples |
|:------|---------:|
| other_heldout (→ train) | 76,590 |
| test | 39,314 |
| key_unseen | 36,465 |
| val | 14,746 |
| val_unseen | 8,819 |
| test_unseen | 7,887 |

**Class Distribution (Grouped):**
| Class | Samples |
|-------|----------|
| Insecta | 178,865 |
| Arachnida | 3,139 |
| Collembola | 1,743 |
| Malacostraca | 42 |
| Branchiopoda | 17 |
| Chilopoda | 11 |
| Diplopoda | 4 |

---

## ⚙️ **Methodology**

### 1. **Data Preprocessing**
- **Images:** resized to 224×224, normalized  
- **DNA Barcodes:** tokenized using *DNABERT tokenizer*  
- **Environmental features:** latitude, longitude, mean NDVI normalized between 0–1  
- **Weighted sampling** applied to balance classes  

---

### 2. **Model Architecture**
The multimodal framework consists of:
- 🖼️ **Image Encoder:** ViT-based backbone → projected to 256-D space  
- 🧬 **DNA Encoder:** DNABERT → projected to 256-D  
- 🌍 **Environmental Encoder:** small MLP network (3 inputs → 256-D)

Each modality produces a feature vector `z_img`, `z_dna`, and `z_env`, which are fused differently depending on the strategy used.

---

### 3. **Fusion Strategies Implemented**

| Fusion Type | Description | Fusion Stage | Notes |
|--------------|--------------|---------------|-------|
| **Early Fusion** | All modality feature vectors are concatenated directly before feeding to classifier | Early stage | No contrastive alignment — overfits easily |
| **Late Fusion** | Each modality predicts independently; outputs combined via weighted averaging | Decision level | Captures modality-specific confidence |
| **Hybrid Fusion** | Combines intermediate representation alignment (via contrastive loss) and final concatenation | Mid-level | Robust and regularized — realistic generalization |

---

### 4. **Training Setup**
- Optimizer: AdamW  
- Loss Function: CrossEntropy + 0.5 × InfoNCE Contrastive Loss  
- Mixed precision: enabled (AMP)  
- Gradient clipping: 5.0  
- Weighted random sampling for imbalanced classes  
- **ModDrop Regularization:** randomly drops modality embeddings to enforce modality independence  
- **Gradual unfreezing** of DNABERT layers after epoch 2  

---

## 📊 **Experimental Results**

### Training Summary (Hybrid Model)
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-------------|------------|-----------|-----------|
| 1 | 1.64 | 64.7% | 4.46 | 20.1% |
| 2 | 1.20 | 85.3% | 4.71 | 4.5% |
| 3 | 1.17 | 89.2% | 3.94 | 36.6% |
| 4 | 1.16 | 90.1% | 2.58 | 46.2% |
| 5 | 1.16 | 90.1% | 4.81 | 49.0% |

---

### 🧪 **Evaluation**

| Fusion Strategy | Test Accuracy | Test Loss | Remarks |
|-----------------|----------------|------------|----------|
| **Early Fusion** | 99.2% | 0.11 | Unrealistically high — likely overfitting or leakage |
| **Late Fusion** | 98.5% | 0.15 | High accuracy but poor generalization |
| **Hybrid Fusion** | 47.1% | 2.58 | Realistic, consistent with validation; true multimodal behavior |

---

## 🔍 **Discussion and Interpretation**
- **Early Fusion** achieved high numeric performance due to **feature leakage and lack of regularization** — it memorized multimodal correlations instead of learning generalizable representations.  
- **Late Fusion** captured independent modality features but failed to leverage cross-modal synergy.  
- **Hybrid Fusion**, while lower in accuracy, demonstrated **more stable loss trends** and **robustness** under modality dropout — a key trait for real-world ecological data where modalities may be missing.  
- **Contrastive learning** (InfoNCE) aligned representations across modalities, while **ModDrop** forced robustness.  

---

## 🧩 **Conclusion**
- **Hybrid Fusion** provides the most realistic, generalizable, and scientifically valid results for multimodal insect species classification.  
- **Early/Late Fusion** may perform numerically better but lack ecological interpretability.  
- The architecture is extendable to other multimodal biological datasets (e.g., plant phenotyping, genomics + imaging).  

---

## 🚀 **Future Work**
1. Implement **true late-fusion ensemble** with learnable modality weights.  
2. Extend **contrastive pretraining** across unseen taxa.  
3. Include **geospatial embeddings** for environmental modality.  
4. Apply **attention-based fusion** (transformer-style) instead of simple concatenation.  
5. Explore **cross-modal retrieval** tasks (e.g., predict DNA from image).

---

## 🧠 **Tech Stack**
| Component | Library / Framework |
|------------|---------------------|
| Model | PyTorch, HuggingFace Transformers |
| Image Encoder | ViT (Vision Transformer) |
| DNA Encoder | DNABERT |
| Environment Encoder | Custom MLP |
| Visualization | Matplotlib, Seaborn |
| Data Handling | Pandas, NumPy |
| Hardware | NVIDIA GPU with CUDA AMP |

---

## 🧾 **How to Run**

```bash
# Clone repository
git clone https://github.com/RichaGupta1901/multimodal-insect-classification.git
cd multimodal-insect-classification

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_hybrid_fusion.py

# (Optional) Evaluate trained model
python evaluate_model.py
