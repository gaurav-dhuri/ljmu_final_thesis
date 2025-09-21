# Brain Tumor Classification using Transformer Architectures

This repository contains the implementation and experiments conducted as part of a **Master’s Thesis in Artificial Intelligence and Machine Learning (AIML)** at **Liverpool John Moores University (LJMU)**.  
The research investigates the effectiveness of **transformer-based architectures** (ViT, Swin Transformer, MaxViT) in classifying brain tumors from MRI images.

---

## 📖 Project Overview

Brain tumors pose serious diagnostic challenges due to their high mortality and variability in MRI scans. Manual interpretation is time-consuming, subjective, and error-prone.  
This work leverages **deep learning**, specifically **transformer architectures**, to automate brain tumor classification.

The study compares:

- **Base Architectures**: Models trained from scratch (ViT, Swin, MaxViT).  
- **Pretrained Architectures**: Fine-tuned ImageNet models (via Hugging Face).  

The objective is to evaluate their performance on MRI datasets across **accuracy, precision, recall, F1-score, and test accuracy**.

---

## 📊 Dataset

- Source: Publicly available **Brain Tumor MRI Dataset (Kaggle, Masoud Nickparvar)**  
- Classes: **Glioma, Meningioma, Pituitary, No Tumor**  
- Total Images: ~7,032  
- Balanced to **8,000 samples** (2,000 per class) using random sampling.  
- Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data

**Preprocessing and EDA steps included**:
- Corrupt image detection & filtering  
- Class balancing  
- Aspect ratio analysis  
- Channel & image quality analysis  
- Image resizing to **224 × 224**  

---

## 🏗️ Methodology

### Workflow
1. Dataset Collection & Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Model Design (Base and Pretrained)  
4. Training and Evaluation  
5. Comparative Analysis  

### Models
- **Vision Transformer (ViT)** – Splits images into patches, applies self-attention to model global dependencies.  
- **Swin Transformer** – Hierarchical design with shifted windows to capture local and global features.  
- **MaxViT** – Combines MBConv blocks with local and grid attention for efficient multi-scale representation.  

### Training Strategy
- **Base models**: Trained from scratch with heavy augmentation, ~50–80 epochs.  
- **Pretrained models**: Fine-tuned Hugging Face models with ImageNet weights, converged faster (5–10 epochs).  
- **Optimizers**: Adam / AdamW  
- **Learning Rate**: 1e-4 with scheduling (ReduceLROnPlateau, CosineAnnealingLR)  
- **Callbacks**: Early stopping, checkpointing, learning rate reduction  

---

## ⚙️ Implementation

### Frameworks & Tools
- **TensorFlow/Keras** → Base Architectures  
- **PyTorch + Hugging Face Transformers** → Pretrained Architectures  
- **scikit-learn** → Evaluation metrics  
- **Matplotlib/Seaborn** → Visualization  

### Training Logs
- **Base ViT**: ~82% accuracy  
- **Base Swin**: ~79% accuracy  
- **Base MaxViT**: ~73% accuracy  
- **Pretrained ViT**: ~98% accuracy (best performer)  
- **Pretrained Swin**: ~97% accuracy  
- **Pretrained MaxViT**: ~95% accuracy  

---

## 📈 Evaluation Metrics

- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  
- **Test Accuracy**  

⚠️ Accuracy alone is not sufficient in medical imaging. Precision, recall, and F1-score ensure better reliability, especially under class imbalance.  

---

## 📌 Key Findings

- Pretrained models significantly **outperformed base models**.  
- **ViT (pretrained)** achieved the highest accuracy (~98%).  
- Swin and MaxViT also showed strong results, validating the role of **transformers in medical imaging**.  
- Dataset size, class imbalance, and GPU compute cost were the main **limitations**.  

---

## 🏛️ Academic Context

- **University**: Liverpool John Moores University (LJMU)  
- **Program**: MSc in Artificial Intelligence and Machine Learning  
- **Thesis Title**: *COMPARATIVE STUDY OF DEEP LEARNING ARCHITECTURES FOR ACCURATE DIAGNOSIS OF BRAIN TUMOR DETECTION USING MEDICAL IMAGING.*  
- **Author**: Gaurav Dhuri  
- **Supervisor**: Vijay Prakash
- **Year**: 2025  

---

## 📚 References
- Dosovitskiy et al., 2020 (ViT)  
- Liu et al., 2021 (Swin)  
- Tu et al., 2022 (MaxViT)  
- Masoud Nickparvar, Kaggle Dataset (Brain MRI)  

---

✨ *This project bridges the gap in comparative studies of transformer-based models for brain tumor MRI classification and demonstrates the potential of transfer learning in medical imaging.*  
