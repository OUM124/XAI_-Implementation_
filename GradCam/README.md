# GradCAM Project - README

## 📁 Project Overview
This project implements **GradCAM (Gradient-weighted Class Activation Mapping)** and related explainability techniques from scratch for a **plant classification task** with 4 classes: **Rudo, Baya, Greg, Yuki**.

---

## 🗂️ Project Structure

### 📊 **Datasets**
```
train/      → 800 training images (200 per class)
val/        → 200 validation images (50 per class)
test/       → 80 test images (20 per class)
```

**Image Naming Convention**: `{class_label}_{unique_id}.jpg`
- Class 0 = Rudo
- Class 1 = Baya
- Class 2 = Greg
- Class 3 = Yuki

**Example**: `0_cafefa5f668e62183b8f.jpg` → Rudo plant

---

### 🧠 **Models** (43 MB each)

| File | Description | Test Accuracy |
|------|-------------|---------------|
| `plant_classifier_final.pth` | Final ResNet18 model trained on plant dataset | 75.00% |
| `best_plant_classifier.pth` | Best checkpoint during training | ~75% |

**Model Architecture**: ResNet18 (pretrained on ImageNet, fine-tuned for 4 plant classes)

**Checkpoint Contents**:
- `model_state_dict` - Model weights
- `class_names` - ['Rudo', 'Baya', 'Greg', 'Yuki']
- `train_acc`, `val_acc`, `test_acc` - Performance metrics
- `epoch` - Training epoch number

---

### 📓 **Jupyter Notebooks**

#### 🔧 **Training & Fine-tuning**
- **`FineTune_Resnet.ipynb`** (4.5 MB)
  - Complete pipeline for fine-tuning ResNet18 on plant dataset
  - Includes: data loading, model setup, training loop, validation, checkpoint saving
  - **Use this to**: Retrain or fine-tune the model

#### 🎯 **GradCAM Implementations**

- **`GradCAM_Tutorial.ipynb`** (4.6 MB)
  - Comprehensive GradCAM tutorial with theory and math
  - From-scratch implementation of GradCAM class
  - Includes LaTeX formulas and detailed explanations
  - **Use this to**: Learn how GradCAM works

- **`GradCAMExperiment.ipynb`** (5.4 MB)
  - Experimental GradCAM variations
  - Tests different layers, configurations, and parameters
  - **Use this to**: Experiment with GradCAM settings

#### 🔍 **Advanced Explainability Techniques**

- **`GuidedBackpropagation.ipynb`** (3.6 MB)
  - Implements Guided Backpropagation from scratch
  - Modifies ReLU backward pass to propagate only positive gradients
  - Shows fine-grained pixel-level feature importance
  - **Use this to**: Get detailed gradient visualizations

- **`GuidedGradCAM.ipynb`** (10.2 MB)
  - **Combines** GradCAM + Guided Backpropagation
  - Formula: `Guided Grad-CAM = Guided Backprop ⊙ Upsample(Grad-CAM)`
  - Provides both coarse localization AND fine-grained details
  - **Use this to**: Get the best explainability visualization

---

### 🛠️ **Utility Scripts** (`utils/`)

| File | Purpose |
|------|---------|
| `datasets.py` | Dataset classes for loading plant images |
| | - `ImageDataset`: Loads images with class labels from filename |
| | - `SegmentationDataset`: For satellite imagery (not used here) |
| | - `preprocess_imagenet_image()`: Preprocessing for ImageNet models |
| `visualise.py` | Visualization utilities |
| | - `get_edge()`: Canny edge detection |
| | - `process_attributions()`: Process and visualize gradients |
| | - `add_edge_to_attributions()`: Overlay edges on heatmaps |
| | - `display_imagenet_output()`: Show ImageNet predictions |

---

### 🖼️ **Sample Images**

| File | Description |
|------|-------------|
| `imagenet_classes.txt` | 1000 ImageNet class names |
| `sample_image.jpg` (97 KB) | Generic test image |
| `shark.jpg` (142 KB) | Shark image for ImageNet testing |
| `shark2.jpg` (127 KB) | Another shark image |
| `Sushi.png` (10 MB) | Sushi image for demos |

---

## 📚 Key Concepts Implemented

### **1. GradCAM**
- **Purpose**: Localization - WHERE the model looks
- **Method**: Weighted average of activation maps using gradient importance
- **Output**: Coarse heatmap showing important regions
- **Formula**: `CAM = ReLU(Σ αₖ × Aₖ)` where `αₖ = global_avg_pool(∂y/∂Aₖ)`

### **2. Guided Backpropagation**
- **Purpose**: Fine-grained details - WHAT features matter
- **Method**: Modified backprop that only passes positive gradients through ReLU
- **Output**: High-resolution gradient map
- **Key**: Sets `ReLU.inplace = False` to avoid autograd conflicts

### **3. Guided Grad-CAM**
- **Purpose**: Combine localization + details
- **Method**: Element-wise multiply Guided Backprop with upsampled GradCAM
- **Output**: High-quality visualization with both WHERE and WHAT
- **Formula**: `Guided Grad-CAM = Guided Backprop ⊙ GradCAM↑`

---



## 📊 Model Performance

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | ~95% | ~80% | 75% |
| Classes | 4 (Rudo, Baya, Greg, Yuki) |
| Architecture | ResNet18 (pretrained) |
| Optimizer | Adam (lr=0.001) |
| Loss | CrossEntropyLoss |

---

## 🎯 Usage Recommendations

**For Learning**: Start with `GradCAM_Tutorial.ipynb`
**For Experimentation**: Use `GradCAMExperiment.ipynb`
**For Best Visualizations**: Use `GuidedGradCAM.ipynb`
**For Training**: Use `FineTune_Resnet.ipynb`
**For Production**: Load `plant_classifier_final.pth` and use GradCAM classes

---

## 🔗 Dependencies

```bash
torch
torchvision
numpy<2.0
opencv-python
matplotlib
scikit-image
scikit-learn
seaborn
PIL
scipy
```

---

## 📝 File Summary

**Total Files**: 10 notebooks + 3 utils + 2 models + 5 images + 1080 dataset images
**Total Size**: ~220 MB (including datasets)
**Language**: Python 3.8+
**Framework**: PyTorch

---

