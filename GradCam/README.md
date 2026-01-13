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
| `download.py` | Download utilities (not critical) |

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

## 🚀 Quick Start

### 1️⃣ **Load Trained Model**
```python
import torch
import torch.nn as nn
from torchvision import models

# Create model architecture
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 4)

# Load weights
checkpoint = torch.load('plant_classifier_final.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Classes: {checkpoint['class_names']}")
print(f"Test Accuracy: {checkpoint['test_acc']:.2f}%")
```

### 2️⃣ **Run GradCAM**
```python
from utils.datasets import ImageDataset
import glob

# Load test image
test_paths = glob.glob("./test/*.jpg")
dataset = ImageDataset(test_paths, num_classes=4)
image, target = dataset[0]

# Get prediction
input_tensor = image.unsqueeze(0)
output = model(input_tensor)
pred_class = output.argmax(dim=1).item()

# Apply GradCAM (see notebooks for GradCAM class)
target_layer = model.layer4[-1]
cam = GradCAM(model, target_layer)
heatmap = cam(input_tensor, target_class=pred_class)
```

### 3️⃣ **Generate Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get predictions for all test samples
y_true, y_pred = [], []
for i in range(len(dataset)):
    image, target = dataset[i]
    true_label = target.argmax().item()
    pred_label = model(image.unsqueeze(0)).argmax(dim=1).item()
    y_true.append(true_label)
    y_pred.append(pred_label)

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=plant_names,
            yticklabels=plant_names)
```

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

## 🔍 Common Issues & Solutions

### **Issue 1: RuntimeError with hooks**
```
RuntimeError: cannot register a hook on a tensor that doesn't require gradient
```
**Solution**: Don't use `torch.no_grad()` when computing GradCAM. Gradients are needed!

### **Issue 2: Path parsing on Windows**
```
ValueError: invalid literal for int() with base 10: 'test\\0'
```
**Solution**: Already fixed in `utils/datasets.py` using `os.path.basename()`

### **Issue 3: NumPy 2.0 compatibility**
```
module compiled using NumPy 1.x cannot run in NumPy 2.0
```
**Solution**: Install `numpy<2.0`

### **Issue 4: In-place ReLU conflicts**
```
RuntimeError: Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace
```
**Solution**: Set `module.inplace = False` for all ReLU layers before registering hooks

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

## 🎓 References

1. **Grad-CAM**: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
2. **Guided Backprop**: [Springenberg et al., 2015](https://arxiv.org/abs/1412.6806)
3. **ResNet**: [He et al., 2016](https://arxiv.org/abs/1512.03385)

---

**Last Updated**: January 2026
**Project Type**: Educational / Research - Explainable AI for Plant Classification
