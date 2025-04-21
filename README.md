# 🦴 Bone Fracture Detection Using WideResNet

A deep learning solution for automated bone fracture detection using WideResNet, designed for medical imaging classification tasks. This project leverages transfer learning and image preprocessing to achieve high diagnostic accuracy on radiographic images.

## 🧠 Project Overview

Bone fractures are a common concern in medical diagnostics, and radiological interpretation can be time-consuming and error-prone. This project presents a machine learning pipeline using **WideResNet** to automate fracture detection from X-ray images, improving diagnostic efficiency and aiding healthcare professionals.

## 🎯 Objectives

- Build a deep learning model for fracture classification
- Employ **WideResNet** for its robust feature extraction capabilities
- Utilize transfer learning to adapt pre-trained models to X-ray data
- Provide a reproducible, modular, and interpretable notebook-based implementation

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV, NumPy, Pandas
- Matplotlib, Seaborn (for visualization)
- Google Colab / Jupyter Notebook

## 📂 Dataset

- **Source**: Medical X-ray image datasets (e.g., MURA, or other labeled sources)
- **Images**: X-rays labeled as `fractured` or `normal`
- **Format**: `.jpg` or `.png`
- **Structure**:
  ```
  dataset/
  ├── train/
  │   ├── fractured/
  │   └── normal/
  └── test/
      ├── fractured/
      └── normal/
  ```

## 🔄 Preprocessing Pipeline

- Image resizing and grayscale normalization
- Data augmentation (rotation, flipping, zoom) for robustness
- Categorical label encoding
- Train-validation-test splitting

## 🧪 Model Architecture

- **Base Model**: WideResNet (Wide Residual Network)
- **Layers**:
  - Input layer with normalized pixel values
  - WideResNet backbone with frozen layers
  - Dense output layer with sigmoid activation for binary classification
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Evaluation**: Accuracy, Precision, Recall, AUC

## 📈 Results & Evaluation

| Metric      | Value     |
|-------------|-----------|
| Accuracy    | ~92%      |
| Precision   | High      |
| Recall      | High      |
| AUC         | >0.90     |

Model demonstrated robust classification capabilities across diverse test images, maintaining strong performance even in cases with occlusions or low contrast.

## 🔍 Visualizations

- Confusion matrix
- ROC curve and AUC analysis
- Model training and validation accuracy/loss curves
- Sample predictions on unseen images

## 🚀 Future Enhancements

- Integrate Grad-CAM or saliency maps for explainability
- Explore ensemble models with ResNet, DenseNet, and EfficientNet
- Expand to multi-class detection (e.g., hairline fractures, dislocations)
- Deploy as a Flask web app or REST API for clinical use

## 👩‍💻 Contributors

- **Sairam Murari** – sairam.murari@gmail.com  

---

> “Empowering healthcare with AI-driven diagnostic tools – because every second counts in saving lives.”
