# Automated Product Defect Detection — Casting Dataset

A computer vision system for automatically classifying manufactured casting images as **Normal** or **Defective** using **MobileNetV2 transfer learning**.

---

## 1. Objective

The goal of this project is to automate visual inspection of manufactured casting components.

Given a casting image, the model predicts one of two classes:

* `def_front` — Defective
* `ok_front` — Normal

This is a **binary image classification** problem.

---

## 2. Project Workflow

```text
Casting Dataset
      ↓
Train / Test Split
      ↓
ImageFolder
      ↓
Image Preprocessing
      ↓
Resize to 224 × 224
      ↓
Data Augmentation
(Training Only)
      ↓
Tensor Conversion
      ↓
ImageNet Normalization
      ↓
Pretrained MobileNetV2
      ↓
Frozen Feature Extractor
      ↓
Trainable Classifier
      ↓
Defective / Normal Prediction
      ↓
Model Evaluation
      ↓
Accuracy | Precision | Recall | F1
      ↓
Confusion Matrix
      ↓
Misclassified Image Analysis
```

---

## 3. Dataset

The dataset contains casting images divided into two classes:

```text
def_front → Defective
ok_front  → Normal
```

The dataset is organized into predefined training and testing sets.

### Dataset Structure

```text
casting_data/
├── train/
│   ├── def_front/
│   └── ok_front/
│
└── test/
    ├── def_front/
    └── ok_front/
```

Here:

* `train` and `test` are **dataset splits**
* `def_front` and `ok_front` are **classes**

PyTorch `ImageFolder` is used to load the images and automatically assign labels based on the class folders.

Expected class output:

```text
Classes found: ['def_front', 'ok_front']
```

---

## 4. Image Preprocessing

All images are resized to:

```text
224 × 224
```

### Training Augmentation

Augmentation is applied only to training images:

* Random Horizontal Flip
* Random Vertical Flip
* Random Rotation ±15°
* Brightness Adjustment
* Contrast Adjustment

Validation/test images are not augmented.

### Normalization

Since MobileNetV2 uses ImageNet pretrained weights, ImageNet normalization is applied:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

## 5. Model

### MobileNetV2

The project uses **MobileNetV2 pretrained on ImageNet**.

```text
MobileNetV2
│
├── Feature Extractor
│      └── Frozen
│
└── Classifier
       └── Trainable
```

The pretrained feature extractor is frozen:

```python
for param in model.features.parameters():
    param.requires_grad = False
```

The original ImageNet classifier is replaced with a binary classifier:

```text
1280 → 2
```

The two outputs correspond to:

```text
def_front
ok_front
```

---

## 6. Why MobileNetV2?

MobileNetV2 was selected because it provides a good balance between:

* Classification performance
* Computational efficiency
* Training speed
* Model size
* Deployment suitability

It is particularly useful when a lightweight CNN is preferred over a larger architecture.

---

## 7. Transfer Learning

Instead of training the complete CNN from scratch, the project uses ImageNet pretrained weights.

The pretrained network already contains useful visual features such as:

```text
Edges
Textures
Shapes
Patterns
```

These features are reused for casting-image classification.

Only the final classification layer is trained for the new dataset.

---

## 8. Model Parameters

The implemented model contains:

```text
Total Parameters:      2,226,434
Trainable Parameters:     2,562
```

The trainable parameters belong to:

```text
Linear(1280, 2)
```

Calculation:

```text
Weights = 1280 × 2 = 2560
Bias    = 2

Total   = 2562
```

Thus, the majority of the MobileNetV2 network remains frozen while the classifier learns the casting-specific classification task.

---

## 9. Training

### Configuration

| Parameter          |            Value |
| ------------------ | ---------------: |
| Model              |      MobileNetV2 |
| Pretrained Weights |         ImageNet |
| Input Size         |        224 × 224 |
| Batch Size         |               32 |
| Epochs             |               10 |
| Learning Rate      |           0.0001 |
| Optimizer          |             Adam |
| Loss Function      | CrossEntropyLoss |
| Feature Extractor  |           Frozen |
| Classifier         |        Trainable |
| Random Seed        |               42 |

### Training Process

```text
Input Image
    ↓
Preprocessing
    ↓
MobileNetV2 Features
    ↓
Classification Layer
    ↓
Prediction
    ↓
CrossEntropyLoss
    ↓
Backpropagation
    ↓
Update Classifier
```

Only the classifier parameters are updated during training.

---

## 10. Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Misclassified Images

For defect detection, `def_front` is treated as the **positive class**.

### Accuracy

Percentage of correctly classified images.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Of the images predicted as defective, how many were actually defective?

```text
Precision = TP / (TP + FP)
```

### Recall

Of all actual defective images, how many were detected?

```text
Recall = TP / (TP + FN)
```

### F1-Score

Harmonic mean of precision and recall.

```text
F1 = 2 × Precision × Recall
     -----------------------
       Precision + Recall
```

Recall is especially important in defect detection because a **false negative can allow a defective product to be classified as normal**.

---

## 11. Confusion Matrix

The confusion matrix summarizes the classification results:

```text
                     Predicted
                  Defective  Normal
Actual
Defective            TP        FN
Normal               FP        TN
```

Where:

* **TP** — Defective correctly detected
* **TN** — Normal correctly identified
* **FP** — Normal incorrectly classified as defective
* **FN** — Defective incorrectly classified as normal

The confusion matrix is saved as:

```text
confusion_matrix.png
```

---

## 12. Error Analysis

Misclassified images are identified by comparing:

```text
Actual Label ≠ Predicted Label
```

Up to three incorrectly classified images are displayed with their actual and predicted labels.

Output:

```text
misclassified_examples.png
```

This helps identify difficult cases such as subtle defects, reflections, lighting variations, or visually similar samples.

---

## 13. Training Visualization

The project generates:

```text
training_curves.png
```

The plot shows:

* Training Loss
* Validation/Test Loss
* Training Accuracy
* Validation/Test Accuracy

These curves help analyze model learning and possible overfitting or underfitting.

---

## 14. Output Files

After execution, the project generates:

```text
training_curves.png
confusion_matrix.png
misclassified_examples.png
mobilenetv2_casting_defect.pth
```

### Model File

```text
mobilenetv2_casting_defect.pth
```

Contains the trained MobileNetV2 model weights.

---

## 15. Project Structure

```text
Automated-Product-Defect-Detection/
│
├── main.py
├── requirements.txt
├── README.md
│
├── casting_data/
│   ├── train/
│   │   ├── def_front/
│   │   └── ok_front/
│   │
│   └── test/
│       ├── def_front/
│       └── ok_front/
│
├── training_curves.png
├── confusion_matrix.png
├── misclassified_examples.png
│
└── mobilenetv2_casting_defect.pth
```

---

## 16. Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
torch
torchvision
numpy
matplotlib
scikit-learn
Pillow
```

---

## 17. Run the Project

Make sure the dataset is placed in the project directory:

```text
main.py
casting_data/
```

Then run:

```bash
python main.py
```

The program will:

1. Load the casting dataset
2. Preprocess the images
3. Apply training augmentation
4. Load pretrained MobileNetV2
5. Train the classifier
6. Evaluate the model
7. Generate performance visualizations
8. Save the trained model

---

## 18. Results

Results should be updated using the final correctly configured training run.

| Metric    | Score |
| --------- | ----: |
| Accuracy  |   TBD |
| Precision |   TBD |
| Recall    |   TBD |
| F1-Score  |   TBD |

The final run should confirm:

```text
Classes found: ['def_front', 'ok_front']
```

Results from an incorrect configuration where:

```text
Classes found: ['test', 'train']
```

should not be used, because in that case the dataset split folders are incorrectly treated as classes.

---

## 19. Key Technologies

```text
Python
PyTorch
Torchvision
MobileNetV2
Transfer Learning
Convolutional Neural Networks
Computer Vision
Scikit-learn
Matplotlib
Pillow
```

---

## 20. Conclusion

This project implements an end-to-end computer vision solution for **automated casting defect detection**.

Using **MobileNetV2 transfer learning**, the system extracts visual features from casting images and classifies them as:

```text
DEFECTIVE
     or
NORMAL
```

The model is evaluated using classification metrics, a confusion matrix, training curves, and misclassified examples to provide a complete view of its performance.
