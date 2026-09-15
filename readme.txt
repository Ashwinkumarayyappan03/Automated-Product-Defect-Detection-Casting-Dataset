# **Automated Product Defect Detection — Casting Dataset**

A computer vision system for automatically classifying manufactured casting images as **Defective** or **Normal** using **MobileNetV2 transfer learning**.

---

## **1. Objective**

The goal of this project is to automate visual inspection of manufactured casting components using **Deep Learning**.

Given a casting image, the model predicts one of two classes:

* **Defective** — Sub-standard casting with physical flaws
* **Normal** — Non-defective, standard casting

---

## **2. Dataset Structure**

The dataset consists of **6,633 total images** split into training and validation sets:

* **Train Samples:** 5,307
* **Validation Samples:** 1,326

```text
casting_data/
├── train/
│   ├── Defective/
│   └── Normal/
│
└── test/
    ├── Defective/
    └── Normal/
```

Classes automatically identified by `ImageFolder`:

```text
Classes found: ['Defective', 'Normal']
```

---

## **3. Preprocessing & Model Architecture**

* **Input Size:** `224 × 224`
* **Augmentations (Training Only):**

  * Random Horizontal Flip
  * Random Vertical Flip
  * Random Rotation (±15°)
  * Brightness Adjustment
  * Contrast Adjustment
* **Base Model:** Pretrained **MobileNetV2** on ImageNet
* **Feature Extractor:** Frozen
* **Classifier:** Custom Linear layer replacing the original classifier head
* **Classifier Architecture:** `1280 → 2`

### **Parameter Summary**

| **Parameter**            |     **Value** |
| ------------------------ | ------------: |
| **Total Parameters**     | **2,226,434** |
| **Trainable Parameters** |     **2,562** |

---

## **4. Hyperparameters & Training Configuration**

| **Parameter**     | **Value**        |
| ----------------- | ---------------- |
| **Model**         | MobileNetV2      |
| **Optimizer**     | Adam             |
| **Loss Function** | CrossEntropyLoss |
| **Learning Rate** | 0.0001           |
| **Batch Size**    | 32               |
| **Epochs**        | 10               |
| **Device**        | CUDA / GPU       |

---

## **5. Performance & Metrics**

The model achieved a **peak validation accuracy of 94.87% on Epoch 8**.

### **Final Validation Metrics**

| **Metric**    |           **Value** |
| ------------- | ------------------: |
| **Accuracy**  | **0.9487 (94.87%)** |
| **Precision** | **0.9624 (96.24%)** |
| **Recall**    | **0.9472 (94.72%)** |
| **F1-Score**  | **0.9547 (95.47%)** |

---

## **6. Visualizations & Error Analysis**

### **Training & Validation Curves**

The training and validation performance across all epochs is shown below.

![Training and Validation Curves](./training_curves.png)

---

### **Confusion Matrix**

The confusion matrix shows how accurately the model classified **Defective** and **Normal** casting images.

Out of **1,326 validation samples**, the model correctly identified:

* **717 Defective** items
* **541 Normal** items

| **Classification**                         | **Count** |
| ------------------------------------------ | --------: |
| **True Positives (Defective → Defective)** |   **717** |
| **False Positives (Normal → Defective)**   |    **28** |
| **False Negatives (Defective → Normal)**   |    **40** |
| **True Negatives (Normal → Normal)**       |   **541** |

![Confusion Matrix](./confusion_matrix.png)

---

### **Misclassified Samples**

Only **68 out of 1,326** total samples were misclassified by the best-performing model.

The following visualization shows examples of incorrectly classified images along with their actual and predicted labels.

![Misclassified Samples](./misclassified_examples.png)

---

## **7. Project Structure**

```text
Automated-Product-Defect-Detection/
│
├── main.py
├── requirement.txt
├── README.md
│
├── casting_data/
│   ├── train/
│   │   ├── Defective/
│   │   └── Normal/
│   │
│   └── test/
│       ├── Defective/
│       └── Normal/
│
├── training_curves.png
├── confusion_matrix.png
├── misclassified_examples.png
│
└── mobilenetv2_casting_defect.pth
```

---

## **8. Quick Start**

### **1. Install Dependencies**

```bash
pip install -r requirement.txt
```

### **2. Run Training and Evaluation**

```bash
python main.py
```

The script will train the MobileNetV2 model, evaluate its performance, generate the visualizations, and save the trained model weights.

---

## **9. Model Output**

The trained model is saved as:

```text
mobilenetv2_casting_defect.pth
```

This file contains the trained **MobileNetV2 classifier weights** and can be used for future inference.

---

## **10. Technologies Used**

* **Python**
* **PyTorch**
* **Torchvision**
* **MobileNetV2**
* **Transfer Learning**
* **Computer Vision**
* **Scikit-learn**
* **NumPy**
* **Matplotlib**
* **Pillow**
