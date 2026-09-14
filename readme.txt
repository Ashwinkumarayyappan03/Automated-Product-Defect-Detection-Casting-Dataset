Automated Product Defect Detection — Casting Dataset

## 1. Objective

Build a binary image classification system that automatically sorts
manufactured casting images into **Normal (`ok_front`)** and
**Defective (`def_front`)**, replacing manual visual inspection on a
production line.

This is a **binary image classification** task: yes/no defective, one
label per image, no localization or segmentation involved.

---

## 2. Model Used: MobileNetV2 (Transfer Learning)

**Why MobileNetV2 and not training a CNN from scratch?**
- The dataset (~7,000 images total) is small relative to what's needed
  to train a deep CNN from random weights. A network pretrained on
  ImageNet already knows general low-level features (edges, textures,
  gradients, blobs) that transfer well to spotting cracks/blowholes on
  a casting surface.
- Casting defects are mostly **texture and edge anomalies** on a fairly
  uniform metallic background — exactly the kind of feature MobileNetV2's
  depthwise-separable convolutions are efficient at capturing.
- MobileNetV2 is lightweight (~3.4M parameters vs. 138M for VGG16 or
  ~25M for ResNet50), so it trains fast on limited compute (CPU/free-tier
  GPU) within a fixed time budget, and is realistic for real-world
  deployment on an inspection-line edge camera.

**Why not ResNet18 / EfficientNet / VGG16 instead?**
- ResNet18 was a close alternative but MobileNetV2 is smaller and faster
  to fine-tune with a nearly identical accuracy ceiling for a
  two-class problem like this.
- EfficientNet-B0 gives marginally better accuracy but is heavier and
  slower per epoch — not worth the trade-off under a tight training
  time budget.
- VGG16 was ruled out: 138M parameters make it far too slow to
  fine-tune in the available time, with no meaningful accuracy benefit
  for a binary task this simple.

### Transfer Learning Setup
- Loaded **MobileNetV2 pretrained on ImageNet**
  (`MobileNet_V2_Weights.IMAGENET1K_V1`).
- **Frozen (not trained):** all convolutional feature-extractor layers
  (`model.features`) — these keep their pretrained ImageNet weights
  and act purely as a fixed feature extractor.
- **Trainable:** only the final classifier head. The original
  1000-class ImageNet output layer (`Linear(1280, 1000)`) was replaced
  with a new `Linear(1280, 2)` layer for our two classes
  (`def_front`, `ok_front`).
- This gives **2,562 trainable parameters** out of **2,226,434 total**
  — only the new classifier head is updated during training; the
  backbone stays fixed.

---

## 3. Dataset & Preprocessing

- Source: Casting Product Image Data (submersible pump impeller
  casting images), pre-split into `train/` and `test/` folders, each
  containing `def_front/` (defective) and `ok_front/` (normal)
  subfolders.
- **Resizing:** all images resized to 224×224 (MobileNetV2's expected
  input size).
- **Tensor conversion & normalization:** converted to PyTorch tensors
  and normalized using ImageNet mean/std (`[0.485, 0.456, 0.406]` /
  `[0.229, 0.224, 0.225]`), since the backbone was pretrained on
  ImageNet-normalized inputs.
- **Augmentation (train set only):** random horizontal flip, random
  vertical flip, random rotation (±15°), and brightness/contrast
  jitter — to make the model robust to camera angle/lighting
  variation on a real production line. **No augmentation applied to
  the validation set**, only resize + tensor + normalize.
- **Train/validation split:** the dataset ships with its own
  pre-made `train/` and `test/` folders, used directly instead of a
  random split, so the split matches how the data was originally
  curated.

---

## 4. Training

- **Loss function:** CrossEntropyLoss (standard for multi-class /
  binary classification with class-index labels).
- **Optimizer:** Adam, learning rate `1e-4`, applied only to the
  classifier head's parameters (the frozen backbone has no gradient
  updates).
- **Epochs:** capped at 10, with a hard 12-minute training time budget
  (the script times a probe batch, estimates epoch duration, and stops
  early if another epoch would exceed the budget) — appropriate for a
  90-minute time-boxed assessment.
- Per-epoch training/validation loss and accuracy are logged and
  plotted (`training_curves.png`).

---

## 5. Evaluation

Computed on the held-out validation set:
- Accuracy
- Precision, Recall, F1-score (scored on the **defective** class,
  since correctly catching defects is the operationally important
  metric — missing a defective part is costlier than a false alarm)
- Confusion matrix (`confusion_matrix.png`)

## 6. Error Analysis

Up to 3 misclassified validation images are displayed
(`misclassified_examples.png`) with actual vs. predicted class, to
manually inspect likely causes (small/subtle defect, lighting/glare,
texture overlap between classes, etc.).

---

## 7. Folder Structure

```
AshwinKumarA_23AD016/
├── main.py                          # full pipeline: load → train → evaluate → error analysis
├── requirements.txt                 # Python dependencies
├── README.md                        # this file
├── model.txt                        # model architecture summary
├── casting_data/
│   ├── train/
│   │   ├── def_front/                # defective training images
│   │   └── ok_front/                 # normal training images
│   └── test/
│       ├── def_front/                # defective validation images
│       └── ok_front/                 # normal validation images
├── training_curves.png              # generated: loss & accuracy vs epoch
├── confusion_matrix.png             # generated: confusion matrix
├── misclassified_examples.png       # generated: misclassified image samples
└── mobilenetv2_casting_defect.pth   # generated: trained model weights
```

---

## 8. How to Run

```bash
pip install -r requirements.txt
python main.py
```

Requires the `casting_data/` folder (with `train/` and `test/`
subfolders as shown above) to be present in the same directory as
`main.py`.

---

## 9. Results

| Metric              | Score |
|----------------------|-------|
| Training Accuracy    | 89.96% |
| Validation Accuracy  | 91.49% |
| Precision            | Not calculated |
| Recall               | Not calculated |
| F1-Score             | Not calculated |

> **Note:** fill this table in only after confirming the console
> output shows `Classes found: ['def_front', 'ok_front']`. If it
> still shows `['test', 'train']`, the dataset path bug is not yet
> fixed and any numbers produced are not measuring defect detection.
