import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


DATA_DIR   = "casting_data"
IMG_SIZE   = 224
BATCH_SIZE = 32
VAL_SPLIT  = 0.2
EPOCHS     = 10
LR         = 1e-4
SEED       = 42


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(SEED)

print("Using device:", device)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


# Load the TRAIN folder only
TRAIN_DIR = os.path.join(DATA_DIR, "train")

full_dataset = datasets.ImageFolder(
    root=TRAIN_DIR
)

class_names = full_dataset.classes

print("Classes found:", class_names)


val_size = int(
    len(full_dataset) * VAL_SPLIT
)

train_size = len(full_dataset) - val_size


train_subset, val_subset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)


class TransformedSubset(torch.utils.data.Dataset):

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):

        img, label = self.subset[idx]

        img = self.transform(img)

        return img, label


train_dataset = TransformedSubset(
    train_subset,
    train_transform
)

val_dataset = TransformedSubset(
    val_subset,
    val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print(
    f"Train samples: {len(train_dataset)} | "
    f"Val samples: {len(val_dataset)}"
)


model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
)


for param in model.features.parameters():
    param.requires_grad = False


num_features = model.classifier[1].in_features


model.classifier[1] = nn.Linear(
    num_features,
    len(class_names)
)


for param in model.classifier.parameters():
    param.requires_grad = True


model = model.to(device)


trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


total = sum(
    p.numel()
    for p in model.parameters()
)


print(
    f"Trainable params: "
    f"{trainable:,} / {total:,}"
)


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LR
)


history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}


def run_epoch(loader, training):

    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total_samples = 0

    for imgs, labels in loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(imgs)

            loss = criterion(
                outputs,
                labels
            )

            if training:
                loss.backward()
                optimizer.step()

        running_loss += (
            loss.item() * imgs.size(0)
        )

        preds = outputs.argmax(
            dim=1
        )

        correct += (
            preds == labels
        ).sum().item()

        total_samples += labels.size(0)

    return (
        running_loss / total_samples,
        correct / total_samples
    )


best_val_acc = 0.0

best_model_wts = copy.deepcopy(
    model.state_dict()
)


# TRAIN FOR ALL 10 EPOCHS
for epoch in range(1, EPOCHS + 1):

    train_loss, train_acc = run_epoch(
        train_loader,
        training=True
    )

    val_loss, val_acc = run_epoch(
        val_loader,
        training=False
    )

    history["train_loss"].append(
        train_loss
    )

    history["train_acc"].append(
        train_acc
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_acc"].append(
        val_acc
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} "
        f"Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} "
        f"Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        best_model_wts = copy.deepcopy(
            model.state_dict()
        )


model.load_state_dict(
    best_model_wts
)


print(
    f"\nBest validation accuracy: "
    f"{best_val_acc:.4f}"
)


# =========================
# TRAINING CURVES
# =========================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


axes[0].plot(
    history["train_loss"],
    label="Train Loss"
)

axes[0].plot(
    history["val_loss"],
    label="Val Loss"
)

axes[0].set_title(
    "Loss vs Epoch"
)

axes[0].set_xlabel(
    "Epoch"
)

axes[0].set_ylabel(
    "Loss"
)

axes[0].legend()


axes[1].plot(
    history["train_acc"],
    label="Train Accuracy"
)

axes[1].plot(
    history["val_acc"],
    label="Val Accuracy"
)

axes[1].set_title(
    "Accuracy vs Epoch"
)

axes[1].set_xlabel(
    "Epoch"
)

axes[1].set_ylabel(
    "Accuracy"
)

axes[1].legend()


plt.tight_layout()

plt.savefig(
    "training_curves.png",
    dpi=150
)

plt.show()


# =========================
# VALIDATION PREDICTIONS
# =========================

model.eval()

all_preds = []
all_labels = []
all_imgs = []


with torch.no_grad():

    for imgs, labels in val_loader:

        imgs_dev = imgs.to(device)

        outputs = model(
            imgs_dev
        )

        preds = (
            outputs
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        all_preds.extend(
            preds
        )

        all_labels.extend(
            labels.numpy()
        )

        all_imgs.extend(
            imgs.numpy()
        )


all_preds = np.array(
    all_preds
)

all_labels = np.array(
    all_labels
)


# =========================
# FIND DEFECTIVE CLASS
# =========================

def_candidates = [
    c for c in class_names
    if (
        "def" in c.lower()
        or "defect" in c.lower()
    )
]


if len(def_candidates) > 0:

    def_class_name = def_candidates[0]

else:

    def_class_name = class_names[0]

    print(
        "\nWarning: Defective class "
        "could not be identified automatically."
    )


def_class_idx = class_names.index(
    def_class_name
)


# =========================
# METRICS
# =========================

acc = accuracy_score(
    all_labels,
    all_preds
)


prec = precision_score(
    all_labels,
    all_preds,
    average="binary",
    pos_label=def_class_idx,
    zero_division=0
)


rec = recall_score(
    all_labels,
    all_preds,
    average="binary",
    pos_label=def_class_idx,
    zero_division=0
)


f1 = f1_score(
    all_labels,
    all_preds,
    average="binary",
    pos_label=def_class_idx,
    zero_division=0
)


print(
    "\n=== VALIDATION METRICS ==="
)

print(
    f"Accuracy : {acc:.4f}"
)

print(
    f"Precision: {prec:.4f}"
)

print(
    f"Recall   : {rec:.4f}"
)

print(
    f"F1-Score : {f1:.4f}"
)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    all_labels,
    all_preds
)


print(
    "\n=== CONFUSION MATRIX ==="
)

print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)


disp.plot(
    cmap="Blues"
)


plt.title(
    "Confusion Matrix"
)

plt.savefig(
    "confusion_matrix.png",
    dpi=150
)

plt.show()


# =========================
# DENORMALIZE IMAGE
# =========================

def denormalize(img_tensor):

    mean = np.array(
        IMAGENET_MEAN
    ).reshape(3, 1, 1)

    std = np.array(
        IMAGENET_STD
    ).reshape(3, 1, 1)

    img = (
        img_tensor * std
        + mean
    )

    img = np.clip(
        img,
        0,
        1
    )

    return np.transpose(
        img,
        (1, 2, 0)
    )


# =========================
# MISCLASSIFIED IMAGES
# =========================

misclassified_idx = np.where(
    all_preds != all_labels
)[0]


print(
    f"\nTotal misclassified: "
    f"{len(misclassified_idx)} / "
    f"{len(all_labels)}"
)


num_to_show = min(
    3,
    len(misclassified_idx)
)


if num_to_show > 0:

    fig, axes = plt.subplots(
        1,
        num_to_show,
        figsize=(
            5 * num_to_show,
            5
        )
    )


    if num_to_show == 1:

        axes = [axes]


    for i, idx in enumerate(
        misclassified_idx[
            :num_to_show
        ]
    ):

        img = denormalize(
            all_imgs[idx]
        )

        actual = class_names[
            all_labels[idx]
        ]

        predicted = class_names[
            all_preds[idx]
        ]


        axes[i].imshow(
            img
        )


        axes[i].set_title(
            f"Actual: {actual}\n"
            f"Predicted: {predicted}"
        )


        axes[i].axis(
            "off"
        )


    plt.tight_layout()


    plt.savefig(
        "misclassified_examples.png",
        dpi=150
    )


    plt.show()


# =========================
# SAVE MODEL
# =========================

torch.save(
    model.state_dict(),
    "mobilenetv2_casting_defect.pth"
)


print(
    "\nModel saved as "
    "mobilenetv2_casting_defect.pth"
)
