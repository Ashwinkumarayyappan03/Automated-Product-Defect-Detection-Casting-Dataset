import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================
# SETTINGS
# =========================

DATA_DIR = r"C:\Users\Admin\Downloads\AshwinKumarA_23AD016\casting_data"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "mobilenetv2_casting_defect.pth"

IMG_SIZE = 224
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TEST DATA
# =========================

# Renamed variable to test_transforms so it matches line 34
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,  # Uses defined TEST_DIR path
    transform=test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.classes

print("Test classes:", class_names)
print("Test samples:", len(test_dataset))

# =========================
# LOAD MODEL
# =========================

model = models.mobilenet_v2(
    weights=None
)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()

# =========================
# PREDICTIONS
# =========================

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# =========================
# METRICS
# =========================

accuracy = accuracy_score(
    all_labels,
    all_preds
)

precision = precision_score(
    all_labels,
    all_preds,
    average="binary",
    pos_label=class_names.index("Defective"),
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_preds,
    average="binary",
    pos_label=class_names.index("Defective"),
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_preds,
    average="binary",
    pos_label=class_names.index("Defective"),
    zero_division=0
)

print("\n=== TEST RESULTS ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(all_labels, all_preds))