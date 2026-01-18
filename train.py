import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# -------------------------
# CIFAR-10 normalization
# -------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

# Transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# Datasets
train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

# Loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

print("Train batches:", len(train_loader))
print("Test batches:", len(test_loader))

# -------------------------
# Model & Training
# -------------------------
import torch.nn as nn
import torch.optim as optim
import wandb
from model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="cnn-cifar10", name="cnn-with-augmentation")

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 30

for epoch in range(epochs):
    model.train()
    correct, total, running_loss = 0, 0, 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    train_acc = 100. * correct / total

    # ---- Test ----
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    test_acc = 100. * correct / total

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": running_loss / len(train_loader),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc
    })

    print(f"Epoch [{epoch+1}/{epochs}] | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

print("Training complete.")

# -------------------------
# Evaluation utilities
# -------------------------
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

def get_misclassified(model, loader, device, max_samples=25):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append((
                        images[i].cpu(),
                        labels[i].cpu().item(),
                        preds[i].cpu().item()
                    ))
                if len(misclassified) >= max_samples:
                    return misclassified

    return misclassified

# -------------------------
# Confusion Matrix
# -------------------------
labels, preds = evaluate(model, test_loader)
cm = confusion_matrix(labels, preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix — CIFAR-10 CNN (Aug)")
plt.show()

class_names = train_dataset.classes
per_class_acc = cm.diagonal() / cm.sum(axis=1)

for i, acc in enumerate(per_class_acc):
    print(f"{class_names[i]}: {acc * 100:.2f}%")

# -------------------------
# Error Analysis (Day 3)
# -------------------------
def plot_misclassified(samples, class_names):
    plt.figure(figsize=(10, 10))

    mean = torch.tensor(CIFAR_MEAN)
    std  = torch.tensor(CIFAR_STD)

    for idx, (img, true_label, pred_label) in enumerate(samples):
        img = img.permute(1, 2, 0)
        img = img * std + mean
        img = img.clamp(0, 1)

        plt.subplot(5, 5, idx + 1)
        plt.imshow(img)
        plt.title(f"T: {class_names[true_label]}\nP: {class_names[pred_label]}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# ---- RUN ERROR ANALYSIS ----
misclassified = get_misclassified(model, test_loader, device, max_samples=25)
plot_misclassified(misclassified, class_names)
