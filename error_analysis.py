import os
import torch
import matplotlib.pyplot as plt

def get_misclassified(model, loader, class_names, device, max_samples=25):
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
