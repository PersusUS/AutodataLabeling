# === evaluation.py ===
import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, AutoModel
import joblib
from autolabelling_system_inferenceflow import *  # <-- import your inference function

# ==== 1. Load pretrained DINOv2 model ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# ==== 2. Load saved clustering artifacts ====
kmeans_model = joblib.load("kmeans_model.pkl")         # saved centroids
cluster_label_names = joblib.load("cluster_labels.pkl")  # saved cluster names

# ==== 3. Load CIFAR-10 dataset ====
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

full_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Pick 100 images per class
N_PER_CLASS = 100
indices, class_counts = [], Counter()
for idx, (_, label) in enumerate(full_dataset):
    if class_counts[label] < N_PER_CLASS:
        indices.append(idx)
        class_counts[label] += 1
    if len(indices) == 10 * N_PER_CLASS:  # stop once we have 1000 images
        break

eval_dataset = torch.utils.data.Subset(full_dataset, indices)

# DataLoader for evaluation
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False)


# ==== 4. Run inference on CIFAR-10 test set ====
predicted_labels = []
true_labels = []
for img, label in cifar10_dataset:
    # assign cluster + label
    cluster_name = label_new_image(
        img,
        processor,
        model,
        cluster_label_names=cluster_label_names,
        device=device,
        kmeans_model=kmeans_model
    )
    predicted_labels.append(cluster_name)
    true_labels.append(label)

# ==== 5. Evaluation ====
# CIFAR-10 canonical class names
cifar10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

true_label_names = [cifar10_names[l] for l in true_labels]

# Accuracy calculation
correct = sum(pred == true for pred, true in zip(predicted_labels, true_label_names))
accuracy = correct / len(true_label_names)
print(f"Labelling accuracy: {accuracy:.3f}")

# ==== 6. Optional: Save predictions for further analysis ====
import pandas as pd
df = pd.DataFrame({"true": true_label_names, "pred": predicted_labels})
df.to_csv("evaluation_results.csv", index=False)
