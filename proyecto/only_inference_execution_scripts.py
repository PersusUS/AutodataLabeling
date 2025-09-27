import random
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel
from autolabelling_system_trainflow import *
from autolabelling_system_inferenceflow import *

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load trained artifacts ===
k_model = joblib.load("kmeans_model.pkl")
cluster_label_names = joblib.load("cluster_labels.pkl")
processor = joblib.load("processor.pkl")
all_embeddings = joblib.load("all_embeddings.pkl")   # embeddings from training
train_images = joblib.load("train_images.pkl")       # original training images (or paths)

# Reload DINOv2 for inference
dino_model_name = "facebook/dinov2-base"
model = AutoModel.from_pretrained(dino_model_name).to(device)
model.eval()

# === Load CIFAR10 for testing ===
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])
dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# CIFAR10 names
cifar10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

# === Inference example ===
idx = random.randint(0, len(dataset)-1)
pil_img, true_label = dataset[idx]

predicted_label, pred_cluster_idx = label_new_image(
    pil_img,
    processor,
    model,
    cluster_label_names=cluster_label_names,
    device=device,
    kmeans_model=k_model,
    return_cluster_idx=True   # <- make sure your inference function supports returning cluster index
)

print("\n=== Inference Example ===")
print(f"True Label: {cifar10_names[true_label]}")
print(f"Predicted Cluster: {pred_cluster_idx} → {predicted_label}")

# === Show the queried image ===
plt.figure(figsize=(2,2))
plt.imshow(transforms.ToPILImage()(pil_img))
plt.axis("off")
plt.title(f"Query Image\nTrue: {cifar10_names[true_label]}\nPred: {predicted_label}")
plt.show()

# === Show example images from same cluster ===
cluster_assignments = k_model.predict(all_embeddings)
same_cluster_idxs = np.where(cluster_assignments == pred_cluster_idx)[0]

if len(same_cluster_idxs) > 0:
    example_idxs = np.random.choice(same_cluster_idxs, size=min(5, len(same_cluster_idxs)), replace=False)
    plt.figure(figsize=(12,3))
    for i, ex_idx in enumerate(example_idxs):
        plt.subplot(1, len(example_idxs), i+1)
        plt.imshow(train_images[ex_idx])   # assumes `train_images` are stored as PIL images or numpy arrays
        plt.axis("off")
        plt.title(f"Cluster {pred_cluster_idx}")
    plt.suptitle("Example Images from Predicted Cluster")
    plt.show()
else:
    print("⚠️ No examples found for this cluster.")
