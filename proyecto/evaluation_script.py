# === evaluation.py ===
import torch
import numpy as np
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel
import joblib
from autolabelling_system_inferenceflow import label_new_image  # import your inference function
from collections import Counter
import pandas as pd

# ==== 1. Load pretrained DINOv2 model ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# ==== 2. Load saved clustering artifacts ====
kmeans_model = joblib.load("kmeans_model.pkl")           # saved centroids
cluster_label_names = joblib.load("cluster_labels.pkl")  # saved cluster names

# ==== 3. Load balanced CIFAR-10 dataset (100 per class) ====
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

full_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Select 100 samples per class
N_PER_CLASS = 100
indices, class_counts = [], Counter()
for idx, (_, label) in enumerate(full_dataset):
    if class_counts[label] < N_PER_CLASS:
        indices.append(idx)
        class_counts[label] += 1
    if len(indices) == 10 * N_PER_CLASS:
        break

eval_dataset = torch.utils.data.Subset(full_dataset, indices)

print(f"Evaluation dataset size: {len(eval_dataset)} (100 per class)")

# ==== 4. Run inference ====
predicted_labels = []
true_labels = []

for img, label in eval_dataset:
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
cifar10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']
true_label_names = [cifar10_names[l] for l in true_labels]

correct = sum(pred == true for pred, true in zip(predicted_labels, true_label_names))
accuracy = correct / len(true_label_names)
print(f"Labelling accuracy: {accuracy:.3f}")

# ==== 6. Save predictions ====
df = pd.DataFrame({"true": true_label_names, "pred": predicted_labels})
df.to_csv("evaluation_results.csv", index=False)
print("Evaluation results saved to evaluation_results.csv")
