import random
import joblib
import torch
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel
from autolabelling_system_trainflow import *
from autolabelling_system_inferenceflow import *
from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # === Train and Save ===
    train_flow()

    # === Load trained artifacts ===
    k_model = joblib.load("kmeans_model.pkl")
    cluster_label_names = joblib.load("cluster_labels.pkl")
    processor = joblib.load("processor.pkl")

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

    # === Inference example ===
    idx = random.randint(0, len(dataset)-1)
    pil_img, true_label = dataset[idx]
    cifar10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck']

    predicted_label = label_new_image(
        pil_img,
        processor,
        model,
        cluster_label_names=cluster_label_names,
        device=device,
        kmeans_model=k_model
    )

    print("\n=== Inference Example ===")
    print(f"True Label: {cifar10_names[true_label]}")
    print(f"Predicted Cluster Label: {predicted_label}")
    cluster_assignments = k_model.predict(all_embeddings)
    n_clusters = len(set(cluster_assignments))

    for cluster_id in range(n_clusters):
        indices = np.where(cluster_assignments == cluster_id)[0]
        if len(indices) == 0:
            continue

        example_idxs = np.random.choice(indices, size=min(10, len(indices)), replace=False)

        plt.figure(figsize=(15,3))
        for i, ex_idx in enumerate(example_idxs):
            plt.subplot(1, len(example_idxs), i+1)
            plt.imshow(train_images[ex_idx])  # train_images must be stored as PIL/numpy in train_flow
            plt.axis("off")
        plt.suptitle(f"Cluster {cluster_id} → {cluster_label_names.get(cluster_id, 'Unknown')}")
        plt.show()