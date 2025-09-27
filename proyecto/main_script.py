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
def show_cluster_images(images, labels, cluster_id, num_images=16):
    """Display sample images from a specific cluster."""
    idxs = np.where(labels == cluster_id)[0]
    if len(idxs) == 0:
        print(f"No images found for cluster {cluster_id}.")
        return
    idxs = np.random.choice(idxs, size=min(num_images, len(idxs)), replace=False)
    cols = int(np.sqrt(len(idxs)))
    rows = int(np.ceil(len(idxs)/cols))

    plt.figure(figsize=(cols*2, rows*2))
    for i, idx in enumerate(idxs):
        img = images[idx].permute(1,2,0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f"Cluster {cluster_id} Samples")
    plt.show()

# ==== Example usage ====
labels=joblib.load("kmeans_labels.pkl")
show_cluster_images(dataset, labels, cluster_id=0, num_images=16)
