import torch
import numpy as np
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
from collections import Counter
from kneed import KneeLocator
from PIL import Image
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_flow():
    # === Load models ===
    dino_model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(dino_model_name)
    model = AutoModel.from_pretrained(dino_model_name).to(device)
    model.eval()

    clip_model_name = "openai/clip-vit-base-patch16"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()

    # === Load CIFAR10 ===
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
        if len(indices) == 10 * N_PER_CLASS:
            break

    subset = torch.utils.data.Subset(full_dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)

    # === Extract embeddings ===
    all_embeddings, all_labels, original_images = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(pixel_values=images)
            cls_emb = outputs.last_hidden_state[:,0,:].cpu().numpy()
            all_embeddings.append(cls_emb)
            all_labels.append(labels.numpy())
            original_images.append(images.cpu())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    original_images = torch.cat(original_images, dim=0)

    print(f"Balanced subset: {len(all_labels)} images, embeddings shape {all_embeddings.shape}")

    # === Find optimal K ===
    wcss, K_range = [], range(1, 15)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(all_embeddings)
        wcss.append(km.inertia_)

    knee = KneeLocator(K_range, wcss, curve="convex", direction="decreasing")
    best_k = knee.knee or 10
    print(f"Best number of clusters = {best_k}")

    k_model = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(all_embeddings)
    labels_kmeans = k_model.predict(all_embeddings)

    # === Label clusters with CLIP ===
    def get_cluster_centroid_images(images, embeddings, cluster_labels, n_clusters, top_k=3):
        centroids = []
        for c in range(n_clusters):
            idxs = np.where(cluster_labels == c)[0]
            cluster_embs = embeddings[idxs]
            centroid = cluster_embs.mean(axis=0)
            dists = np.linalg.norm(cluster_embs - centroid, axis=1)
            top_idxs = idxs[np.argsort(dists)[:top_k]]
            centroids.append(top_idxs)
        return centroids

    def clip_label_mode(images, indices, clip_model, clip_processor, device):
        candidate_labels = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck", "animal", "vehicle"
        ]
        captions = []
        for i in indices:
            img = images[i].permute(1,2,0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            pil_img = Image.fromarray((img*255).astype(np.uint8)).convert("RGB")

            inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)

            text_inputs = clip_processor(text=candidate_labels, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                text_features = clip_model.get_text_features(**text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T
            best_idx = similarity.argmax().item()
            captions.append(candidate_labels[best_idx])

        return Counter(captions).most_common(1)[0][0]

    centroid_indices = get_cluster_centroid_images(original_images, all_embeddings, labels_kmeans, best_k, top_k=5)
    cluster_label_names = []
    for c, idxs in enumerate(centroid_indices):
        label = clip_label_mode(original_images, idxs, clip_model, clip_processor, device)
        cluster_label_names.append(label)
        print(f"Cluster {c}: {label}")

    # === Save artifacts ===
    joblib.dump(k_model, "kmeans_model.pkl")
    joblib.dump(cluster_label_names, "cluster_labels.pkl")
    joblib.dump(processor, "processor.pkl")
    print("Training done. Models saved!")
