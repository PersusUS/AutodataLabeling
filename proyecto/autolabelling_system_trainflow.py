import os
import numpy as np
import torch
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import mode
from kneed import KneeLocator
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from collections import Counter

# Load CLIP model and processor
clip_model_name = "openai/clip-vit-base-patch16"
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_model.eval()
# ==== 1. Load pretrained DINOv2 ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'facebook/dinov2-base'
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# ==== 2. Balanced CIFAR-10 subset ====
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

full_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

N_PER_CLASS = 100
indices, class_counts = [], Counter()
for idx, (_, label) in enumerate(full_dataset):
    if class_counts[label] < N_PER_CLASS:
        indices.append(idx)
        class_counts[label] += 1
    if len(indices) == 10 * N_PER_CLASS:
        break

subset = torch.utils.data.Subset(full_dataset, indices)
dataloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)

# ==== 3. Extract DINOv2 embeddings ====
all_embeddings, all_labels, original_images = [], [], []
with torch.no_grad():
    for images, labels in dataloader:
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

# ==== 4. Helper functions ====
def cluster_purity(pred_labels, true_labels):
    clusters = set(pred_labels)
    total_correct = 0
    for c in clusters:
        if c == -1:
            continue
        mask = pred_labels == c
        if np.sum(mask) == 0:
            continue
        most_common = mode(true_labels[mask], keepdims=False).mode
        total_correct += np.sum(true_labels[mask] == most_common)
    return total_correct / len(true_labels)

# ==== 5. KMeans with Automatic Elbow Detection ====
wcss, K_range = [], range(1, 15)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(all_embeddings)
    wcss.append(km.inertia_)

knee = KneeLocator(K_range, wcss, curve='convex', direction='decreasing')
best_k = knee.knee
print(f"Automatically detected optimal K: {best_k}")

plt.figure(figsize=(8,5))
plt.plot(K_range, wcss, marker='o', label='WCSS')
if best_k:
    plt.axvline(best_k, color='red', linestyle='--', label=f'Elbow at k={best_k}')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.xticks(K_range)
plt.legend()
plt.grid(True)
plt.show()
k_model = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(all_embeddings)
labels_kmeans = k_model.predict(all_embeddings)

purity = cluster_purity(labels_kmeans, all_labels)


# === 1. Get top-k images closest to each cluster centroid ===
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

# === 2. Generate text labels using CLIP (without predefined candidates) ===
def clip_label_mode(images, indices, clip_model, clip_processor, device):
    """Assign a single-word label to a cluster using the mode of CLIP-generated captions."""
    captions = []
    for i in indices:
        img = images[i].permute(1,2,0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        pil_img = Image.fromarray((img*255).astype(np.uint8)).convert("RGB")

        # Use CLIP to encode the image
        inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)

        # Compare against a set of single-word candidates (you can define dynamic words if needed)
        # Here we use the top-k closest words from CLIP’s tokenizer vocabulary
        # For simplicity, let's use a set of common nouns as candidates
        candidate_labels = [ "airplane", "automobile", "bird", "cat", "deer", "dog", "frog","mammal", "horse", "ship", "truck",'animals','others' ]
        text_inputs = clip_processor(text=candidate_labels, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)

        # Compute similarity
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = image_features_norm @ text_features_norm.T
        best_idx = similarity.argmax().item()
        captions.append(candidate_labels[best_idx])

    # Return the mode (most frequent) label among the top 3 images
    most_common_label = Counter(captions).most_common(1)[0][0]
    return most_common_label

# === 3. Apply to clusters ===
cluster_labels = labels_kmeans
n_clusters = len(set(cluster_labels))
centroid_indices = get_cluster_centroid_images(original_images, all_embeddings, cluster_labels, n_clusters, top_k=5)

cluster_label_names = []
for c, idxs in enumerate(centroid_indices):
    label = clip_label_mode(original_images, idxs, clip_model, clip_processor, device)
    cluster_label_names.append(label)
    print(f"Cluster {c}: {label}")

