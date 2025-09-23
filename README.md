# Data Auto-Labeling

This project automates the process of labeling image datasets using unsupervised learning and **CLIP-based semantic matching**.  
It’s designed to help you turn large collections of unlabeled images into structured, labeled datasets with minimal manual effort.  

---

## 📌 Pipeline

The pipeline follows these steps:

1. **Input Images**  
   Raw images are passed into the system.  

2. **Autoencoder Embedding**  
   An autoencoder extracts **latent embeddings** that capture the essential visual features of each image.  

3. **Clustering**  
   Images are grouped into clusters based on their embeddings (e.g., with KMeans).  

4. **Centroid Selection**  
   For each cluster, the centroid is computed. The image closest to this centroid is chosen as the cluster representative.  

5. **CLIP Labeling**  
   Representative images are fed into **CLIP** (Contrastive Language-Image Pretraining) to generate semantic labels.  
   These labels are then assigned to all images in the corresponding cluster.  

---

## 🔑 Features
- **Scalable** → Handles large unlabeled datasets.  
- **Semantic-Aware** → Labels generated via CLIP provide human-like understanding.  
- **Consistent** → Cluster-level labeling ensures visually similar images share the same tag.  

---
