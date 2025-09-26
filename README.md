# AutoData Labeling System - K-means Edition

Sistema automatizado de etiquetado de imágenes que utiliza **DINOv2 embeddings**, **K-means clustering** y **CLIP** para generar etiquetas semánticas de forma automática y escalable.

**🔥 Ahora optimizado con K-means clustering para mejor rendimiento y métricas avanzadas.**ta Auto-Labeling

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

### 4. **Selección de Representativas con K-means**
- Centroides calculados automáticamente por K-means
- Selección de imágenes más cercanas al centroide de cada cluster
- Balance entre representatividad y diversidad

### 5. **Etiquetado Semántico con CLIP**
- Generación automática de etiquetas usando **CLIP ViT-B/32**
- Etiquetas contextualmente relevantes
- Asignación de etiquetas a todos los miembros del cluster

---

## 🔑 Características K-means

- **✨ Algoritmo Optimizado** → K-means con inicialización k-means++ para mejor convergencia
- **📊 Métricas Avanzadas** → Inercia, Silhouette Score, Calinski-Harabasz Index
- **⚡ Escalable** → Maneja datasets grandes eficientemente  
- **🎯 Configurable** → Número de clusters y parámetros ajustables
- **📈 Auto-optimización** → Búsqueda automática de k óptimo

---

## �️ Instalación Rápida

```bash
# Clonar e instalar
git clone <repo-url>
cd AutodataLabeling
pip install -r requirements.txt
```

## 🚀 Uso con K-means

```python
from src.core import AutoDataLabelingPipeline

# Pipeline optimizado con K-means
pipeline = AutoDataLabelingPipeline(
    n_clusters=8,           # Número de clusters
    kmeans_init="k-means++", # Inicialización optimizada
    embedding_model_size="base"
)

# Entrenar y obtener métricas
dataset = pipeline.train_pipeline("path/to/images")
info = pipeline.clusterer.get_cluster_info()
print(f"Inercia: {info['inertia']}")
print(f"Silhouette Score: {info['silhouette_avg']}")
```

---
