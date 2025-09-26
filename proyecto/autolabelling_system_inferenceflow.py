import torch
import numpy as np
from torchvision import transforms
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def label_new_image(image, processor, model, cluster_label_names, device, kmeans_model):
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    if not isinstance(image, torch.Tensor):
        image = transform(image).unsqueeze(0)
    else:
        image = image.unsqueeze(0)

    image = image.to(device)
    with torch.no_grad():
        outputs = model(pixel_values=image)
        emb = outputs.last_hidden_state[:,0,:].cpu().numpy()

    centroids = kmeans_model.cluster_centers_
    distances = np.linalg.norm(centroids - emb, axis=1)
    cluster_id = np.argmin(distances)
    return cluster_label_names[cluster_id]
