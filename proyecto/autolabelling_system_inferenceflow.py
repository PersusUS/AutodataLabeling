def label_new_image(image, processor, model, cluster_label_names, device, kmeans_model=k_model):
    """
    Label a new image by assigning it to the nearest cluster and returning the cluster's name.

    Args:
        image: PIL.Image or torch.Tensor
        processor: AutoImageProcessor used for embeddings
        model: DINOv2 model
        kmeans_model: trained KMeans model
        cluster_label_names: list of names for each cluster (cluster ID → name)
        device: torch device

    Returns:
        cluster_name: str, name of the assigned cluster
        cluster_id: int, ID of the assigned cluster
        distance: float, distance to the cluster centroid
    """
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
        emb = outputs.last_hidden_state[:,0,:].cpu().numpy()  # shape (1, embedding_dim)

    centroids = kmeans_model.cluster_centers_
    distances = np.linalg.norm(centroids - emb, axis=1)
    cluster_id = np.argmin(distances)
    cluster_name = cluster_label_names[cluster_id]
    
    return cluster_name
def unnormalize(img_tensor, mean, std):
    """
    img_tensor: torch tensor of shape (C,H,W) normalized
    mean, std: lists of length C
    """
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)  # unnormalize
    return img_tensor
predicted_labels = []
for idx in range(len(original_images)):
    query_image = original_images[idx].clone()  # clone to avoid modifying original

    # Unnormalize
    query_image = unnormalize(query_image, processor.image_mean, processor.image_std)

    # Convert to PIL
    img_to_show = transforms.ToPILImage()(query_image)

    # Assign cluster
    cluster_name = label_new_image(
        img_to_show,
        processor,
        model,
        cluster_label_names=cluster_label_names,
        device=device
    )
    predicted_labels.append(cluster_name)
    