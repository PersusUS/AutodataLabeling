# Evaluate accuracy against true_labels (assuming cluster_label_names are CIFAR-10 names)
# Map true_labels to their string names if needed
cifar10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
true_label_names = [cifar10_names[l] for l in true_labels]

# Compute accuracy
correct = sum([pred == true for pred, true in zip(predicted_labels, true_label_names)])
accuracy = correct / len(true_label_names)
print(f"Labelling accuracy: {accuracy:.3f}")