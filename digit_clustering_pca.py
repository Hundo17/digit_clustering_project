import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# 1. Load the handwritten digits dataset
digits = datasets.load_digits()
X = digits.data          # flattened pixel values
y = digits.target        # true labels (used only for evaluation)

print("Dataset loaded for clustering.")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features per sample (pixels): {X.shape[1]}")

# 2. Reduce dimensionality to 2D using PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

print("PCA dimensionality reduction complete.")

# 3. Visualize digits in 2D space, colored by true label
plt.figure(figsize=(8, 6))
scatter_true = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=15)
plt.legend(*scatter_true.legend_elements(), title="Digit", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("PCA Projection of Handwritten Digits (colored by true label)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.savefig("pca_digits_true_labels.png")
plt.show()

print("PCA visualization (true labels) saved as pca_digits_true_labels.png")

# 4. Apply KMeans clustering in the original feature space
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

print("KMeans clustering complete.")

# 5. Evaluate clustering quality using Adjusted Rand Index (ARI)
ari = adjusted_rand_score(y, cluster_labels)
print(f"Adjusted Rand Index (ARI) between clusters and true labels: {ari:.4f}")

# 6. Visualize 2D PCA points colored by cluster assignments
plt.figure(figsize=(8, 6))
scatter_clusters = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', s=15)
plt.legend(*scatter_clusters.legend_elements(), title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("PCA Projection of Digits (colored by KMeans cluster)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.savefig("pca_digits_cluster_labels.png")
plt.show()

print("PCA visualization (cluster labels) saved as pca_digits_cluster_labels.png")

# 7. Visualize cluster centers as 8x8 images
centers = kmeans.cluster_centers_.reshape(n_clusters, 8, 8)

plt.figure(figsize=(8, 4))
for i in range(n_clusters):
    plt.subplot(2, 5, i + 1)
    plt.imshow(centers[i], cmap='gray')
    plt.title(f"Cluster {i}")
    plt.axis('off')

plt.suptitle("KMeans Cluster Centers (as digit-like images)")
plt.tight_layout()
plt.savefig("kmeans_cluster_centers.png")
plt.show()

print("Cluster center images saved as kmeans_cluster_centers.png")