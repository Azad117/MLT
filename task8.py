import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Generate synthetic data
x, y_true = make_blobs(n_samples=500, centers=3, random_state=42)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
y_means = kmeans.fit_predict(x)

# Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=3, random_state=42)
y_gmm = gmm.fit_predict(x)

# Calculate metrics
ari_kmeans = adjusted_rand_score(y_true, y_means)
ari_gmm = adjusted_rand_score(y_true, y_gmm)
sil_kmeans = silhouette_score(x, y_means)
sil_gmm = silhouette_score(x, y_gmm)

# Print clustering results
print("Clustering Results:")
print(f"K-Means Adjusted Rand Index: {ari_kmeans:.3f}, Silhouette Score: {sil_kmeans:.3f}")
print(f"GMM (EM) Adjusted Rand Index: {ari_gmm:.3f}, Silhouette Score: {sil_gmm:.3f}")

# Plotting results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(x[:, 0], x[:, 1], c=y_means, cmap='viridis')
ax1.set_title('K-Means Clustering Labels')
ax2.scatter(x[:, 0], x[:, 1], c=y_gmm, cmap='viridis')
ax2.set_title('GMM (EM) Clustering Labels')
plt.show()



'''
Clustering Results:
K-Means Adjusted Rand Index: 1.000, Silhouette Score: 0.844
GMM (EM) Adjusted Rand Index: 1.000, Silhouette Score: 0.844

'''
