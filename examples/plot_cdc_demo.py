"""
=============================================================
CDC Clustering Demo
=============================================================

Demonstrates the CDC clustering algorithm on a synthetic dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from cdc import CDC, cdc_cluster

def plot_clustering(X, labels, title):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"{title} - Number of clusters: {n_clusters}")
    if -1 in unique_labels:
        print("Noise points detected")

    plt.figure(figsize=(8, 6))
    
    # Create colors
    unique_labels_set = set(unique_labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels_set))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title(title)
    plt.show()

# Example 1: Class usage on Moons
print("Running CDC Class on Moons dataset...")
X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
cdc_moons = CDC(n_neighbors=10, ratio=0.9)
cdc_moons.fit(X_moons)
# plot_clustering(X_moons, cdc_moons.labels_, "CDC (Class) on Moons")

# Example 2: Function usage on Blobs
print("Running CDC Function on Blobs dataset...")
X_blobs, _ = make_blobs(n_samples=200, centers=3, random_state=42)
labels_blobs = cdc_cluster(X_blobs, n_neighbors=20, ratio=0.9)
# plot_clustering(X_blobs, labels_blobs, "CDC (Function) on Blobs")

print("CDC Demo ran successfully. Uncomment plot lines to see figures.")
