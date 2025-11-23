import numpy as np
import matplotlib.pyplot as plt
import itertools
# Utility: create vertices of an n-simplex polygon for visualization

def get_polygon_vertices(latent_dim, radius=1.0):
    if latent_dim == 3:
        angles = np.linspace(0, 2 * np.pi, latent_dim, endpoint=False)+np.pi/2
    else:
        angles = np.linspace(0, 2 * np.pi, latent_dim, endpoint=False)
        
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

# === Simplified MNIST simplex plot ===
def plot_mnist_simplex(latent_matrix, labels,
                       latent_dim=3, fig_size=(8, 8), ax=None):
    """
    Visualizes the MNIST latent simplex learned by a Dirichlet/CC model.

    latent_matrix: numpy array (n_samples, latent_dim), latent codes (sum to 1)
    labels: list or array of digit labels
    """
    should_show = ax == None
    # 1. Setup figure and simplex geometry
    if should_show:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_aspect('equal')
        ax.axis('off')

    vertices = get_polygon_vertices(latent_dim, radius=1.0)

    # 2. Draw edges (triangle)
    for i, j in itertools.combinations(range(latent_dim), 2):
        ax.plot([vertices[i, 0], vertices[j, 0]],
                [vertices[i, 1], vertices[j, 1]],
                'k--', alpha=0.4, lw=1)

    # 3. Project latent points to 2D via convex combination
    projected = latent_matrix @ vertices

    # 4. Plot latent points, colored by digit label
    cmap = plt.get_cmap('tab10')
    for digit in np.unique(labels):
        idxs = np.where(labels == digit)[0]
        ax.scatter(projected[idxs, 0], projected[idxs, 1],
                   s=10, alpha=0.6, color=cmap(digit % 10), label=str(digit))
        

    ax.legend(title="Digits", bbox_to_anchor=(1.05, 1), loc='upper left')
    if should_show:
        plt.tight_layout()
        plt.show()

