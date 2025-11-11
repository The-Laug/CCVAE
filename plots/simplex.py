import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
from PIL import Image

# Utility: create vertices of an n-simplex polygon for visualization
def get_polygon_vertices(n_archetypes, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n_archetypes, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

# Utility: load an image and resize it
def load_image(path, size=64):
    img = Image.open(path).convert("L")
    img = img.resize((size, size))
    return np.array(img)

# === Simplified MNIST simplex plot ===
def plot_mnist_simplex(latent_matrix, labels, image_folder,
                       latent_dim=3, fig_size=(8, 8), image_size=64,
                       save_path=None):
    """
    Visualizes the MNIST latent simplex learned by a Dirichlet/CC model.

    latent_matrix: numpy array (n_samples, latent_dim), latent codes (sum to 1)
    labels: list or array of digit labels
    image_folder: folder containing corresponding MNIST images (indexed numerically)
    """

    assert latent_dim == 3, "Simplex plot works only for 3D Dirichlet latent spaces."

    # 1. Setup figure and simplex geometry
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_aspect('equal')
    ax.axis('off')

    n_archetypes = latent_dim
    vertices = get_polygon_vertices(n_archetypes, radius=1.0)

    # 2. Draw edges (triangle)
    for i, j in itertools.combinations(range(n_archetypes), 2):
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

    # 5. Plot representative images at vertices (archetypes)
    for i in range(n_archetypes):
        best_idx = np.argmax(latent_matrix[:, i])
        img_path = f"{image_folder}/{best_idx}.jpg"
        img = load_image(img_path, size=image_size)
        imagebox = OffsetImage(img, zoom=0.25, cmap='gray')
        ab = AnnotationBbox(imagebox, vertices[i], frameon=False, zorder=3)
        ax.add_artist(ab)

    # 6. Optional: midpoint images between archetypes
    midpoint_coords = []
    for i, j in itertools.combinations(range(n_archetypes), 2):
        midpoint = (vertices[i] + vertices[j]) / 2
        midpoint_coords.append(midpoint)
    for midpoint in midpoint_coords:
        circ = plt.Circle(midpoint, 0.02, color='gray', fill=True, alpha=0.4)
        ax.add_artist(circ)

    ax.legend(title="Digits", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
