import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# TensorFlow Playground style colormap: orange (class 0) -> white -> blue (class 1)
TF_PLAYGROUND_CMAP = LinearSegmentedColormap.from_list(
    "tf_playground",
    ["#F59322", "#FFFFFF", "#0877BD"]
)


def plot_decision_boundary(net, X, y, title="Decision Boundary", save_path=None):
    """
    Plot the decision boundary learned by a trained network on a 2D dataset,
    using a TensorFlow Playground style orange/blue color scheme.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    h = 0.02  # grid resolution
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = net.forward(grid_points)
    predictions = predictions.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))

    contour = ax.contourf(
        xx, yy, predictions, levels=50, cmap=TF_PLAYGROUND_CMAP, alpha=0.8
    )
    ax.contour(xx, yy, predictions, levels=[0.5], colors="black", linewidths=2)

    scatter = ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap=TF_PLAYGROUND_CMAP,
        edgecolors="black", s=40, vmin=0, vmax=1
    )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    fig.colorbar(contour, ax=ax, label="Predicted probability")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig