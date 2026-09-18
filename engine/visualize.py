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

    Parameters:
        net: a trained SimpleNetwork instance with a forward(X) method
        X: input data of shape (n_samples, 2)
        y: true labels of shape (n_samples,)
        title: plot title
        save_path: if provided, saves the figure to this path instead of
                    only displaying it
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    h = 0.02  # grid resolution
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]