import numpy as np
import torch
from engine.network import SimpleNetwork
from engine.losses import binary_cross_entropy


def verify_against_pytorch(n_samples=20, input_dim=2, hidden_dim=4, seed=42):
    """
    Build the same network architecture in both NumPy and PyTorch,
    initialize them with identical weights, run forward + backward
    passes on identical data, and compare the resulting gradients.
    """
    rng = np.random.default_rng(seed)
    X_np = rng.standard_normal((n_samples, input_dim))
    y_np = rng.integers(0, 2, size=n_samples).astype(float)

    weight_rng = np.random.default_rng(seed + 1)
    W1_np = weight_rng.standard_normal((input_dim, hidden_dim)) * 0.1
    b1_np = np.zeros(hidden_dim)
    W2_np = weight_rng.standard_normal((hidden_dim, 1)) * 0.1
    b2_np = np.zeros(1)

    # --- NumPy engine ---
    net = SimpleNetwork(input_dim=input_dim, hidden_dim=hidden_dim, seed=seed)
    net.W1, net.b1, net.W2, net.b2 = W1_np.copy(), b1_np.copy(), W2_np.copy(), b2_np.copy()

    net.forward(X_np)
    numpy_loss = binary_cross_entropy(y_np, net.y_hat)
    dW1_np, db1_np, dW2_np, db2_np = net.backward(y_np)

    # --- PyTorch reference ---
    X_torch = torch.tensor(X_np, dtype=torch.float64)
    y_torch = torch.tensor(y_np, dtype=torch.float64).reshape(-1, 1)
    W1_t = torch.tensor(W1_np, dtype=torch.float64, requires_grad=True)
    b1_t = torch.tensor(b1_np, dtype=torch.float64, requires_grad=True)
    W2_t = torch.tensor(W2_np, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2_np, dtype=torch.float64, requires_grad=True)

    Z1_t = X_torch @ W1_t + b1_t
    A1_t = torch.relu(Z1_t)
    Z2_t = A1_t @ W2_t + b2_t
    y_hat_t = torch.sigmoid(Z2_t)

    eps = 1e-8
    torch_loss = -torch.mean(
        y_torch * torch.log(y_hat_t + eps) + (1 - y_torch) * torch.log(1 - y_hat_t + eps)
    )
    torch_loss.backward()

    results = {
        "loss": (numpy_loss, torch_loss.item()),
        "W1": (dW1_np, W1_t.grad.numpy()),
        "b1": (db1_np, b1_t.grad.numpy()),
        "W2": (dW2_np, W2_t.grad.numpy()),
        "b2": (db2_np, b2_t.grad.numpy()),
    }

    print(f"{'Quantity':<10} {'Max Abs Diff':<15} {'Status':<10}")
    print("-" * 40)
    all_passed = True
    for name, (numpy_val, torch_val) in results.items():
        if name == "loss":
            diff = abs(numpy_val - torch_val)
        else:
            diff = np.max(np.abs(numpy_val - torch_val))
        passed = diff < 1e-6
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"{name:<10} {diff:<15.2e} {status:<10}")

    return all_passed


if __name__ == "__main__":
    passed = verify_against_pytorch()
    print()
    print("All quantities match PyTorch autograd" if passed else "Mismatch detected")