# Math Derivation: From Chain Rule to backward()

This document derives the backpropagation equations used in this project by hand, and shows exactly how each term in the math corresponds to a line of code in engine/network.py and engine/mnist_network.py. The goal is a one-to-one mapping: every symbol on the page should be traceable to a specific variable in the code.

---

## 1. Network Definition (Toy Network)

The toy network used in engine/network.py is a two-layer fully connected network for binary classification:

    Z1 = X @ W1 + b1
    A1 = ReLU(Z1)
    Z2 = A1 @ W2 + b2
    y_hat = sigmoid(Z2)

Shapes (with n = number of samples):
- X: (n, 2), W1: (2, 4), b1: (4,)
- Z1, A1: (n, 4), W2: (4, 1), b2: (1,)
- Z2, y_hat: (n, 1)

## 2. Loss Function

Binary cross-entropy:

    L = -mean( y * log(y_hat) + (1 - y) * log(1 - y_hat) )

If the true label y = 1 and y_hat = 0.9, then L = -log(0.9) ≈ 0.105 (small loss, good prediction).
If y = 1 and y_hat = 0.1, then L = -log(0.1) ≈ 2.303 (large loss, bad prediction).

This is implemented in engine/losses.py::binary_cross_entropy.

## 3. Why We Need the Gradient

For any parameter w, the derivative dL/dw tells us: if we nudge w slightly, does the loss go up or down, and how fast? The basic gradient descent update rule is:

    w_new = w_old - lr * dL/dw

If dL/dw > 0, increasing w increases the loss, so we decrease w. If dL/dw < 0, we increase w. This is implemented in engine/network.py::update().

## 4. Backpropagation: Deriving dL/dZ2

We start from the loss and work backward using the chain rule. The first quantity we need is dL/dZ2, since Z2 is the last linear layer before the sigmoid.

For the combination of sigmoid activation + binary cross-entropy loss, a key simplification happens. Writing out the chain rule:

    dL/dZ2 = (dL/dy_hat) * (dy_hat/dZ2)

We know:
- dL/dy_hat = -(y/y_hat - (1-y)/(1-y_hat))
- dy_hat/dZ2 = y_hat * (1 - y_hat)   [this is the derivative of sigmoid]

Multiplying these together and simplifying algebraically:

    dL/dZ2 = -(y/y_hat - (1-y)/(1-y_hat)) * y_hat * (1 - y_hat)
           = -(y*(1-y_hat) - (1-y)*y_hat)
           = -(y - y*y_hat - y_hat + y*y_hat)
           = -(y - y_hat)
           = y_hat - y

So the messy combination of two derivatives collapses into a remarkably simple expression: dL/dZ2 = y_hat - y. (Dividing by n accounts for the mean in the loss function.)

Code correspondence (engine/network.py::backward):

    dZ2 = (self.y_hat - y_true) / n

This single line is the direct implementation of the derivation above.

## 5. Backpropagating to W2 and b2

Since Z2 = A1 @ W2 + b2, we need dZ2/dW2 and dZ2/db2:

    dZ2/dW2 = A1^T   (transpose of A1, due to matrix calculus rules for Z = A @ W)
    dZ2/db2 = 1      (summed over samples)

By the chain rule:

    dL/dW2 = A1^T @ dL/dZ2
    dL/db2 = sum(dL/dZ2, over samples)

Code correspondence:

    dW2 = self.A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

## 6. Backpropagating Through the Hidden Layer

To continue backward, we need dL/dA1, which requires undoing the matrix multiplication Z2 = A1 @ W2:

    dL/dA1 = dL/dZ2 @ W2^T

Code correspondence:

    dA1 = dZ2 @ self.W2.T

Next, we must pass through the ReLU activation. Since A1 = ReLU(Z1), the derivative of ReLU is:

    ReLU'(z) = 1 if z > 0, else 0

So:

    dL/dZ1 = dL/dA1 * ReLU'(Z1)

This is an element-wise multiplication (Hadamard product), not a matrix multiplication, because ReLU is applied element-wise.

Code correspondence:

    dZ1 = dA1 * relu_derivative(self.Z1)

where relu_derivative(z) = (z > 0).astype(float) in engine/activations.py.

## 7. Backpropagating to W1 and b1

Following the exact same pattern as step 5, but one layer earlier:

    dL/dW1 = X^T @ dL/dZ1
    dL/db1 = sum(dL/dZ1, over samples)

Code correspondence:

    dW1 = self.X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

## 8. Summary: The Full Chain

Putting it all together, backpropagation for this network is nothing more than the chain rule applied five times in sequence, moving backward through the computation graph built during the forward pass:

    dL/dZ2 = y_hat - y_true                (step 4)
    dL/dW2 = A1^T @ dL/dZ2                 (step 5)
    dL/db2 = sum(dL/dZ2)                   (step 5)
    dL/dA1 = dL/dZ2 @ W2^T                 (step 6)
    dL/dZ1 = dL/dA1 * ReLU'(Z1)            (step 6)
    dL/dW1 = X^T @ dL/dZ1                  (step 7)
    dL/db1 = sum(dL/dZ1)                   (step 7)

This entire sequence is implemented in engine/network.py::backward().

## 9. Extending to Softmax + Categorical Cross-Entropy (MNIST Network)

The MNIST network (engine/mnist_network.py) has an extra layer and uses softmax + categorical cross-entropy instead of sigmoid + binary cross-entropy, for multi-class classification (10 digit classes).

Softmax converts raw output scores (logits) z_1, ..., z_10 into a probability distribution:

    p_k = e^(z_k) / sum_j(e^(z_j)),   such that sum_k(p_k) = 1

Categorical cross-entropy, using a one-hot encoded true label y:

    L = -sum_k( y_k * log(p_k) )

Since y is one-hot (only the true class has y_k = 1, all others are 0), this reduces to L = -log(p_true_class): the loss is just the negative log of the probability the model assigned to the correct class.

The key result: just as sigmoid + binary cross-entropy simplified to dL/dZ = y_hat - y, the combination of softmax + categorical cross-entropy simplifies to the exact same structural form:

    dL/dZ_output = y_hat - y_true   (where y_hat and y_true are now vectors over all classes)

This is why engine/mnist_network.py::backward() starts with the same pattern as the toy network:

    dZ3 = (self.y_hat - y_true_onehot) / n

Everything after this point follows the identical chain-rule pattern as steps 5-7 above, just repeated for three layers (W3/b3, then W2/b2, then W1/b1) instead of two.

## 10. Numerical Gradient Check: Why and How

Hand-derived backpropagation is easy to get subtly wrong (transposed matrices, wrong axis for a sum, forgetting to divide by n) and these mistakes often do not raise any error — the code runs, but computes the wrong thing (this project hit exactly this kind of bug once; see the README's "A real bug this caught" section).

Numerical gradient checking provides an independent, from-first-principles estimate of the gradient using the definition of a derivative:

    dL/dw ≈ ( L(w + eps) - L(w - eps) ) / (2 * eps),   eps ≈ 1e-6

If the hand-derived (analytic) gradient and this numerical estimate agree closely (relative error below roughly 1e-5), that is strong evidence the analytic derivation and its code implementation are correct.

Code correspondence: engine/gradient_check.py::numerical_gradient and check_gradients, which perturb each parameter one element at a time and compare against the analytic gradients returned by backward().

In this project, all gradient checks pass with relative errors in the 1e-7 to 1e-10 range (see tests/test_network.py and tests/test_mnist_network.py).