# Math Derivation: From Chain Rule to backward()

This document derives the backpropagation equations used in this project by hand, and shows exactly how each term in the math corresponds to a line of code in `engine/network.py` and `engine/mnist_network.py`.

---

## 1. Network Definition (Toy Network)

$$
Z^{[1]} = X W^{[1]} + b^{[1]}
$$

$$
A^{[1]} = \text{ReLU}(Z^{[1]})
$$

$$
Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}
$$

$$
\hat{y} = \sigma(Z^{[2]}), \quad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Shapes (with $n$ = number of samples):
- $X$: (n, 2), $W^{[1]}$: (2, 4), $b^{[1]}$: (4,)
- $Z^{[1]}, A^{[1]}$: (n, 4), $W^{[2]}$: (4, 1), $b^{[2]}$: (1,)
- $Z^{[2]}, \hat{y}$: (n, 1)

## 2. Loss Function

Binary cross-entropy:

$$
L = -\text{mean}\left( y \log(\hat{y}) + (1 - y)\log(1 - \hat{y}) \right)
$$

If $y = 1$ and $\hat{y} = 0.9$: $L = -\log(0.9) \approx 0.105$ (small loss, good prediction).
If $y = 1$ and $\hat{y} = 0.1$: $L = -\log(0.1) \approx 2.303$ (large loss, bad prediction).

Implemented in `engine/losses.py::binary_cross_entropy`.

## 3. Why We Need the Gradient

$$
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}
$$

where $\eta$ is the learning rate. If $\partial L/\partial w > 0$, increasing $w$ increases the loss, so we decrease $w$. If negative, we increase $w$. Implemented in `engine/network.py::update()`.

## 4. Backpropagation: Deriving dL/dZ2

We start from the loss and work backward. The first quantity we need is $\partial L / \partial Z^{[2]}$:

$$
\frac{\partial L}{\partial Z^{[2]}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial Z^{[2]}}
$$

We know:

$$
\frac{\partial L}{\partial \hat{y}} = -\left( \frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}} \right)
$$

$$
\frac{\partial \hat{y}}{\partial Z^{[2]}} = \hat{y}(1 - \hat{y}) \quad \text{(derivative of sigmoid)}
$$

Multiplying and simplifying algebraically:

$$
\frac{\partial L}{\partial Z^{[2]}} = -\left( \frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}} \right) \hat{y}(1-\hat{y})
$$

$$
= -\left( y(1-\hat{y}) - (1-y)\hat{y} \right)
$$

$$
= -\left( y - y\hat{y} - \hat{y} + y\hat{y} \right)
$$

$$
= -(y - \hat{y}) = \hat{y} - y
$$

This messy combination collapses into a remarkably simple result:

$$
\boxed{\frac{\partial L}{\partial Z^{[2]}} = \hat{y} - y}
$$

(Dividing by $n$ accounts for the mean in the loss function.)

**Code correspondence** (`engine/network.py::backward`):

```python
dZ2 = (self.y_hat - y_true) / n
```

## 5. Backpropagating to W2 and b2

Since $Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}$:

$$
\frac{\partial L}{\partial W^{[2]}} = (A^{[1]})^T \frac{\partial L}{\partial Z^{[2]}}
$$

$$
\frac{\partial L}{\partial b^{[2]}} = \sum_{\text{samples}} \frac{\partial L}{\partial Z^{[2]}}
$$

**Code correspondence:**

```python
dW2 = self.A1.T @ dZ2
db2 = np.sum(dZ2, axis=0)
```

## 6. Backpropagating Through the Hidden Layer

$$
\frac{\partial L}{\partial A^{[1]}} = \frac{\partial L}{\partial Z^{[2]}} (W^{[2]})^T
$$

**Code correspondence:**

```python
dA1 = dZ2 @ self.W2.T
```

Since $A^{[1]} = \text{ReLU}(Z^{[1]})$, and $\text{ReLU}'(z) = 1$ if $z > 0$ else $0$:

$$
\frac{\partial L}{\partial Z^{[1]}} = \frac{\partial L}{\partial A^{[1]}} \odot \text{ReLU}'(Z^{[1]})
$$

($\odot$ denotes element-wise multiplication, since ReLU is applied element-wise.)

**Code correspondence:**

```python
dZ1 = dA1 * relu_derivative(self.Z1)
```

## 7. Backpropagating to W1 and b1

Same pattern as step 5, one layer earlier:

$$
\frac{\partial L}{\partial W^{[1]}} = X^T \frac{\partial L}{\partial Z^{[1]}}, \qquad
\frac{\partial L}{\partial b^{[1]}} = \sum_{\text{samples}} \frac{\partial L}{\partial Z^{[1]}}
$$

**Code correspondence:**

```python
dW1 = self.X.T @ dZ1
db1 = np.sum(dZ1, axis=0)
```

## 8. Summary: The Full Chain

$$
\frac{\partial L}{\partial Z^{[2]}} = \hat{y} - y
\quad\to\quad
\frac{\partial L}{\partial W^{[2]}}, \frac{\partial L}{\partial b^{[2]}}
\quad\to\quad
\frac{\partial L}{\partial A^{[1]}}
\quad\to\quad
\frac{\partial L}{\partial Z^{[1]}}
\quad\to\quad
\frac{\partial L}{\partial W^{[1]}}, \frac{\partial L}{\partial b^{[1]}}
$$

This entire sequence is implemented in `engine/network.py::backward()`.

## 9. Extending to Softmax + Categorical Cross-Entropy (MNIST Network)

Softmax converts logits $z_1, \dots, z_{10}$ into a probability distribution:

$$
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad \sum_k p_k = 1
$$

Categorical cross-entropy with one-hot label $y$:

$$
L = -\sum_k y_k \log(p_k)
$$

Since $y$ is one-hot, this reduces to $L = -\log(p_{\text{true class}})$.

**Key result**: exactly as before, softmax + categorical cross-entropy simplifies to the same structural form:

$$
\frac{\partial L}{\partial Z_{\text{output}}} = \hat{y} - y
$$

(now vectors over all 10 classes). This is why `engine/mnist_network.py::backward()` starts identically:

```python
dZ3 = (self.y_hat - y_true_onehot) / n
```

Everything after follows the same chain-rule pattern as steps 5-7, repeated for three layers instead of two.

## 10. Numerical Gradient Check: Why and How

Hand-derived backprop is easy to get subtly wrong, and mistakes often do not raise any error. Numerical gradient checking provides an independent estimate using the definition of a derivative:

$$
\frac{\partial L}{\partial w} \approx \frac{L(w + \varepsilon) - L(w - \varepsilon)}{2\varepsilon}, \qquad \varepsilon \approx 10^{-6}
$$

If the analytic and numerical gradients agree closely (relative error below roughly $10^{-5}$), that is strong evidence the implementation is correct.

**Code correspondence:** `engine/gradient_check.py::numerical_gradient` and `check_gradients`.

In this project, all gradient checks pass with relative errors in the $10^{-7}$ to $10^{-10}$ range.