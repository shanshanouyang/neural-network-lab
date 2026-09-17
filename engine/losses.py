import numpy as np

def binary_cross_entropy(y_true, y_pred, eps=1e-8):
    y_true = y_true.reshape(y_pred.shape)   # 新增这一行，强制对齐形状
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))