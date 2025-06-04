import numpy as np
from ..utils import ensure_tensor, create_operation_node

def _backward_matmul(grad, node, parents):
    """
    Backward pass for matrix multiplication.
    
    For output Z = A @ B:
    dA = grad @ B.T
    dB = A.T @ grad
    """
    A, B = parents
    
    if A.requires_grad:
        if B.value.ndim > 1:
            grad_A = np.dot(grad, B.value.T)
        else:
            # If B is vector, use outer product
            grad_A = np.outer(grad, B.value)
        # Sum gradients properly if broadcasting occurred
        while grad_A.ndim > A.value.ndim:
            grad_A = grad_A.sum(axis=0)
        for axis, size in enumerate(A.value.shape):
            if size == 1:
                grad_A = grad_A.sum(axis=axis, keepdims=True)
        A.grad += grad_A

    if B.requires_grad:
        if A.value.ndim > 1:
            grad_B = np.dot(A.value.T, grad)
        else:
            # If A is vector, outer product
            grad_B = np.outer(A.value, grad)
        while grad_B.ndim > B.value.ndim:
            grad_B = grad_B.sum(axis=0)
        for axis, size in enumerate(B.value.shape):
            if size == 1:
                grad_B = grad_B.sum(axis=axis, keepdims=True)
        B.grad += grad_B

def matmul(a, b):
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    requires_grad = a.requires_grad or b.requires_grad
    return create_operation_node('matmul', np.dot(a.value, b.value), 
                                (a, b), _backward_matmul, requires_grad)

def _backward_transpose(grad_output, out_node, input_nodes):
    """
    Backward pass for transpose operation.
    Gradient of transpose is transpose of upstream gradient.
    """
    a = input_nodes[0]
    
    if a.requires_grad:
        a.grad += grad_output.T

def transpose(a):
    a = ensure_tensor(a)
    return create_operation_node('transpose', a.value.T, (a,), _backward_transpose, a.requires_grad)
