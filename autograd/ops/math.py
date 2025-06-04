import numpy as np
from ..utils import ensure_tensor, create_operation_node, EPSILON

def _backward_exp(grad_output, out_node, input_nodes):
    """
    Backward pass for the exponential function.
    Gradient: d(exp(a))/da = exp(a)
    """
    a = input_nodes[0]
    if a.requires_grad:
        a.grad += grad_output * out_node.value

def exp(a):
    a = ensure_tensor(a)
    return create_operation_node('exp', np.exp(a.value), (a,), _backward_exp, a.requires_grad)

def _backward_log(grad_output, out_node, input_nodes):
    """
    Backward pass for the natural logarithm function.
    Gradient: d(log(a))/da = 1 / a
    """
    a = input_nodes[0]
    if a.requires_grad:
        a.grad += grad_output / (a.value + EPSILON)

def log(a):
    a = ensure_tensor(a)
    return create_operation_node('log', np.log(a.value + EPSILON), (a,), _backward_log, a.requires_grad)

def _backward_pow(grad_output, out_node, input_nodes):
    """
    Backward pass for power function: base ** exponent
    Gradients:
        d(base**exponent)/dbase = exponent * base^(exponent-1)
        d(base**exponent)/dexponent = base**exponent * log(base)
    """
    base, exponent = input_nodes
    
    if base.requires_grad:
        # Handle potential broadcasting
        grad_base = grad_output * exponent.value * np.power(base.value, exponent.value - 1)
        # Sum gradients properly if broadcasting occurred
        while grad_base.ndim > base.value.ndim:
            grad_base = grad_base.sum(axis=0)
        for axis, size in enumerate(base.value.shape):
            if size == 1:
                grad_base = grad_base.sum(axis=axis, keepdims=True)
        base.grad += grad_base

    if exponent.requires_grad:
        # Gradient w.r.t exponent: base^exponent * log(base)
        # Add EPSILON inside log to avoid log(0)
        grad_exp = grad_output * out_node.value * np.log(base.value + EPSILON)
        while grad_exp.ndim > exponent.value.ndim:
            grad_exp = grad_exp.sum(axis=0)
        for axis, size in enumerate(exponent.value.shape):
            if size == 1:
                grad_exp = grad_exp.sum(axis=axis, keepdims=True)
        exponent.grad += grad_exp

def pow(base, exponent):
    base = ensure_tensor(base)
    exponent = ensure_tensor(exponent)
    requires_grad = base.requires_grad or exponent.requires_grad
    return create_operation_node('pow', np.power(base.value, exponent.value), 
                                (base, exponent), _backward_pow, requires_grad)
