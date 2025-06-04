import numpy as np
from ..utils import ensure_tensor, create_operation_node

def _sum_grad_to_shape(grad, shape):
    """
    Sum the gradient to match the target shape, accounting for broadcasting.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad

def _backward_add(grad_output, out_node, input_nodes):
    a, b = input_nodes
    
    if a.requires_grad:
        a.grad += _sum_grad_to_shape(grad_output, a.value.shape)
    
    if b.requires_grad:
        b.grad += _sum_grad_to_shape(grad_output, b.value.shape)

def add(a, b):
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    requires_grad = a.requires_grad or b.requires_grad
    return create_operation_node('add', a.value + b.value, (a, b), _backward_add, requires_grad)

def _backward_sub(grad_output, out_node, input_nodes):
    a, b = input_nodes
    
    if a.requires_grad:
        a.grad += _sum_grad_to_shape(grad_output, a.value.shape)
    
    if b.requires_grad:
        b.grad -= _sum_grad_to_shape(grad_output, b.value.shape)

def subtract(a, b):
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    requires_grad = a.requires_grad or b.requires_grad
    return create_operation_node('sub', a.value - b.value, (a, b), _backward_sub, requires_grad)

def _backward_mul(grad_output, out_node, input_nodes):
    a, b = input_nodes
    
    if a.requires_grad:
        a.grad += _sum_grad_to_shape(grad_output * b.value, a.value.shape)
    
    if b.requires_grad:
        b.grad += _sum_grad_to_shape(grad_output * a.value, b.value.shape)

def multiply(a, b):
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    requires_grad = a.requires_grad or b.requires_grad
    return create_operation_node('mul', a.value * b.value, (a, b), _backward_mul, requires_grad)

def _backward_neg(grad_output, out_node, input_nodes):
    a = input_nodes[0]
    
    if a.requires_grad:
        a.grad += -grad_output

def negate(a):
    a = ensure_tensor(a)
    return create_operation_node('neg', -a.value, (a,), _backward_neg, a.requires_grad)