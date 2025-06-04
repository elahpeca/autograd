import numpy as np
from ..utils import ensure_tensor, create_operation_node

def _backward_sum(grad_output, out_node, input_nodes):
    """
    Backward pass for the sum operation.

    If the sum operation reduced dimensions (axis is not None), 
    the gradient needs to be broadcast to the original shape.
    """
    a = input_nodes[0]
    if a.requires_grad:
        # Handle cases where sum reduced dimensions
        output_shape = a.value.shape
        grad = grad_output
        
        if out_node.value.shape != output_shape:
            # Reshape grad_output to allow broadcasting to a.value.shape
            # Add dimensions of size 1 where the sum operation occurred.
            axes = tuple(i for i, (x, y) in enumerate(zip(output_shape, out_node.value.shape)) if x != y)
            if axes: # if axes is not empty
                grad = np.reshape(grad, out_node.value.shape)
                for axis in axes:
                    grad = np.expand_dims(grad, axis=axis)


        a.grad += grad

def sum(a, axis=None, keepdims=False):
    a = ensure_tensor(a)
    value = np.sum(a.value, axis=axis, keepdims=keepdims)
    return create_operation_node('sum', value, (a,), _backward_sum, a.requires_grad)

def _backward_mean(grad_output, out_node, input_nodes):
    """
    Backward pass for the mean operation.

    The gradient is the upstream gradient divided by the number of elements 
    used to compute the mean. If the mean operation reduced dimensions, the
    gradient needs to be broadcast to the original shape.
    """
    a = input_nodes[0]
    if a.requires_grad:
        output_shape = a.value.shape
        grad = grad_output

        if out_node.value.shape != output_shape:
            axes = tuple(i for i, (x, y) in enumerate(zip(output_shape, out_node.value.shape)) if x != y)
            if axes:
                grad = np.reshape(grad, out_node.value.shape)
                for axis in axes:
                    grad = np.expand_dims(grad, axis=axis)
            n_elements = np.prod(a.value.shape) / np.prod(out_node.value.shape)
            a.grad += (grad / n_elements)
        else:
            n_elements = np.prod(a.value.shape) / np.prod(out_node.value.shape)
            a.grad += (grad_output / n_elements)

def mean(a, axis=None, keepdims=False):
    a = ensure_tensor(a)
    value = np.mean(a.value, axis=axis, keepdims=keepdims)
    return create_operation_node('mean', value, (a,), _backward_mean, a.requires_grad)