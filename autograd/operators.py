from .core.node import Node
from .utils import ensure_tensor, create_operation_node
from .ops.base import _backward_add, _backward_sub, _backward_mul, _backward_neg
from .ops.math import _backward_pow
from .ops.matrix import _backward_matmul
import numpy as np

def _add_operator_methods():
    """"Adds operator methods to the Node class."""
    
    def __add__(self, other):
        other = ensure_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return create_operation_node('add', self.value + other.value, 
                                    (self, other), _backward_add, requires_grad)

    Node.__add__ = __add__
    Node.__radd__ = __add__

    def __sub__(self, other):
        other = ensure_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return create_operation_node('sub', self.value - other.value,
                                    (self, other), _backward_sub, requires_grad)

    Node.__sub__ = __sub__

    def __rsub__(self, other):
        other = ensure_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return create_operation_node('sub', other.value - self.value,
                                    (other, self), _backward_sub, requires_grad)

    Node.__rsub__ = __rsub__

    def __mul__(self, other):
        other = ensure_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return create_operation_node('mul', self.value * other.value,
                                    (self, other), _backward_mul, requires_grad)

    Node.__mul__ = __mul__
    Node.__rmul__ = __mul__

    def __neg__(self):
        return create_operation_node('neg', -self.value, (self,), 
                                    _backward_neg, self.requires_grad)

    Node.__neg__ = __neg__

    def __pow__(self, other):
        other = ensure_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return create_operation_node('pow', self.value ** other.value,
                                    (self, other), _backward_pow, requires_grad)

    Node.__pow__ = __pow__

    def __matmul__(self, other):
        other = ensure_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return create_operation_node('matmul', np.matmul(self.value, other.value),
                                    (self, other), _backward_matmul, requires_grad)

    Node.__matmul__ = __matmul__

_add_operator_methods()