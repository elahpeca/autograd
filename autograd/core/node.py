import numpy as np

class Node:
    """
    Fundamental building block of the computational graph.
    Represents a node with value and gradient capabilities.
    """
    
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float64)
        self._value = data
        self._grad = np.zeros_like(data, dtype=np.float64) if requires_grad else None
        self.requires_grad = requires_grad
        self._operation_type = None
        self._input_nodes = ()
        self._backward_fn = None

    @property
    def value(self):
        """Returns the node's value."""
        return self._value
    
    @value.setter
    def value(self, new_value):
        """Sets the node's value with type conversion."""
        self._value = np.asarray(new_value, dtype=np.float64)

    @property
    def grad(self):
        """Returns the node's gradient."""
        return self._grad

    @grad.setter
    def grad(self, value):
        """Sets the node's gradient with type checking."""
        if self._grad is None and self.requires_grad:
            self._grad = np.zeros_like(self._value, dtype=np.float64)
        if self._grad is not None:
            self._grad = np.asarray(value, dtype=np.float64) 

    def zero_grad(self):
        """Resets the gradient to zero."""
        if self._grad is not None:
            self._grad.fill(0.0)

    def backward(self, grad_output=None):
        """
        Performs backward propagation through the computational graph.
        
        Args:
            grad_output: Optional upstream gradient. Defaults to 1.0 for scalar outputs.
        """
        if not self.requires_grad:
            return
            
        grad_output = np.ones_like(self._value) if grad_output is None else grad_output
        if self._grad is None:
            self._grad = grad_output.copy()
        else:
            self._grad += grad_output
        
        # Topological sort (DFS)
        visited = set()
        order = []
        def build(v):
            if v not in visited:
                visited.add(v)
                for parent in v._input_nodes:
                    build(parent)
                order.append(v)
        build(self)
        
        # Backward pass
        for node in reversed(order):
            if node._backward_fn is not None and node._grad is not None:
                node._backward_fn(node._grad, node, node._input_nodes)

    def __repr__(self):
        """String representation of the node."""
        return f"Node(value={self.value}, grad={'None' if self._grad is None else '[...]'}, requires_grad={self.requires_grad})"