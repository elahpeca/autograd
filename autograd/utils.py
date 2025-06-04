from .core.node import Node

EPSILON = 1e-10

def ensure_tensor(value):
    """
    Converts numerical types to Node tensors.
    
    Args:
        value: Input value (scalar, np.ndarray, or Node)
    
    Returns:
        Node: Tensor version of input
    """
    if not isinstance(value, Node):
        return Node(value)
    return value

def create_operation_node(op_type, value, input_nodes, backward_fn, requires_grad):
    """
    Factory function for operation nodes.
    
    Args:
        op_type: Operation type identifier
        value: Computed output value
        input_nodes: Tuple of parent nodes
        backward_fn: Gradient computation function
        requires_grad: Whether gradient is needed
    
    Returns:
        Node: Configured operation node
    """
    node = Node(value, requires_grad=requires_grad)
    node._operation_type = op_type
    node._input_nodes = input_nodes
    node._backward_fn = backward_fn
    return node