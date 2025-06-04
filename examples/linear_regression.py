import numpy as np
from autograd import Node, operators, mean, matmul

class LinearRegression:
    def __init__(self, n_features, learning_rate=0.1, random_state=None):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def initialize_parameters(self):
        """Initialize weights and bias with random values"""
        if self.random_state:
            np.random.seed(self.random_state)
        self.weights = Node(np.random.randn(self.n_features), requires_grad=True)
        self.bias = Node(0.0, requires_grad=True)
    
    @staticmethod
    def mse_loss(y_pred, y_true):
        """Mean Squared Error loss function"""
        return mean((y_pred - y_true) ** 2)
    
    def fit(self, X, y, epochs=1000, verbose=True):
        """Train the model using gradient descent"""
        if self.random_state:
            np.random.seed(self.random_state)
            
        self.initialize_parameters()
        self.loss_history = []
        
        for epoch in range(1, epochs + 1):
            # Forward pass
            y_pred = matmul(X, self.weights) + self.bias
            loss = self.mse_loss(y_pred, Node(y))
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.weights.value -= self.learning_rate * self.weights.grad
            self.bias.value -= self.learning_rate * self.bias.grad
            
            # Reset gradients
            self.weights.zero_grad()
            self.bias.zero_grad()
            
            self.loss_history.append(loss.value)
            
            if verbose and epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch}, Loss: {loss.value:.4f}")
    
    def predict(self, X):
        """Make predictions using learned parameters"""
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return matmul(X, self.weights).value + self.bias.value
    
    def get_parameters(self):
        """Return learned parameters"""
        return {
            'weights': self.weights.value,
            'bias': self.bias.value
        }

def generate_data(n_samples=100, n_features=3, noise=0.1, random_state=42):
    """Generate synthetic linear regression data"""
    np.random.seed(random_state)
    X = np.random.rand(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_bias = np.random.randn()
    y = X @ true_weights + true_bias + np.random.normal(0, noise, size=n_samples)
    return X, y, true_weights, true_bias

def train_model(n_samples=100, n_features=3, learning_rate=0.1, epochs=1000, random_state=42):
    """Complete training pipeline"""
    X, y, true_weights, true_bias = generate_data(
        n_samples=n_samples, 
        n_features=n_features,
        random_state=random_state
    )
    
    model = LinearRegression(
        n_features=n_features,
        learning_rate=learning_rate,
        random_state=random_state
    )
    model.fit(X, y, epochs=epochs)
    
    print("\nLearned parameters:")
    params = model.get_parameters()
    print(f"Weights: {params['weights']}")
    print(f"Bias: {params['bias']}")
    
    print("\nTrue parameters:")
    print(f"Weights: {true_weights}")
    print(f"Bias: {true_bias}")
    
    return model