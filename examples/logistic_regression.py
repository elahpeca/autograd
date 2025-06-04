import numpy as np
from autograd import Node, operators, exp, log, mean, matmul

class LogisticRegression:
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
    def sigmoid(x):
        """Sigmoid activation function"""
        return (1 + exp(-x)) ** (-1)
    
    @staticmethod
    def binary_cross_entropy(y_pred, y_true):
        """Binary cross entropy loss"""
        return mean(-(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)))
    
    def fit(self, X, y, epochs=1000, verbose=True):
        """Train the model using gradient descent"""
        if self.random_state:
            np.random.seed(self.random_state)
            
        self.initialize_parameters()
        self.loss_history = []
        
        for epoch in range(1, epochs + 1):
            # Forward pass
            logits = matmul(X, self.weights) + self.bias
            y_pred = self.sigmoid(logits)
            loss = self.binary_cross_entropy(y_pred, Node(y))
            
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
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        logits = matmul(X, self.weights).value + self.bias.value
        return 1 / (1 + np.exp(-logits))
    
    def predict(self, X, threshold=0.5):
        """Make class predictions"""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def get_parameters(self):
        """Return learned parameters"""
        return {
            'weights': self.weights.value,
            'bias': self.bias.value
        }

def generate_classification_data(n_samples=100, n_features=3, random_state=42):
    """Generate synthetic binary classification data"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_bias = np.random.randn()
    logits = X @ true_weights + true_bias
    y = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
    return X, y, true_weights, true_bias

def train_model(n_samples=100, n_features=3, learning_rate=0.1, epochs=1000, random_state=42):
    """Complete training pipeline for logistic regression"""
    X, y, true_weights, true_bias = generate_classification_data(
        n_samples=n_samples, 
        n_features=n_features,
        random_state=random_state
    )
    
    model = LogisticRegression(
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
    
    # Calculate accuracy
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nTraining accuracy: {accuracy:.2%}")
    
    return model