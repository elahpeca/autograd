# from examples.linear_regression import train_model
from examples.logistic_regression import train_model

if __name__ == "__main__":
        model = train_model(
        n_samples=200,
        n_features=5,
        learning_rate=0.4,
        epochs=500,
        random_state=42
    )   