# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Main Application Logic ---

def main():
    """
    Main function to run the data preparation, model training, and evaluation.
    """
    # 1. Load and preprocess the dataset
    X_train, X_test, y_train, y_test = get_prepared_iris_data()

    # 2. Set up network hyperparameters
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    hidden_nodes = 10
    epochs = 1000
    learning_rate = 0.1

    # 3. Initialize and train the neural network
    print("Building and training the model...")
    model = SimpleNeuralNetwork(
        input_dim=input_size, 
        hidden_dim=hidden_nodes, 
        output_dim=output_size
    )
    model.fit(X_train, y_train, epochs=epochs, learning_rate=learning_rate)
    print("✅ Training complete.")

    # 4. Evaluate the model's performance
    accuracy = model.evaluate(X_test, y_test)
    print(f"\nFinal Model Accuracy: {accuracy:.2%}")

# --- Data Preparation ---

def get_prepared_iris_data():
    """
    Loads the Iris dataset, scales features, and one-hot encodes labels.

    Returns:
        A tuple containing split data: (X_train, X_test, y_train, y_test)
    """
    # Load raw data
    features, labels = load_iris(return_X_y=True)
    labels = labels.reshape(-1, 1) # Reshape for the encoder

    # Standardize features (mean=0, variance=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # One-hot encode target labels
    encoder = OneHotEncoder(sparse_output=False)
    labels_encoded = encoder.fit_transform(labels)

    # Return the split dataset
    return train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)

# --- Neural Network Class ---

class SimpleNeuralNetwork:
    """
    A simple feedforward neural network with one hidden layer.
    
    This class encapsulates the weights, biases, and the core logic for
    training and prediction.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initializes the network's weights and biases."""
        # Use a seed for reproducible results
        np.random.seed(42)
        # Initialize weights with small random values and biases to zero
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    def _sigmoid(self, z):
        """The sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def _sigmoid_prime(self, a):
        """The derivative of the sigmoid function."""
        return a * (1 - a)

    def _forward_prop(self, X):
        """Performs a forward pass and caches intermediate values."""
        z1 = X @ self.W1 + self.b1
        a1 = self._sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._sigmoid(z2)
        # Cache values needed for backpropagation
        cache = {'a1': a1, 'a2': a2}
        return a2, cache

    def _back_prop(self, X, y, cache):
        """Performs backpropagation to calculate gradients."""
        m = X.shape[0] # Number of training examples
        a1, a2 = cache['a1'], cache['a2']

        # Calculate error derivatives for the output layer
        dZ2 = (a2 - y) * self._sigmoid_prime(a2)
        dW2 = (1/m) * a1.T @ dZ2
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # Calculate error derivatives for the hidden layer
        dZ1 = (dZ2 @ self.W2.T) * self._sigmoid_prime(a1)
        dW1 = (1/m) * X.T @ dZ1
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def fit(self, X, y, epochs, learning_rate):
        """Trains the network using gradient descent."""
        for epoch in range(epochs):
            # Perform a full forward and backward pass
            y_pred, cache = self._forward_prop(X)
            grads = self._back_prop(X, y, cache)

            # Update all weights and biases
            self.W1 -= learning_rate * grads['dW1']
            self.b1 -= learning_rate * grads['db1']
            self.W2 -= learning_rate * grads['dW2']
            self.b2 -= learning_rate * grads['db2']
            
            # Print the loss periodically to monitor training
            if epoch % 100 == 0:
                loss = np.mean((y - y_pred) ** 2)
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict(self, X):
        """Makes predictions for a given input."""
        probabilities, _ = self._forward_prop(X)
        return np.argmax(probabilities, axis=1)

    def evaluate(self, X_test, y_test):
        """Calculates the accuracy of the model on a test set."""
        predictions = self.predict(X_test)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

# --- Entry point of the script ---
if __name__ == "__main__":
    main()