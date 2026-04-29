import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.accuracy_history = []
        
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.01
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.relu(z))
        
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(self.softmax(z))
        
        return self.activations[-1]

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true.astype(int)] + 1e-8)
        return np.mean(log_likelihood)

    def backward(self, X, y_true, learning_rate):
        m = X.shape[0]
        delta = self.activations[-1].copy()
        delta[range(m), y_true.astype(int)] -= 1
        delta /= m

        for i in range(len(self.weights) - 1, -1, -1):
            dw = self.activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            
            if i > 0:
                delta = delta @ self.weights[i].T * self.relu_derivative(self.z_values[i-1])
            
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

    def train_step(self, X, y, learning_rate=0.01):
        y_pred = self.forward(X)
        loss = self.compute_loss(y_pred, y)
        self.backward(X, y, learning_rate)
        
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y)
        
        self.loss_history.append(float(loss))
        self.accuracy_history.append(float(accuracy))
        
        return float(loss), float(accuracy)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)