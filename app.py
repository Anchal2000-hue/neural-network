import sys
sys.path.insert(0, 'E:/pypackages')

from flask import Flask, render_template, jsonify, request
import numpy as np
from neural_network import NeuralNetwork
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

nn = None
X_train = None
y_train = None
X_test = None
y_test = None
scaler = None
current_epoch = 0

def load_digit_data():
    global X_train, y_train, X_test, y_test, scaler
    digits = load_digits()
    X = digits.data
    y = digits.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    split = int(0.8 * len(X))
    idx = np.random.permutation(len(X))
    X_train = X[idx[:split]]
    y_train = y[idx[:split]]
    X_test = X[idx[split:]]
    y_test = y[idx[split:]]

load_digit_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init', methods=['POST'])
def init_network():
    global nn, current_epoch
    data = request.json
    layers = data.get('layers', [64, 128, 64, 10])
    nn = NeuralNetwork(layers)
    current_epoch = 0
    return jsonify({'status': 'initialized', 'layers': layers})

@app.route('/train', methods=['POST'])
def train():
    global nn, current_epoch
    if nn is None:
        nn = NeuralNetwork([64, 128, 64, 10])

    data = request.json
    epochs = data.get('epochs', 10)
    lr = data.get('learning_rate', 0.01)
    batch_size = data.get('batch_size', 32)

    results = []
    for _ in range(epochs):
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]

        epoch_loss = 0
        epoch_acc = 0
        batches = 0

        for i in range(0, len(X_shuffled), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            loss, acc = nn.train_step(X_batch, y_batch, lr)
            epoch_loss += loss
            epoch_acc += acc
            batches += 1

        current_epoch += 1
        results.append({
            'epoch': current_epoch,
            'loss': epoch_loss / batches,
            'accuracy': epoch_acc / batches
        })

    test_pred = nn.predict(X_test)
    test_acc = float(np.mean(test_pred == y_test))

    return jsonify({
        'results': results,
        'test_accuracy': test_acc,
        'total_epochs': current_epoch
    })

@app.route('/predict_drawing', methods=['POST'])
def predict_drawing():
    if nn is None:
        return jsonify({'error': 'Train the network first!'})

    data = request.json
    pixels = np.array(data['pixels']).reshape(1, -1)
    pixels = scaler.transform(pixels)
    probs = nn.forward(pixels)[0]
    prediction = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return jsonify({
        'prediction': prediction,
        'confidence': round(confidence * 100, 1),
        'probabilities': probs.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)