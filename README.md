# Neural Network Digit Recognizer 🧠

A neural network built completely from scratch using pure Python and NumPy — no TensorFlow, no PyTorch, no ML frameworks!

## Features

- 🧠 Neural network built from scratch (forward + backpropagation)
- ✍️ Draw any digit (0-9) with your mouse and AI guesses it
- 📊 Real-time training visualization (loss & accuracy charts)
- ⚡ Adjustable architecture, learning rate, and batch size
- 🎯 Achieves 90%+ accuracy on digit recognition

## Tech Stack

- **Python** + **Flask** (backend)
- **NumPy** (neural network math)
- **Scikit-learn** (digits dataset)
- **Chart.js** (live training charts)
- **HTML/CSS/JavaScript** (frontend)

## How It Works

Built completely from scratch:
- **Forward propagation** — data flows through layers
- **ReLU activation** — for hidden layers
- **Softmax activation** — for output layer
- **Cross-entropy loss** — measures prediction error
- **Backpropagation** — adjusts weights to reduce error
- **Mini-batch gradient descent** — efficient training

## Run Locally

1. Clone the repo
```bash
   git clone https://github.com/Anchal2000-hue/neural-network.git
   cd neural-network
```

2. Install dependencies
```bash
   pip install flask numpy scikit-learn
```

3. Run the app
```bash
   python app.py
```

4. Open http://127.0.0.1:5000

## How to Use

1. Click **"Initialize Network"**
2. Click **"Start Training"** and watch it learn in real time
3. Wait until accuracy reaches 80%+
4. **Draw a digit** on the black canvas
5. Click **"Predict"** — see the AI guess your number!

## Architecture
- Input: 8x8 pixel digit image (64 features)
- Output: 10 classes (digits 0-9)

## Author

**Anchal** — [github.com/Anchal2000-hue](https://github.com/Anchal2000-hue)