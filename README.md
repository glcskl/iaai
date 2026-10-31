# iaai — neural networks from scratch

A set of small programs that build a neural network by hand instead of calling `fit()`. Forward pass, backpropagation, gradient descent and the layer maths are all written out in NumPy, so every step is visible and nothing is hidden behind a framework. The resulting networks are then applied to real tasks: recognising handwritten letters, counting fingers on a webcam and recognising air-drawn shapes.

Built as a personal project on 18 September 2026.

## Features

- Neural network implemented from the ground up, forward and backward
- Training loop with gradient descent written explicitly, no framework calls
- XOR solved from scratch, with the loss curve plotted as proof of convergence
- Handwritten letter classifier for A to Z, trained on EMNIST, weights saved to disk
- Finger counting from a webcam feed
- Air writing: draw a letter in the air, the network recognises it
- No heavy dependencies, everything readable in a single file per demo

## Tech stack

| Layer | Technology |
| --- | --- |
| Language | Python 3 |
| Numerics | NumPy |
| Computer vision | OpenCV |
| Hand tracking | MediaPipe, hand landmarker model |
| Data source | EMNIST |
| Visualisation | Matplotlib |

## Getting started

### Requirements

- Python 3.9 or newer
- A webcam, required only for the two vision demos

### Environment variables

None.

### Installation

```bash
git clone https://github.com/glcskl/iaai.git
cd iaai
python -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python mediapipe matplotlib
```

Download the EMNIST dataset for the letter classifier:

```bash
sh get_data.sh
```

### Running

Train on XOR and plot the loss curve:

```bash
python train_xor.py
```

Classify handwritten letters against the saved weights:

```bash
python letters_net.py
```

Count fingers using the webcam:

```bash
python finger_counter.py
```

Draw a letter in the air and have it recognised:

```bash
python air_write.py
```

## Project structure

```
network.py       network core: layers, forward pass, backpropagation
train_xor.py     XOR training demo, writes xor_training.png
letters_net.py   A to Z classifier built on network.py
air_write.py     air drawing with both hands, plus recognition
finger_counter.py finger counting through the webcam
get_data.sh      fetches the EMNIST dataset
models/          MediaPipe hand landmarker model
weights/         saved network weights
tests/           pytest suite for the network and the letter classifier
xor_training.png training loss curve
```

## Tests

```bash
pip install numpy pytest
python -m pytest -q
```

`tests/test_network.py` compares the analytic gradients with central differences, trains XOR to convergence and checks the backpropagation shapes. `tests/test_letters.py` covers the softmax stability and the weight file layout. `.github/workflows/ci.yml` runs ruff and the suite on every push.

## How it works

`network.py` implements dense layers, the activation function, the loss and the derivative chain. `train_xor.py` trains on the four XOR cases and plots the loss, which falls close to zero. The vision demos reuse the same training code: the image or landmark data is flattened into a vector, and the trained weights are loaded from `weights/`.

MediaPipe is used only to locate hand landmarks. The classification itself runs on the network written here.

## Notes

This project is educational. The networks are small, trained on modest data and are not competitive with current libraries for accuracy or speed. The point is the visible derivation, not the result.