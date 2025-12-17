# MNIST Digit Recognition Using Deep Learning

A simple deep learning project for recognizing handwritten digits (0–9) using the MNIST dataset.

## Overview

This project trains a neural network to classify handwritten digit images from the MNIST dataset. It demonstrates the basic deep learning workflow: loading data, preprocessing, training a model, and evaluating performance.

## Dataset

* MNIST Handwritten Digits
* 60,000 training images
* 10,000 test images
* Image size: 28×28 grayscale
* Classes: digits 0–9

The dataset is automatically downloaded when the program is run.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

Run the main script:

```bash
python main.py
```

The program will:

* Load the MNIST dataset
* Train the neural network
* Evaluate accuracy on test data

## Model

* Input: 28×28 digit images
* Fully connected neural network
* ReLU activation in hidden layers
* Softmax output layer for classification

## Output

After training, the program prints:

* Training progress
* Final test accuracy

## Purpose

This project is intended for learning and understanding the fundamentals of deep learning and image classification.

## License

Educational use only.

