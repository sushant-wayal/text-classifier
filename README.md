# Text Classifier 🎬

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)


This repository contains a simple text classification model built using PyTorch and TorchText.  It uses the IMDB movie review dataset to train a recurrent neural network (LSTM) to classify movie reviews as positive or negative.

## Table of Contents

- [Description](#description)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)


## Description

This project demonstrates a basic text classification pipeline using PyTorch. It covers:

* Data loading and preprocessing with TorchText's IMDB dataset and `basic_english` tokenizer.
* Vocabulary creation.
* Building a custom `Dataset` and `DataLoader`.
* Implementing an LSTM-based classifier.
* Training the model.
* Making predictions.

## Workflow

The system follows these steps:

1. **Data Preparation:** The IMDB dataset is loaded and tokenized using the `basic_english` tokenizer. A vocabulary is built from the training data.
2. **Dataset and DataLoader:** A custom `TextDataset` class is created to handle data loading and preprocessing.  A `DataLoader` is used for efficient batching and shuffling.
3. **Model Building:** An LSTM-based classifier is implemented.  The model consists of an embedding layer, an LSTM layer, a fully connected layer, and a sigmoid activation function.
4. **Training:** The model is trained using the Adam optimizer and binary cross-entropy loss.
5. **Prediction:** The trained model predicts the sentiment of input text.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sushant_wayal/text-classifier.git 
   cd text-classifier
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install torch torchtext
   ```


## Usage

To train the model, simply run the `Text Classifier.py` script:

```bash
python Text_Classifier.py
```

The script will train the model for 10 epochs and print the loss at the end of each epoch.  

To use the trained model for prediction, you can use the `predict` function (see example in `Text_Classifier.py`):

```python
print(predict('This movie is great!')) # Expected output close to 1
print(predict('This movie is terrible!')) # Expected output close to 0
```

This will output a probability score between 0 and 1, where values closer to 1 indicate a positive sentiment and values closer to 0 indicate a negative sentiment. 👍 👎
