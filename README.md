# Sentiment Analysis with FFNN and RNN

This project implements Feed-Forward Neural Networks (FFNN) and Recurrent Neural Networks (RNN) for sentiment analysis on text data.

## Setup

1.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install torch tqdm numpy
    ```

3.  **Data**:
    Ensure the following files are in the root directory:
    *   `training.json`
    *   `validation.json`
    *   `test.json`
    *   `word_embedding.pkl`

## Implemented Features

*   **FFNN**: Implemented the forward pass using a Bag-of-Words approach with a hidden layer and LogSoftmax output.
*   **RNN**: Implemented the forward pass using a "Bag-of-Hidden-States" approach (summing outputs over time steps).
    *   Added logic to save the model based on **validation accuracy** (instead of test accuracy) to prevent overfitting.
    *   Added separate testing phase after training.
    *   Added support for loading pre-trained models.

## Usage

### Feed-Forward Neural Network (FFNN)

Train and test the FFNN model:

```bash
python3 ffnn.py \
    --hidden_dim 10 \
    --epochs 5 \
    --train_data training.json \
    --val_data validation.json \
    --test_data test.json
```

### Recurrent Neural Network (RNN)

Train and test the RNN model. This will save the best model to `rnn_model.pth` and predictions to `rnn_predictions.txt`.

```bash
python3 rnn.py \
    --hidden_dim 32 \
    --epochs 5 \
    --train_data training.json \
    --val_data validation.json \
    --test_data test.json \
    --output_file rnn_predictions.txt
```

**Arguments:**
*   `--hidden_dim`: Size of the hidden layer.
*   `--epochs`: Number of training epochs.
*   `--train_data`: Path to training JSON.
*   `--val_data`: Path to validation JSON.
*   `--test_data`: Path to test JSON.
*   `--output_file`: (RNN only) File to save predictions (default: `rnn_predictions.txt`).
*   `--load_model`: (RNN only) Path to load a saved model state dict (optional).
