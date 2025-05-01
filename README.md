# LSTM Text Generation

This project demonstrates how to generate text using a Long Short-Term Memory (LSTM) model implemented in PyTorch. The example provides basic data processing, model training, and testing functionalities. It uses a small snippet of text as an example dataset.

## Overview

The script includes:

1. **Data Processing**: Converts text into a format suitable for training.
2. **Model Definition**: Constructs an LSTM-based neural network for text prediction.
3. **Hyperparameter Tuning**: Searches for optimal hyperparameters for the model.
4. **Model Training**: Trains the final model using the best-found hyperparameters.
5. **Text Prediction**: Predicts the next sequence of words given a starting word.

## Prerequisites

- Python 3.6 or later
- PyTorch (1.7.0 or later recommended)
- Other Python packages: `torch`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/lstm-text-generation.git
    cd lstm-text-generation
    ```

2. Install the necessary Python packages:
    ```sh
    pip install torch
    ```

## Usage

### Data Preparation

- The text used in this script is hard-coded within the script. For different text input, you can replace the `text` variable with your content.

### Training the Model

- The script includes a hyperparameter tuning section where different configurations of the embedding sizes, hidden sizes, and learning rates are evaluated to find the best setup. 

- To train the model, simply run the script:
    ```sh
    python lstm_text_generation.py
    ```

- The script will output the training progress and best hyperparameters found.

### Predicting Text

- After training, you can generate text by specifying a starting word and the number of words to predict:
    ```python
    start_word = "elara"
    num_words = 10
    predicted_sequence = predict_sequence(start_word, num_words)
    print("Predicted sequence:", ' '.join(predicted_sequence))
    ```

## Understanding the Code

- **Data Processing**: Converts text into lowercased words and removes unnecessary characters (e.g., periods). Generates word indices.
- **LSTM Model**: Uses an embedding layer, LSTM layer, and a fully connected layer to predict the next word.
- **Training**: Uses a cross-entropy loss function and Adam optimizer.
- **Prediction**: Uses the trained model to predict a sequence of words starting from a given word.

## License

This project is licensed under the MIT License.
