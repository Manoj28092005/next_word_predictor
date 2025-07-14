
# Next Word Predictor

This repository provides a machine learning model for predicting the next word in a given sequence of text. The project demonstrates the use of Natural Language Processing (NLP) and deep learning techniques, leveraging TensorFlow and Keras for model development.

## Features

- **Text Tokenization and Padding:**  
  Utilizes TensorFlow's tokenizer to convert text into sequences of integers, followed by padding to standardize input lengths for the neural network.

- **Deep Learning Model:**  
  The core model is built using Keras' Sequential API with the following architecture:
  ```python
  model = Sequential()
  model.add(Embedding(total_tokens+1, 100, input_length=max_len-1))
  model.add(LSTM(150, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(100))
  model.add(Dense(total_tokens+1, activation='softmax'))
  ```
  - **Embedding Layer:** Converts input tokens into dense vectors of fixed size (100).
  - **LSTM Layers:** Two stacked LSTM layers (150 and 100 units) to capture temporal dependencies in sequences.
  - **Dropout Layer:** Prevents overfitting by randomly setting input units to 0 with a frequency of 20% during training.
  - **Dense Output Layer:** Uses softmax activation to output probabilities over the vocabulary for next word prediction.

- **Easy Experimentation:**  
  Jupyter Notebooks are used for a step-by-step approach, making it easy to visualize data processing, training, and predictions.

## Getting Started

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/Manoj28092005/next_word_predictor.git
   cd next_word_predictor
   ```

2. **Install Dependencies:**
   - Python 3.x
   - TensorFlow
   - Keras
   - Numpy
   - Pandas
   - Jupyter Notebook

   Install with pip:
   ```sh
   pip install tensorflow keras numpy pandas jupyter
   ```

3. **Run the Notebook:**
   Open the provided Jupyter Notebook and follow the cells to preprocess data, train the model, and predict the next word.

## Usage

- Input a sequence of words in the notebook.
- The trained model will predict the most probable next word based on the learned data.

## Applications

- Text autocompletion
- Language modeling and experimentation
- Educational purposes for learning NLP and deep learning with TensorFlow



---

Feel free to contribute or modify the model for your own datasets and use cases!
