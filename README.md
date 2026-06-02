# Next Word Prediction using LSTM

A deep learning-based Natural Language Processing (NLP) project that predicts the next word in a sentence using an LSTM (Long Short-Term Memory) neural network.

---

## Project Overview

This project builds a language model that learns patterns from text data and predicts the most probable next word based on a given input sequence.

It works similar to smartphone keyboard suggestions.

### Example:
input: i wonder
Output: i wonder if i’ve been


---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- LSTM (Deep Learning)
- Keras Tokenizer
- Padding Sequences

---

## How It Works

1. Text dataset is collected and preprocessed
2. Tokenizer converts words into numerical sequences
3. Input sequences are generated for training
4. Sequences are padded to fixed length
5. LSTM model learns patterns in sequences
6. Model predicts probability distribution over vocabulary
7. Word with highest probability is selected using `argmax`

---

## Model Architecture

- Embedding Layer (word representation)
- LSTM Layer (sequence learning)
- Dense Layer with Softmax activation (output probabilities)

---

## Prediction Process

The model generates text word by word using a loop:

```python
input_text = "i wonder"
predicting_word_count = 3

for _ in range(predicting_word_count):

    token_list = tokenizer.texts_to_sequences([input_text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding='pre'
    )

    predicted = np.argmax(model.predict(token_list), axis=1)[0]

    output_word = tokenizer.index_word.get(predicted, "")

    input_text += " " + output_word

print(input_text)
