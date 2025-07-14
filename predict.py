
import pickle
from tensorflow import keras
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('my_model.h5', 'rb') as f:
    model = keras.models.load_model(f.name)
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict(text):
    tokens=tokenizer.texts_to_sequences([text])
    tokens = pad_sequences(tokens, maxlen=14, padding='pre')
    prediction = model.predict(tokens)
    word_idxx= prediction.argmax(axis=1)[0]
    return tokenizer.index_word[word_idxx]
predict("Enter Barnardo and Francisco")  # Example usage

