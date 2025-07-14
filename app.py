# app.py
import streamlit as st
from predict import predict

st.title("Next Word Predictor 🔮")
text_input = st.text_input("Enter a sentence:")

if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        next_word = predict(text_input)
        st.success(f"**Predicted next word:** `{next_word}`")
