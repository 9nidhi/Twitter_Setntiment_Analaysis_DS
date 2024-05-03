import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load

# Load model and tokenizer
model_path = 'sentiment_model.h5'
tokenizer_path = 'tokenizer.joblib'

def load_model_and_tokenizer(model_path, tokenizer_path):
    model = None
    tokenizer = None
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")

    try:
        tokenizer = load(tokenizer_path)
        st.success("Tokenizer loaded successfully.")
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

# Set the maximum sequence length based on your training data
max_length = 250

# Function to predict sentiment
def predict_sentiment(text):
    # Ensure model and tokenizer are loaded
    if model is None or tokenizer is None:
        st.error("Model or tokenizer not loaded properly.")
        return None

    # Tokenize and pad the input text
    try:
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=max_length, truncating='post')
    except Exception as e:
        st.error(f"Error tokenizing or padding input text: {e}")
        return None

    # Predict the sentiment
    try:
        prediction = model.predict(padded)
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Streamlit app setup
st.title("Sentiment Analysis App")

# Text input from the user
user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input:
        # Call the prediction function
        prediction = predict_sentiment(user_input)
        
        if prediction is not None:
            # Determine the sentiment based on the prediction
            sentiment = ["Negative", "Neutral", "Positive"][prediction.argmax()]
            
            # Display the result
            st.write(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text for analysis.")
