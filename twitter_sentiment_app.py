import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load

# Define file paths for the model and tokenizer
model_path = 'sentiment_model.h5'
tokenizer_path = 'tokenizer.joblib'

# Function to load model and tokenizer
def load_model_and_tokenizer(model_path, tokenizer_path):
    # Initialize model and tokenizer variables
    model = None
    tokenizer = None

    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Print debugging information
        print(f"Error loading model from {model_path}: {e}")
        return model, tokenizer

    # Load the tokenizer
    try:
        tokenizer = load(tokenizer_path)
        st.success("Tokenizer loaded successfully.")
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        # Print debugging information
        print(f"Error loading tokenizer from {tokenizer_path}: {e}")
        return model, tokenizer

    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

# Define the maximum sequence length based on your training data
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
        # Print debugging information
        print(f"Error tokenizing or padding input text: {e}")
        return None

    # Predict the sentiment
    try:
        prediction = model.predict(padded)
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        # Print debugging information
        print(f"Error making prediction: {e}")
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
