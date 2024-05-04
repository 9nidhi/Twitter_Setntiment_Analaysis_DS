import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load

# Global variables for model and tokenizer
model = None
tokenizer = None

# Define max_length based on your training data
max_length = 250

# Function to load the model and tokenizer
def load_model_and_tokenizer():
    global model, tokenizer
    try:
        # Load the model and tokenizer
        model = tf.keras.models.load_model('sentiment_model.h5')
        tokenizer = load('tokenizer.joblib')
        
        # Check if loaded successfully
        if model is None or tokenizer is None:
            st.error("Model or tokenizer loaded as None.")
            return False
        
        return True
        
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return False

# Call the function to load model and tokenizer
load_success = load_model_and_tokenizer()

# Function to predict sentiment
def predict_sentiment(text):
    # Check if model and tokenizer loaded successfully
    if not load_success:
        st.error("Model or tokenizer not loaded properly.")
        return None
    
    try:
        # Tokenize and pad the input text
        sequences = tokenizer.texts_to_sequences([text])
        if not sequences:
            st.error("Failed to tokenize the input text.")
            return None
        
        padded = pad_sequences(sequences, maxlen=max_length, truncating='post')
        
        # Predict the sentiment
        prediction = model.predict(padded)
        return prediction
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Set up Streamlit app
st.title("Sentiment Analysis App")

# Get user input
user_input = st.text_area("Enter your text:")

# Predict button
if st.button("Predict"):
    if user_input:
        # Make a prediction
        prediction = predict_sentiment(user_input)
        if prediction is not None:
            # Determine the sentiment from the prediction
            sentiment_labels = ["Negative", "Neutral", "Positive"]
            sentiment = sentiment_labels[prediction.argmax()]

            # Display the sentiment
            st.write(f"Sentiment: {sentiment}")
        else:
            st.error("There was an issue with the prediction.")
    else:
        st.warning("Please enter some text for analysis.")
