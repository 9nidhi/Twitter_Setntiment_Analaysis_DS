import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load

# Set up Streamlit app
st.title("Sentiment Analysis App")

# Define max_length based on your training data
max_length = 250

# Load the pre-trained model and tokenizer
try:
    model_path = 'sentiment_model.h5'
    tokenizer_path = 'tokenizer.joblib'
    model = tf.keras.models.load_model(model_path)
    tokenizer = load(tokenizer_path)
    
    if model is None:
        st.error("Model loaded as None.")
    if tokenizer is None:
        st.error("Tokenizer loaded as None.")
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")

# Function to predict sentiment
def predict_sentiment(text):
    try:
        if tokenizer is None or model is None:
            st.error("Model or tokenizer not loaded properly.")
            return None
        
        # Tokenize the input text
        sequences = tokenizer.texts_to_sequences([text])
        if not sequences:
            st.error("Failed to tokenize the input text.")
            return None
        
        # Pad the sequences
        padded = pad_sequences(sequences, maxlen=max_length, truncating='post')
        
        # Predict the sentiment
        prediction = model.predict(padded)
        
        # Return the prediction
        return prediction
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

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
