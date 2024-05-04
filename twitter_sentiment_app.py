import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load

# Load model and tokenizer
try:
    model = tf.keras.models.load_model('sentiment_model.h5')
    tokenizer = load('tokenizer.joblib')
    if model is None:
        st.error("Model loaded as None.")
    if tokenizer is None:
        st.error("Tokenizer loaded as None.")
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")


# Define max_length
max_length = 250  # Adjust this value based on your training data

# Define prediction function
def predict_sentiment(text):
    try:
        if tokenizer is None or model is None:
            st.error("Model or tokenizer not loaded properly.")
            return None

        # Tokenize and pad the input text
        sequences = tokenizer.texts_to_sequences([text])
        if not sequences:
            st.error("Tokenizer failed to process the input text.")
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

# When the Predict button is clicked
if st.button("Predict"):
    if user_input:
        # Make a prediction
        prediction = predict_sentiment(user_input)

        if prediction is not None:
            # Determine the sentiment
            sentiment = ["Negative", "Neutral", "Positive"][prediction.argmax()]

            # Display the sentiment
            st.write(f"Sentiment: {sentiment}")
        else:
            st.error("There was an issue with the prediction.")
    else:
        st.warning("Please enter some text for analysis.")
