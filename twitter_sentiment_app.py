import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
# from google.colab import drive
# drive.mount('/content/drive')

try:
    model = tf.keras.models.load_model('sentiment_model.h5')
    tokenizer = load('tokenizer.joblib')
except Exception as e:
    st.error(f"Error loading model: {e}")

max_length = 250  # Adjust this value based on your training data

# Create a function to predict sentiment
def predict_sentiment(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, truncating='post')

    # Predict the sentiment
    prediction = model.predict(padded)
    print(prediction)
    return prediction

# Set up Streamlit app
st.title("Sentiment Analysis App")

# Text input for user
user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input:
        # Call the prediction function
        prediction = predict_sentiment(user_input)

        # Determine the sentiment based on the prediction
        sentiment = ["Negative", "Neutral", "Positive"][prediction.argmax()]

        # Display the result
        st.write(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text for analysis.")