import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load

# Load the model and tokenizer
try:
    model = tf.keras.models.load_model('sentiment_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Set model to None if there's an error

try:
    tokenizer = load('tokenizer.joblib')
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    tokenizer = None  # Set tokenizer to None if there's an error

# Set the maximum sequence length based on your training data
max_length = 250

# Create a function to predict sentiment
def predict_sentiment(text):
    # Check if model and tokenizer are loaded
    if model is None or tokenizer is None:
        st.error("Model or tokenizer not loaded properly.")
        return None

    # Check if the input text is empty
    if not text:
        st.error("Input text is empty. Please provide valid input.")
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

# Set up Streamlit app
st.title("Sentiment Analysis App")

# Text input for user
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
