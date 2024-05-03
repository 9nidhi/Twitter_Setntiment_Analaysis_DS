import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load

# Load the model and tokenizer
try:
    model = tf.keras.models.load_model('sentiment_model.h5')
    tokenizer = load('tokenizer.joblib')
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# Set the maximum sequence length based on your training data
max_length = 250

# Create a function to predict sentiment
def predict_sentiment(text):
    # Ensure the input text is not empty
    if not text:
        st.error("Input text is empty. Please provide valid input.")
        return None

    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    if not sequences or len(sequences[0]) == 0:
        st.error("Input text could not be tokenized properly.")
        return None
    
    padded = pad_sequences(sequences, maxlen=max_length, truncating='post')

    # Predict the sentiment
    prediction = model.predict(padded)
    
    return prediction

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
