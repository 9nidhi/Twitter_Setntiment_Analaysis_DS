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
# Load Font Awesome CSS to use the Twitter icon
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """,
    unsafe_allow_html=True
)

# Add the title with a Twitter icon and space between the text and icon
st.markdown(
    """
    <h1 style="display: flex; align-items: center;">
        Sentiment Analysis App
        <i class="fab fa-twitter" style="margin-left: 10px;"></i>
    </h1>
    """,
    unsafe_allow_html=True
)


# Add background image using HTML and CSS
background_image_url = "https://img.freepik.com/free-photo/social-media-background-twitte_135149-69.jpg?t=st=1715150409~exp=1715154009~hmac=d6fa0ebe02d7670cfa2cf5fb940dba08eb6842469272795f3171150e3d612d98&w=900"  # Replace with your image URL

# Add CSS styles to set the background image
st.markdown(
    f"""
    <style>
        .st-emotion-cache-13k62yr{{
            background-image: url('{background_image_url}');
            background-size: cover;
            background-repeat: no-repeat;
               
        }}
        .overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);  /* Semi-transparent black overlay */
            display: flex;
            justify-content: center;
            align-items: center;
            color: white;
            font-size: 24px;
        }}
.st-b1 {{
    background-color: rgb(236 237 242);
    /* background-color: rgba(233, 236, 237, 0); */
    box-shadow: rgba(0, 0, 0, 0.1) 5px 2px 10px;
}}  
.st-bv {{
    caret-color: rgb(14, 17, 23);
}}
.st-emotion-cache-1om1ktf div {{
     border-width: none;
}}

.st-emotion-cache-sh2krr p {{
    word-break: break-word;
    margin-bottom: 0px;
    font-size: 27px;
}}

.st-emotion-cache-19rxjzo {{
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 38.4px;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    user-select: none;
    background-color: rgb(10 38 93);
    border: 1px solid rgba(250, 250, 250, 0.2);
}} 
.st-bb {{
    color: rgb(10 38 93);
    font-size:25px;
}}
   
        .st-emotion-cache-1avcm0n {{
    position: fixed;
    top: 0px;
    left: 0px;
    right: 0px;
    height: 2.875rem;
    background: rgb(10 38 93);
    outline: none;
    z-index: 999990;
    display: block;
}}

.st-emotion-cache-cnbvxy p {{
    word-break: break-word;
    color: white;
    font-size: 30px;
    float: right;
      padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
     background: rgb(10 38 93);
}}
    </style>
    """,
    unsafe_allow_html=True,
)


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
