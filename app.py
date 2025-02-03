import streamlit as st
import numpy as np
import re
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random


# Download necessary NLTK resources (only runs once)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Caching function to load the tokenizer
@st.cache_resource
def load_tokenizer():
    try:
        with open("tokenizer.pkl", "rb") as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded successfully!")
        return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

# Caching function to load the model
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model("emotion.h5")
        print("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load tokenizer and model once
tokenizer = load_tokenizer()
model = load_emotion_model()

# Define sequence length
MAX_LENGTH = 100  
EMOTIONS = ["sadness", "joy", "love", "anger", "fear"]

# Streamlit UI
st.title("Moody AI")
user_input = st.text_input("Enter a sentence for emotion detection:")

def preprocess_input(text):
    """Cleans and tokenizes user input."""
    try:
        # Text cleaning
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        text = text.lower()

        # Tokenization & Lemmatization
        words = word_tokenize(text)
        cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        cleaned_text = ' '.join(cleaned_words)

        # Convert text to sequence and pad
        text_seq = tokenizer.texts_to_sequences([cleaned_text])
        text_padded = pad_sequences(text_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

        return text_padded
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None








# Suggestions pool
suggestions = {
    0: [  # Sadness
        "Give yourself time to feel and process your emotions.",
        "Listening to uplifting music or watching a comforting movie might help.",
        "Reach out to someone who can offer a listening ear.",
        "Remember that sadness is a natural part of the healing process.",
        "Consider practicing self-compassion and treating yourself with kindness.",
        "Do something comforting, like a warm bath or watching your favorite show."
    ],
    1: [  # Joy
        "Share your happiness with a friend or loved one.",
        "Capture this moment in a journal or photo to cherish later.",
        "Spread your joy by doing something kind for someone else.",
        "Celebrate your accomplishments, no matter how small they may seem.",
        "Enjoy the moment and be proud of what you’ve achieved.",
        "Take time to reflect on your positive experiences and embrace gratitude."
    ],
    2: [  # Love
        "Express your love and appreciation to those who matter most.",
        "Take a moment to reflect on the love in your life and be grateful.",
        "Do something kind for someone you care about.",
        "Love yourself as much as you love others—self-care is important.",
        "Cherish and nurture your relationships to keep them strong.",
        "Remind yourself that love grows when shared freely."
    ],
    3: [  # Anger
        "Take a deep breath and count to ten.",
        "Try journaling to process your thoughts and emotions.",
        "Physical activity, like a walk or run, might help release pent-up frustration.",
        "Try deep breathing exercises to calm down.",
        "Find a quiet place to reflect and gather your thoughts before reacting.",
        "Talk it out with someone you trust to release your feelings."
    ],
    4: [  # Fear
        "Focus on what you can control and take small steps forward.",
        "Consider grounding techniques like the 5-4-3-2-1 method.",
        "Talking to someone you trust can help you feel supported.",
        "Remind yourself that fear is often just an illusion.",
        "Focus on positive outcomes rather than imagining worst-case scenarios.",
        "Practice mindfulness and stay in the present moment."
    ]
}

# Function to get a randomized suggestion for a given sentiment
def get_suggestion(sentiment):
    return random.choice(suggestions.get(sentiment, ["Stay positive and take care of yourself."]))













if st.button("Predict Emotion"):
    if user_input:
        preprocessed_input = preprocess_input(user_input)
        if preprocessed_input is not None and model is not None:
            try:
                prediction = model.predict(preprocessed_input)
                predicted_class = np.argmax(prediction, axis=1)[0]
                suggestion = get_suggestion(predicted_class)  
                st.write(f"**Predicted Emotion:** {EMOTIONS[predicted_class]}")
                st.write(f"**Suggestion:** {suggestion}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Could not process the input. Please try again.")
    else:
        st.warning("Please enter a sentence before predicting.")
