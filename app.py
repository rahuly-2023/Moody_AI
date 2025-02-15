import streamlit as st
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import os
import shutil
import pandas as pd
import altair as alt
from datetime import datetime


# Set page config
st.set_page_config(page_title="Moody AI", page_icon="😃", layout="wide")

# Sidebar Styling
with st.sidebar:
    st.title("🔍 About Moody AI")
    st.write("This AI detects emotions from text and provides personalized suggestions.")
    st.markdown("---")
    st.write("💡 **How to Use**:")
    st.write("1️⃣ Enter a sentence")
    st.write("2️⃣ Click 'Predict Emotion'")
    st.write("3️⃣ View results & suggestions")

# UI Styling
st.markdown("""
    <style>
        .main-title { text-align: center; font-size: 36px; font-weight: bold; }
        .emotion-text { font-size: 24px; font-weight: bold; }
        .suggestion-box { background-color: #f4f4f4; padding: 15px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; font-size: 32px;'>"
    "🌟 Moody AI - Emotion Detector"
    "</h1><br>",
    unsafe_allow_html=True
)









# Set up a local directory for NLTK data if needed (helpful in some deployments)
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK resources (only runs once)
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("punkt_tab", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download("wordnet", download_dir=nltk_data_dir)

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

# Define sequence length and emotions
MAX_LENGTH = 100  
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOJI_MAP = {"sadness": "😢", "joy": "😊", "love": "❤️", "anger": "😡", "fear": "😨", "surprise": "😲"}
COLOR_MAP = {"sadness": "blue", "joy": "green", "love": "red", "anger": "orange", "fear": "purple", "surprise": "yellow"}


# Streamlit UI
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
    ],
    5: [
        "Take a moment to process this unexpected event.",
        "Embrace the surprises life throws at you—they make life exciting!",
        "If it's a good surprise, celebrate it! If not, take a deep breath and adjust.",
        "Write about what surprised you to reflect on your feelings.",
        "Share your surprise with someone—it might make for a great story!",
        "Unexpected moments can lead to new opportunities—stay open to them."
    ]
}

# Function to get a randomized suggestion for a given sentiment
def get_suggestion(sentiment):
    return random.choice(suggestions.get(sentiment, ["Stay positive and take care of yourself."]))


# Initialize mood log in session state
if "mood_log" not in st.session_state:
    st.session_state.mood_log = []

if st.button("🎭 Predict Emotion"):
    if user_input:
        preprocessed_input = preprocess_input(user_input)
        if preprocessed_input is not None and model is not None:
            try:
                prediction = model.predict(preprocessed_input)
                predicted_class = np.argmax(prediction, axis=1)[0]
                suggestion = get_suggestion(predicted_class) 


                emotion=EMOTIONS[predicted_class]
                emoji = EMOJI_MAP[emotion]
                
                st.markdown(f"""
                    <div style='text-align: center; font-size: 24px; font-weight: bold;'>
                        <strong>Predicted Emotion:</strong> 
                        <span style='color: {COLOR_MAP[emotion]};'>{emoji} {emotion}</span>
                    </div>
                    <p style='text-align: center; font-size: 20px;'><strong>💡 Suggestion:</strong> {suggestion}</p>
                """, unsafe_allow_html=True)

                # Log the mood entry with a timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.mood_log.append({
                    "timestamp": timestamp,
                    "input": user_input,
                    "emotion": EMOTIONS[predicted_class],
                    "suggestion": suggestion
                })
            
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Could not process the input. Please try again.")
    else:
        st.warning("Please enter a sentence before predicting.")


# Mood History Section
st.markdown("<br><br>", unsafe_allow_html=True)

st.subheader("📜 Mood History")

if st.session_state.mood_log:
    st.markdown(
        "<div style='background-color: #f8f9fa; padding: 5px; border-radius: 10px;'>"
        "<p style='font-size: 14px; text-align: center;'>Your previous mood predictions and suggestions.</p>"
        "</div><br>",
        unsafe_allow_html=True
    )
    df = pd.DataFrame(st.session_state.mood_log)

    # Style the DataFrame
    styled_df = df.style.set_properties(
        **{
            "text-align": "center",
            "background-color": "#f4f4f4",
            "border-radius": "10px",
            "padding": "5px",
        }
    ).set_table_styles(
        [
            {"selector": "th", "props": [("background-color", "#007bff"), ("color", "white"), ("font-size", "16px")]}
        ]
    )

    st.dataframe(df, use_container_width=True)

    # Add spacing before Mood Distribution
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Mood Distribution Section
    st.subheader("📊 Mood Distribution")
    st.markdown(
        "<div style='background-color: #e9ecef; padding: 5px; border-radius: 10px;'>"
        "<p style='font-size: 14px; text-align: center;'>A visual representation of your emotions over time.</p>"
        "</div><br>",
        unsafe_allow_html=True
    )

    # Create a bar chart showing the distribution of emotions
    chart_data = df.groupby("emotion").size().reset_index(name="counts")
    chart = alt.Chart(chart_data).mark_bar(cornerRadius=8).encode(
        x=alt.X("emotion:N", title="Emotion", sort="-y"),
        y=alt.Y("counts:Q", title="Count"),
        color=alt.Color("emotion:N", scale=alt.Scale(scheme="tableau20")),
        tooltip=["emotion", "counts"]
    ).properties(
        title="Mood Distribution Over Time",
        width=700,
        height=400
    ).configure_title(fontSize=18, anchor="middle")

    st.altair_chart(chart, use_container_width=True)

else:
    st.write("No mood entries yet. Predict your emotion to see history and analytics!")
