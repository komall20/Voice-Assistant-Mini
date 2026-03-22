

import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pandas as pd
import datetime
import webbrowser
import random
import re
import os
import pyttsx3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Voice Assistant", layout="centered")

st.title("🧠 Voice Assistant (Mini)")
st.caption("Speech → Intent → Action → Speech")

ffmpeg_path = r"R:\2.. Entire_Data_Science_Projects\2.. DL\ffmpeg\ffmpeg\ffmpeg-8.0.1-essentials_build\bin"
os.environ["PATH"] = os.environ["PATH"] + os.pathsep + ffmpeg_path


# -------------------------------
# Load Whisper Model 
# -------------------------------

def load_whisper():
    return whisper.load_model("base")

model = load_whisper()

# -------------------------------
# Load & Train Intent Model
# -------------------------------

def load_intent_model():
    df = pd.read_csv(r"./data/voice_data.csv")

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["prompt"])

    model_intent = MLPClassifier(
        hidden_layer_sizes=(50, 25),
        max_iter=500,
        random_state=42
    )
    model_intent.fit(X, df["intent"])

    return vectorizer, model_intent

vectorizer, intent_model = load_intent_model()

# -------------------------------
# Audio Recording
sd.default.device = (14 , 12)
# -------------------------------
def record_audio(filename = r"R:\2.. Entire_Data_Science_Projects\2.. DL/input2.wav" , duration=5, fs=48000):
    st.info("🎤 Listening...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    st.success("✅ Audio Recorded")

# -------------------------------
# Speech to Text
# -------------------------------
def speech_to_text(audio_path):
    result = model.transcribe(audio_path)
    return result["text"].lower()

# -------------------------------
# Intent Prediction
# -------------------------------
def predict_intent(text):
    X_test = vectorizer.transform([text])
    return intent_model.predict(X_test)[0]

# -------------------------------
# Action Engine
# -------------------------------
def perform_action(intent, prompt=""):

    if intent == "play_music":
        webbrowser.open("https://www.youtube.com/results?search_query=music")
        return "Playing music on YouTube"

    elif intent == "open_website":
        if "youtube" in prompt:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube"
        elif "google" in prompt:
            webbrowser.open("https://www.google.com")
            return "Opening Google"
        return "Which website should I open?"

    elif intent == "news":
        webbrowser.open("https://news.google.com")
        return "Here are today's headlines"

    elif intent == "date_time":
        now = datetime.datetime.now()
        return now.strftime("It is %I:%M %p on %B %d, %Y")

    elif intent == "jokes_fun":
        jokes = [
            "Why do programmers hate nature? Too many bugs!",
            "Why did the computer go to the doctor? It caught a virus!",
            "I would tell you a UDP joke, but you might not get it."
        ]
        return random.choice(jokes)

    elif intent == "general_qa":
        webbrowser.open(f"https://www.google.com/search?q={prompt.replace(' ', '+')}")
        return "Here is what I found on Google"

    else:
        return "Sorry, I didn't understand that."

# -------------------------------
# Text to Speech
# -------------------------------
if "tts_engine" not in st.session_state:
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)
    engine.setProperty("rate", 165)
    st.session_state.tts_engine = engine

def speak(text):
    engine = st.session_state.tts_engine

    try:
        if engine._inLoop:
            engine.endLoop()
    except:
        pass

    engine.stop()
    engine.say(text)
    engine.runAndWait()
    


# -------------------------------
# UI Controls
# -------------------------------
st.markdown("### 🎙️ Talk to Assistant")

duration = st.slider("Recording Duration (seconds)", 3, 10, 5)

if st.button("🎤 Record & Run Assistant"):
    record_audio(duration=duration)

    text = speech_to_text(r"R:\2.. Entire_Data_Science_Projects\2.. DL/input2.wav")
    st.success(f"🗣 You said: **{text}**")

    intent = predict_intent(text)
    st.info(f"🧠 Detected Intent: **{intent}**")

    response = perform_action(intent, text)
    st.success(f"🤖 Assistant: {response}")

    speak(response)

st.markdown("---")
st.caption("Built with Streamlit + Whisper + ML Intent Model")
