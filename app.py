from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deep_translator import GoogleTranslator
import joblib
import pandas as pd
import os
import re
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load ML Models
# -------------------------

# -------------------------
# Load ML Models
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

toxic_model = joblib.load(os.path.join(BASE_DIR, "toxic_model.pkl"))
emotion_model = joblib.load(os.path.join(BASE_DIR, "emotion_model.pkl"))
multilingual_model = joblib.load(os.path.join(BASE_DIR, "multilingual_model.pkl"))
behaviour_model = joblib.load(os.path.join(BASE_DIR, "behaviour_model.pkl"))

rewrite_model = joblib.load(os.path.join(BASE_DIR, "rewrite_model.pkl"))
coach_model = joblib.load(os.path.join(BASE_DIR, "coach_model.pkl"))

# -------------------------
# Load Dataset
# -------------------------


coach_dataset = pd.read_csv(
    os.path.join(BASE_DIR, "datasets", "coach_dataset.csv")
)

# -------------------------
# History Storage
# -------------------------

history = []

# -------------------------
# Request Models
# -------------------------

class Comment(BaseModel):
    text: str

class TranslateRequest(BaseModel):
    text: str
    lang: str


# -------------------------
# Text Cleaning
# -------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------------------------
# Translation
# -------------------------

def translate_to_english(text):
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except:
        return text


# -------------------------
# TOXIC ANALYSIS
# -------------------------

@app.post("/analyze")
def analyze(comment: Comment):
    

    original = comment.text
    translated = translate_to_english(original)
    cleaned = clean_text(translated)

    # 🔥 Root words fix
    positive_words = [
        "love", "like", "beautiful", "nice", "good",
        "great", "awesome", "amazing", "wonderful"
    ]

    toxic_words = [
        "hate", "stupid", "idiot", "shut up",
        "useless", "fool", "dumb"
    ]

    # 🔥 Rule-based override
    if any(word in cleaned for word in positive_words):
        label = "non_toxic"

    elif any(word in cleaned for word in toxic_words):
        label = "toxic"

    else:
        prediction = toxic_model.predict([cleaned])[0]
        label = "toxic" if str(prediction).lower() in ["1", "toxic"] else "non_toxic"

    # ✅ history untouched
    history.append(label)

    return {
        "original": original,
        "translated": translated,
        "prediction": label
    }

# -------------------------
# EMOTION ANALYSIS
# -------------------------

@app.post("/emotion")
def emotion_analysis(comment: Comment):

    translated = translate_to_english(comment.text)
    cleaned = clean_text(translated)

    emotion = emotion_model.predict([cleaned])[0]

    return {
        "original": comment.text,
        "translated": translated,
        "emotion": emotion
    }


# -------------------------
# MULTILINGUAL TOXIC
# -------------------------

@app.post("/multilingual")
def multilingual_analysis(comment: Comment):

    text = comment.text

    try:
        prediction = multilingual_model.predict([text])[0]

    except:
        translated = translate_to_english(text)
        cleaned = clean_text(translated)
        prediction = toxic_model.predict([cleaned])[0]

    label = "toxic" if str(prediction).lower() in ["1", "toxic"] else "non_toxic"

    return {
        "text": text,
        "prediction": label
    }


# -------------------------
# BEHAVIOUR ANALYSIS
# -------------------------

@app.post("/behaviour")
def behaviour_analysis(comment: Comment):

    translated = translate_to_english(comment.text)

    prediction = behaviour_model.predict([translated])[0]

    return {
        "original": comment.text,
        "translated": translated,
        "behaviour": prediction
    }


# -------------------------
# SAFE REWRITE
# -------------------------

@app.post("/safe-rewrite")
def safe_rewrite(comment: Comment):

    translated = translate_to_english(comment.text)
    cleaned = clean_text(translated)

    rewritten = rewrite_model.predict([cleaned])[0]

    return {
        "original": comment.text,
        "translated": translated,
        "safe_rewrite": rewritten
    }


# -------------------------
# SAFE COMMENT COACH
# -------------------------
@app.post("/safe-coach")
def safe_coach(comment: Comment):

    original = comment.text

    translated = translate_to_english(original)

    cleaned = clean_text(translated)

    behaviour = coach_model.predict([cleaned])[0]

    # dataset lookup
    match = coach_dataset[
        coach_dataset["behaviour"] == behaviour
    ]

    suggestion = None

    if len(match) > 0:
        suggestion = match["suggestion"].iloc[0]

    return {
        "original": original,
        "translated": translated,
        "behaviour": behaviour,
        "suggestion": suggestion
    }

# -------------------------
# TRANSLATE API
# -------------------------

@app.post("/translate")
def translate_text(data: TranslateRequest):

    try:
        translated = GoogleTranslator(
            source=data.lang,
            target="en"
        ).translate(data.text)

    except:
        translated = "Translation failed"

    return {
        "original": data.text,
        "translated": translated
    }


# -------------------------
# DASHBOARD STATS
# -------------------------

@app.get("/stats")
def get_stats():

    toxic_count = history.count("toxic")
    non_count = history.count("non_toxic")

    return {
        "total_requests": len(history),
        "toxic": toxic_count,
        "non_toxic": non_count
    }


# -------------------------
# SYSTEM STATE
# -------------------------

@app.get("/state")
def get_state():

    return {
        "app_status": "Running",

        "toxic_model_loaded": toxic_model is not None,
        "emotion_model_loaded": emotion_model is not None,
        "multilingual_model_loaded": multilingual_model is not None,
        "behaviour_model_loaded": behaviour_model is not None,

        "rewrite_model_loaded": rewrite_model is not None,
        "coach_model_loaded": coach_model is not None
    }