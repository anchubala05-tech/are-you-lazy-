"""
app.py
------
A tiny Streamlit UI on top of the trained model.

Run with:
    streamlit run app.py

Requires the model to already be trained (see README): the app loads
saved_models/best_model.pt and saved_models/scaler.json.
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import FEATURE_COLUMNS
from predict import load_model, predict_one, verdict

st.set_page_config(page_title="AI Procrastination Predictor", page_icon="\U0001F4A4")

st.title("\U0001F4A4 AI Procrastination Predictor")
st.caption(
    "A small PyTorch model guesses whether you're about to procrastinate on a task. "
    "Not a licensed therapist. Not licensed at all, actually."
)

MODEL_DIR = "saved_models"

if not os.path.exists(os.path.join(MODEL_DIR, "best_model.pt")):
    st.error(
        "No trained model found. Run this first:\n\n"
        "```\npython data/generate_data.py\npython src/train.py\n```"
    )
    st.stop()


@st.cache_resource
def _load():
    return load_model(MODEL_DIR)


model, scaler, device = _load()

col1, col2 = st.columns(2)
with col1:
    hours_until_deadline = st.slider("Hours until deadline", 0.5, 168.0, 24.0)
    task_difficulty = st.slider("Task difficulty", 1, 10, 5)
    hours_slept = st.slider("Hours slept last night", 2.0, 10.0, 7.0)
    cups_of_coffee = st.slider("Cups of coffee today", 0, 8, 1)
    open_browser_tabs = st.slider("Open browser tabs", 1, 60, 10)

with col2:
    social_media_checks_per_hour = st.slider("Social media checks / hour", 0, 30, 3)
    mood_score = st.slider("Mood right now", 1, 10, 6)
    previous_completion_rate = st.slider("Historical on-time completion rate", 0.0, 1.0, 0.75)
    is_weekend = st.checkbox("Is it the weekend?")
    noise_level = st.slider("Environment noise / distraction", 1, 10, 4)

feature_dict = {
    "hours_until_deadline": hours_until_deadline,
    "task_difficulty": task_difficulty,
    "hours_slept": hours_slept,
    "cups_of_coffee": cups_of_coffee,
    "open_browser_tabs": open_browser_tabs,
    "social_media_checks_per_hour": social_media_checks_per_hour,
    "mood_score": mood_score,
    "previous_completion_rate": previous_completion_rate,
    "is_weekend": 1.0 if is_weekend else 0.0,
    "noise_level": noise_level,
}
assert set(feature_dict) == set(FEATURE_COLUMNS)

if st.button("Predict", type="primary"):
    prob = predict_one(model, scaler, device, feature_dict)
    st.metric("Procrastination probability", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    st.write(verdict(prob))
