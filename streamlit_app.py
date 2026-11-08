
import io
import json

import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# ==========================================
# SETTINGS
# ==========================================

MODEL_NAME = "superb/wav2vec2-base-superb-er"
SAMPLE_RATE = 16000
MAX_SECONDS = 10

GATING_STRENGTH = 0.45

EMOJI = {
    "angry": "😠",
    "happy": "😄",
    "sad": "😢",
    "neutral": "😐"
}

torch.set_num_threads(2)


# ==========================================
# EMOTION CLASSIFIER
# Updated to match the saved model structure
# ==========================================

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource(
    show_spinner="جاري تحميل المودل، قد يستغرق بعض الوقت..."
)
def load_models():

    # Load SUPERB Wav2Vec2 feature extractor
    fe = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    # Load pretrained Wav2Vec2 model
    w2v = Wav2Vec2Model.from_pretrained(
        MODEL_NAME
    ).eval()

    # Load scaler
    scaler = np.load("scaler.npz")

    # Load emotion classes
    with open("classes.json", "r", encoding="utf-8") as f:
        classes = json.load(f)

    # Initialize emotion classifier
    clf = EmotionClassifier(
        input_dim=768,
        num_classes=len(classes)
    )

    # Load trained weights
    state_dict = torch.load(
        "emotion_classifier.pt",
        map_location="cpu",
        weights_only=True
    )

    clf.load_state_dict(state_dict)
    clf.eval()

    return fe, w2v, scaler, classes, clf


# ==========================================
# PREDICTION
# ==========================================

def predict(audio_bytes):

    fe, w2v, scaler, classes, clf = load_models()

    # Load audio and convert to 16 kHz
    signal, _ = librosa.load(
        io.BytesIO(audio_bytes),
        sr=SAMPLE_RATE,
        duration=MAX_SECONDS
    )

    # Check audio length
    if len(signal) < SAMPLE_RATE * 0.5:
        raise ValueError(
            "المقطع قصير جداً، جرّب مقطع أطول من نصف ثانية."
        )

    # Spectral Gating noise reduction
    signal = nr.reduce_noise(
        y=signal,
        sr=SAMPLE_RATE,
        prop_decrease=GATING_STRENGTH
    )

    # Wav2Vec2 input processing
    inputs = fe(
        signal,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).input_values

    # Extract features and predict emotion
    with torch.no_grad():

        hidden = w2v(inputs).last_hidden_state

        # Mean pooling
        feat = hidden.mean(dim=1).squeeze().numpy()

        # StandardScaler
        feat = (
            feat - scaler["mean"]
        ) / scaler["scale"]

        # Convert features to tensor
        features_tensor = torch.tensor(
            feat,
            dtype=torch.float32
        ).unsqueeze(0)

        # Emotion prediction
        logits = clf(features_tensor)

        probs = torch.softmax(
            logits,
            dim=1
        ).squeeze().numpy()

    return dict(zip(classes, probs.tolist()))


# ==========================================
# STREAMLIT INTERFACE
# ==========================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️"
)

st.title("🎙️ Speech Emotion Recognition")

st.caption(
    "Wav2Vec2 (SUPERB) + Spectral Gating | "
    "IEMOCAP: angry, happy, sad, neutral"
)

# Upload and microphone tabs
tab_upload, tab_mic = st.tabs([
    "📁 رفع ملف",
    "🎤 تسجيل"
])

audio_bytes = None


# ==========================================
# UPLOAD AUDIO
# ==========================================

with tab_upload:

    uploaded = st.file_uploader(
        "ارفع مقطع صوتي",
        type=["wav", "mp3", "flac", "ogg", "m4a"]
    )

    if uploaded is not None:

        audio_bytes = uploaded.getvalue()

        st.audio(audio_bytes)


# ==========================================
# RECORD AUDIO
# ==========================================

with tab_mic:

    recorded = st.audio_input(
        "اضغط وسجّل صوتك"
    )

    if recorded is not None:

        audio_bytes = recorded.getvalue()


# ==========================================
# ANALYZE EMOTION
# ==========================================

if audio_bytes is not None:

    if st.button(
        "حلّل المشاعر",
        type="primary"
    ):

        try:

            with st.spinner("جاري التحليل..."):

                probs = predict(audio_bytes)

            # Get predicted emotion
            top = max(
                probs,
                key=probs.get
            )

            # Display result
            st.success(
                f"{EMOJI.get(top, '')} "
                f"المشاعر المتوقعة: **{top}** "
                f"({probs[top] * 100:.1f}%)"
            )

            # Display emotion probabilities
            st.bar_chart(
                pd.Series(
                    probs,
                    name="الاحتمال"
                )
            )

        except Exception as e:

            st.error(
                f"صار خطأ: {e}"
            )
