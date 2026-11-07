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

MODEL_NAME = "superb/wav2vec2-base-superb-er"
SAMPLE_RATE = 16000
MAX_SECONDS = 10  # مقطع أقصر = رام أقل (مهم في الخطة المجانية)

# عدّلها لتطابق طريقة Spectral Gating اللي استخدمتوها في التدريب
GATING_STRENGTH = 0.45

EMOJI = {"angry": "😠", "happy": "😄", "sad": "😢", "neutral": "😐"}

torch.set_num_threads(2)


# نفس الكلاس حق الكولاب بالضبط
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


@st.cache_resource(show_spinner="جاري تحميل الموديل (أول مرة تاخذ دقيقة)...")
def load_models():
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    w2v = Wav2Vec2Model.from_pretrained(MODEL_NAME).eval()
    scaler = np.load("scaler.npz")
    with open("classes.json") as f:
        classes = json.load(f)
    clf = EmotionClassifier(768, len(classes))
    clf.load_state_dict(torch.load("emotion_classifier.pt", map_location="cpu"))
    clf.eval()
    return fe, w2v, scaler, classes, clf


def predict(audio_bytes):
    fe, w2v, scaler, classes, clf = load_models()

    signal, _ = librosa.load(
        io.BytesIO(audio_bytes), sr=SAMPLE_RATE, duration=MAX_SECONDS
    )
    if len(signal) < SAMPLE_RATE * 0.5:
        raise ValueError("المقطع قصير جداً، جرّب مقطع أطول من نصف ثانية.")

    signal = nr.reduce_noise(y=signal, sr=SAMPLE_RATE, prop_decrease=GATING_STRENGTH)

    inputs = fe(signal, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values

    with torch.no_grad():
        hidden = w2v(inputs).last_hidden_state
        feat = hidden.mean(dim=1).squeeze().numpy()
        feat = (feat - scaler["mean"]) / scaler["scale"]
        logits = clf(torch.tensor(feat, dtype=torch.float32).unsqueeze(0))
        probs = torch.softmax(logits, dim=1).squeeze().numpy()

    return dict(zip(classes, probs.tolist()))


st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
st.title("🎙️ Speech Emotion Recognition")
st.caption("Wav2Vec2 (SUPERB) + Spectral Gating | IEMOCAP: angry, happy, sad, neutral")

tab_upload, tab_mic = st.tabs(["📁 رفع ملف", "🎤 تسجيل"])

audio_bytes = None

with tab_upload:
    uploaded = st.file_uploader("ارفع مقطع صوتي", type=["wav", "mp3", "flac", "ogg", "m4a"])
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        st.audio(audio_bytes)

with tab_mic:
    recorded = st.audio_input("اضغط وسجّل صوتك")
    if recorded is not None:
        audio_bytes = recorded.getvalue()

if audio_bytes is not None and st.button("حلّل المشاعر", type="primary"):
    try:
        with st.spinner("جاري التحليل..."):
            probs = predict(audio_bytes)
        top = max(probs, key=probs.get)
        st.success(f"{EMOJI.get(top, '')} المشاعر المتوقعة: **{top}** ({probs[top] * 100:.1f}%)")
        st.bar_chart(pd.Series(probs, name="الاحتمال"))
    except Exception as e:
        st.error(f"صار خطأ: {e}")
