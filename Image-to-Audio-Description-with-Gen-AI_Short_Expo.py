import streamlit as st

# Must be the first command
st.set_page_config(page_title="Image to Audio", layout="centered")

from PIL import Image
import requests
from gtts import gTTS
import torch
from transformers import AutoProcessor
from transformers import BlipForConditionalGeneration

# Groq API Key – replace this with your actual key
GROQ_API_KEY = "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"

# Load BLIP model and processor
@st.cache_resource
def load_blip_model():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

# Function to simplify description using Groq
def query_groq_short(prompt, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "Describe the image briefly in one short sentence."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Groq API failed: {e}"

# Function to convert text to audio
def generate_audio(text):
    tts = gTTS(text)
    audio_path = "short_desc.mp3"
    tts.save(audio_path)
    return audio_path

# App Title
st.title("🖼️ ➜ 🎤 Image to Audio Description")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    if st.button("Generate Audio Description"):
        st.info("Processing image...")

        # Generate image caption
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Shorten description using Groq
        refined = query_groq_short(caption, GROQ_API_KEY)
        final_text = refined if "Groq API failed" not in refined else caption

        # Generate audio from final text
        audio_file = generate_audio(final_text)

        # Layout images side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(image, caption="Processed Image", use_container_width=True)

        # Show description and audio
        st.success(f"📝 Description: {final_text}")
        st.audio(audio_file, format="audio/mp3")
