import streamlit as st
from PIL import Image
import requests
from gtts import gTTS
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# 🛑 Streamlit Page Config
st.set_page_config(page_title="Image to Audio Description", layout="centered")

# 🚀 Load BLIP model (cached)
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

# ✅ Safe text cleaner (removes emojis / non-ascii)
def safe_text(text: str) -> str:
    return text.encode("ascii", "ignore").decode()

# 🧠 Function to query Groq API
def query_groq(prompt, api_key):
    if not api_key:
        return "ERROR: No API key provided. Please enter your Groq API key in the sidebar."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an expert in understanding and describing images."},
            {"role": "user", "content": f"{prompt}"}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        if "choices" not in result:
            return f"ERROR: Unexpected response: {safe_text(str(result))}"

        return safe_text(result["choices"][0]["message"]["content"].strip())

    except requests.exceptions.HTTPError as http_err:
        return f"ERROR: HTTP Error: {safe_text(str(http_err))}"
    except Exception as e:
        return f"ERROR: {safe_text(str(e))}"

# 🎵 Generate audio from text
def generate_audio(text):
    tts = gTTS(text)
    audio_path = "image_description.mp3"
    tts.save(audio_path)
    return audio_path

# 🌟 App Title
st.title("🖼️ Image to Audio Description with Gen AI 🔉 ")

# 📤 Upload Image
uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# 🔑 Groq API Key section
st.sidebar.title("Groq API Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# ✅ Button to verify API Key
if st.sidebar.button("Verify API Key"):
    test_response = query_groq("Hello, can you confirm API connection?", api_key)
    if test_response.startswith("ERROR"):
        st.sidebar.error("❌ Invalid API Key or Connection Failed")
    else:
        st.sidebar.success("✅ API Key Verified Successfully!")

# Main processing
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    if st.button("Generate Audio Description"):
        # Step 1: Generate Caption with BLIP
        st.info("Generating image caption...")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Step 2: Refine using Groq
        refined_description = query_groq(f"Refine this image description: {caption}", api_key)

        if refined_description.startswith("ERROR"):
            st.error(refined_description)
        else:
            st.subheader("Refined Description:")
            st.write(refined_description)

            # Step 3: Show side-by-side images
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", width=300)
            with col2:
                st.image(image, caption="Processed Image", width=300)

            # Step 4: Convert to Audio
            st.info("Converting description to audio...")
            audio_path = generate_audio(refined_description)
            audio_file = open(audio_path, "rb").read()
            st.audio(audio_file, format="audio/mp3")

            # Step 5: Custom Prompt Section
            st.markdown("---")
            st.subheader("📝 Custom Prompt Section")
            custom_prompt = st.text_area("Enter your own prompt related to this image:")
            if st.button("Submit Prompt to Groq"):
                custom_response = query_groq(custom_prompt, api_key)
                if custom_response.startswith("ERROR"):
                    st.error(custom_response)
                else:
                    st.success("Groq Response to Your Prompt:")
                    st.write(custom_response)
