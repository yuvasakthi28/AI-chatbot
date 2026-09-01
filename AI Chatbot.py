import os
import asyncio
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
from audio_recorder_streamlit import audio_recorder
import edge_tts

st.set_page_config(page_title="Groq Voice Chatbot", page_icon="🎙️")
st.title("🎙️ Groq Voice AI Chatbot")

# Helper function to generate TTS audio asynchronously using edge-tts
async def generate_speech(text, output_filename):
    communicate = edge_tts.Communicate(text, "en-US-AvaNeural")
    await communicate.save(output_filename)

# 1. Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Groq API Key:", type="password")
    selected_model = st.selectbox(
    "Choose Model:",
    [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b"
    ]
)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.warning("Please enter your Groq API Key in the sidebar to proceed.")
    st.stop()

client = Groq(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 2. Audio Input Component
st.write("---")
st.write("🗣️ **Speak to the AI:**")
audio_bytes = audio_recorder(
    text="Click to record",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_name="microphone",
    icon_size="2x"
)

user_prompt = None

# Process Voice Input via Groq Whisper
if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    with st.spinner("Transcribing voice..."):
        with open(temp_audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3-turbo",
                response_format="text"
            )
        user_prompt = str(transcription).strip()

    os.remove(temp_audio_path)

# Text Input Fallback
text_input = st.chat_input("Or type your message here...")
if text_input:
    user_prompt = text_input

# 3. Generate AI Response & Neural Audio
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            )
            bot_reply = response.choices[0].message.content
            st.write(bot_reply)

            # Generate Neural Voice Audio using edge-tts
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_tts:
                temp_tts_path = temp_tts.name

            asyncio.run(generate_speech(bot_reply, temp_tts_path))
            st.audio(temp_tts_path, format="audio/mp3", autoplay=True)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
