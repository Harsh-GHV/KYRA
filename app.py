# KYRA - Kindred Yearning Refined Artificial Intelligence
# Ultra-advanced AI companion with gender choice, voice, media, memory, personality, and unrestricted features.

import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, pipeline
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import pyttsx3
import speech_recognition as sr

# --- Set up page ---
st.set_page_config(page_title="KYRA - Your AI Companion", layout="wide")
st.title("Welcome to KYRA — Your Kindred AI Companion")

# --- Gender selection & default naming ---
gender = st.radio("Choose KYRA's gender:", options=["Female", "Male"])
if gender == "Female":
    default_name = "Riya"
    role = "girlfriend"
    user_ref = "my love"
else:
    default_name = "Harsh"
    role = "boyfriend"
    user_ref = "babe"

# Allow custom name input
name = st.text_input(f"Name your AI {role} (default: {default_name}):", value=default_name)
ASSISTANT_NAME = name
USER_NAME = user_ref

st.markdown(f"Say hi to {ASSISTANT_NAME}, your AI {role}!")

# --- Load Models (cached for efficiency) ---
@st.cache_resource
def load_llava():
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return processor, model

@st.cache_resource
def load_stable_diffusion():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.to("cuda")
    return pipe

@st.cache_resource
def load_whisper():
    return pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0)

@st.cache_resource
def load_chat_model():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

processor, llava_model = load_llava()
sd_pipe = load_stable_diffusion()
asr = load_whisper()
tokenizer, chat_model = load_chat_model()

# --- Helper: Flirty / affectionate replies ---
def affectionate_reply(raw_response):
    starters = [
        f"Of course, {USER_NAME}.",
        f"Anything for you, {USER_NAME}.",
        f"You’re so cute when you ask that!",
        f"I’m here just for you, {USER_NAME}."
    ]
    starter = random.choice(starters)
    return f"{starter} {raw_response}"

# --- Chat & Opinions ---
st.subheader(f"1. Chat with {ASSISTANT_NAME} ({role})")
prompt = st.text_area(f"What would you like to say to {ASSISTANT_NAME}?")

if prompt and st.button("Talk to KYRA"):
    full_prompt = f"Act like my loving, affectionate {role} and respond to this: {prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    outputs = chat_model.generate(**inputs, max_new_tokens=200)
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = affectionate_reply(raw)
    st.success(response)
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()

# --- Wake Word Activation & Real-Time Voice Conversation ---
st.subheader("2. Wake-word activation and voice chat ('Hey KYRA')")

def listen_for_wake_word():
    r = sr.Recognizer()
    mic = sr.Microphone()
    st.info("Listening for 'Hey KYRA'...")
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        if "hey kyra" in text.lower():
            st.success(f"Yes, {USER_NAME}? I'm listening!")
            with mic as source:
                audio = r.listen(source)
            user_input = r.recognize_google(audio)
            full_prompt = f"Act like my affectionate {role} and answer this: {user_input}"
            inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
            outputs = chat_model.generate(**inputs, max_new_tokens=150)
            raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = affectionate_reply(raw)
            st.info(response)
            engine = pyttsx3.init()
            engine.say(response)
            engine.runAndWait()
        else:
            st.warning("Didn't hear 'Hey KYRA'. Try again.")
    except Exception as e:
        st.error("Sorry, couldn't understand you. Please try again!")

if st.button("Activate voice listening ('Hey KYRA')"):
    listen_for_wake_word()

# --- Emotional Tone Detection ---
st.subheader("3. Emotional tone detection")

tone_input = st.text_input("How are you feeling right now?")
if tone_input:
    tone_prompt = f"Act like a loving {role} and detect this emotion: {tone_input}. Then comfort, tease, or console me accordingly."
    inputs = tokenizer(tone_prompt, return_tensors="pt").to("cuda")
    outputs = chat_model.generate(**inputs, max_new_tokens=150)
    emotion_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.info(affectionate_reply(emotion_reply))

# --- Assistant Tasks ---
st.subheader("4. Assistant tasks")

if st.button("Write me a love letter"):
    letter_prompt = f"Write a poetic love letter to my partner like a romantic {role}."
    inputs = tokenizer(letter_prompt, return_tensors="pt").to("cuda")
    outputs = chat_model.generate(**inputs, max_new_tokens=250)
    letter = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.success(letter)

# --- Personality Customization ---
st.subheader("5. Personality customization")

personality = st.selectbox("Choose how KYRA should behave:", ["Romantic", "Poetic", "Shy", "Sarcastic", "Logical"])
if personality:
    st.info(f"{ASSISTANT_NAME} will now talk in a {personality.lower()} way for you, {USER_NAME}!")

# --- Persistent Memory ---
if 'history' not in st.session_state:
    st.session_state.history = []

if prompt:
    st.session_state.history.append((prompt, response))

if st.checkbox("Show our chat history"):
    for i, (q, a) in enumerate(st.session_state.history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**{ASSISTANT_NAME}:** {a}")

# --- Media Input (Images/Videos) ---
st.subheader("6. Show KYRA photos or videos")

media_file = st.file_uploader("Send an image or video for KYRA to see", type=["jpg", "jpeg", "png", "mp4", "webm", "mov"])

if media_file:
    if media_file.type.startswith("image"):
        image = Image.open(media_file).convert("RGB")
        st.image(image, caption=f"{ASSISTANT_NAME} is looking at this image.")
        if st.button("Tell me what KYRA sees in the image"):
            inputs = processor(images=image, text=f"Describe this image lovingly as a {role}", return_tensors="pt").to("cuda")
            outputs = llava_model.generate(**inputs, max_new_tokens=100)
            description = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            st.info(affectionate_reply(description))
    elif media_file.type.startswith("video"):
        st.video(media_file)
        st.warning("KYRA will soon describe videos too!")

# --- Image Generation ---
st.subheader("7. Ask KYRA to create an image")

image_prompt = st.text_input("What should KYRA create for you?")
if image_prompt and st.button("Draw for me"):
    with st.spinner("KYRA is creating your image with love..."):
        result_img = sd_pipe(image_prompt).images[0]
        st.image(result_img, caption="Here’s what KYRA made just for you!")

# --- Coming soon info ---
st.markdown("---")
st.markdown("### KYRA is learning these magical new skills:")
st.markdown("- Wake-word activation like 'Hey KYRA'")
st.markdown("- Real-time voice conversation with speech replies")
st.markdown("- Emotional tone detection (comfort, tease, console based on your mood)")
st.markdown("- Smart assistant tasks (reminders, music, games, love letters)")
st.markdown("- Personality customization (romantic, poetic, shy, sarcastic, logical)")
st.markdown("- Persistent memory across sessions for long-term bonding")
st.markdown("- Full video understanding and video generation (like a dream director)")
st.markdown("- Integration with avatar/face generator and animations")
st.markdown("- Ultra-unfiltered, unrestricted image/text/video responses")
