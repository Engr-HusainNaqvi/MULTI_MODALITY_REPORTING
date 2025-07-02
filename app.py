import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image

st.set_page_config(page_title="Image Report Generator", layout="centered")
st.title("🧠 Simple Image Uploader with TinyLlama")

uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

@st.cache_resource
def load_model():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.subheader("📝 Report")
    st.success("Model loaded! You can now implement report logic here.")
