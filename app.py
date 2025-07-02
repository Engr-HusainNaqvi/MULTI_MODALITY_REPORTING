import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

st.set_page_config(page_title="CLIP Report Generator", layout="centered")
st.title("🖼️ Upload an Image for Visual Report")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model + processor
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model

processor, model = load_clip()

# Inference
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("📝 Report")
    st.info("This is a placeholder report. You can extend this with image-text comparisons or captioning.")

    # Example: Use fixed prompt for encoding
    text = ["A photo of a cat", "A photo of a dog", "A landscape", "Medical scan"]
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Show best match
    best_idx = torch.argmax(probs).item()
    st.success(f"Best match: **{text[best_idx]}** (Confidence: {probs[0][best_idx]:.2f})")
