import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

st.set_page_config(page_title="Medical Image Report", layout="centered")
st.title("🖼️ Upload an Image for Visual Report")

# Upload section
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model

processor, model = load_clip()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.subheader("📝 Report")

    # 🩺 Medical labels
    text_labels = [
        "Normal chest X-ray",
        "Chest X-ray showing pneumonia",
        "Lung cancer",
        "Tuberculosis in lungs",
        "COVID-19 infection in lungs"
    ]

    # Run CLIP model
    inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    best_idx = torch.argmax(probs).item()

    st.markdown(f"**Best match:** `{text_labels[best_idx]}`")
    st.markdown(f"**Confidence:** `{probs[0][best_idx]:.2f}`")

    # Optional: add full probability breakdown
    st.markdown("---")
    st.write("### 🔍 Full Label Probabilities")
    for i, label in enumerate(text_labels):
        st.write(f"- {label}: `{probs[0][i].item():.2f}`")

    # OPTIONAL: Enable captioning (requires Hugging Face GPU or local)
    # from transformers import BlipProcessor, BlipForConditionalGeneration
    # @st.cache_resource
    # def load_blip():
    #     proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    #     model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    #     return proc, model
    #
    # blip_proc, blip_model = load_blip()
    # blip_inputs = blip_proc(image, return_tensors="pt").to(device)
    # caption_ids = blip_model.generate(**blip_inputs)
    # caption = blip_proc.decode(caption_ids[0], skip_special_tokens=True)
    #
    # st.markdown("### 🖋️ BLIP Caption")
    # st.success(caption)
