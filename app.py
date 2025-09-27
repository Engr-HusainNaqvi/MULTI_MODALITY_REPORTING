import streamlit as st
import torch
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM,
    GPT2Tokenizer, GPT2LMHeadModel
)

st.set_page_config(page_title="🩺 AI Radiology Assistant", layout="wide")

# ---------- Custom CSS Styling ----------
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Lato', sans-serif;
            background-color: #f7f9fc;
        }
        .report-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #ddd;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .stTextArea textarea {
            font-family: 'Courier New', monospace;
            background-color: #eef2f7;
        }
        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #164863;
            border-bottom: 2px solid #dbe2ef;
            margin-top: 2rem;
            padding-bottom: 5px;
        }
        .upload-box {
            border: 2px dashed #3b82f6;
            padding: 20px;
            background-color: #ecf3fc;
            border-radius: 10px;
            text-align: center;
        }
        .uploaded-image {
            border-radius: 8px;
            margin-top: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar Navigation ----------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/stethoscope.png", width=70)
    st.markdown("## 📁 Navigation")
    st.markdown("""
    - 🧠 **AI Reports**
    - 📤 **Upload Image**
    - 📊 **Patient History**
    - 📂 **Templates**
    """)
    st.markdown("---")
    st.markdown("🔗 [Export Report as PDF](#)", unsafe_allow_html=True)

st.title("🩺 AI-Powered Chest X-ray Interpretation Tool")

# ---------- Upload Section ----------
with st.container():
    st.markdown('<div class="section-title">📤 Upload Chest X-ray</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PNG or JPG", type=["png", "jpg", "jpeg"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    llama_tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    llama_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(device)
    gpt2_tok = GPT2Tokenizer.from_pretrained("distilgpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
    return clip_model, clip_proc, blip_model, blip_proc, llama_tok, llama_model, gpt2_tok, gpt2_model

clip_model, clip_proc, blip_model, blip_proc, llama_tok, llama_model, gpt2_tok, gpt2_model = load_models()

# ---------- Inference Section ----------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="🖼️ Uploaded Chest X-ray", use_container_width=True, output_format="JPEG")

    # Step 1: Captioning
    st.markdown('<div class="section-title">🧠 AI Captioning</div>', unsafe_allow_html=True)
    blip_inputs = blip_proc(images=image, return_tensors="pt").to(device)
    caption_ids = blip_model.generate(**blip_inputs)
    caption = blip_proc.decode(caption_ids[0], skip_special_tokens=True)
    st.success(f"🧠 Caption Generated: **{caption}**")

    # Step 2: Finding Retrieval
    reference_findings = [
        "There is evidence of right lower lobe consolidation.",
        "No acute cardiopulmonary abnormality.",
        "Mild cardiomegaly with clear lungs.",
        "Findings consistent with pneumonia.",
        "Chronic interstitial changes noted."
    ]
    img_inputs = clip_proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_features = clip_model.get_image_features(**img_inputs)
        img_features = torch.nn.functional.normalize(img_features, dim=-1)

    text_inputs = clip_proc(text=reference_findings, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        txt_features = clip_model.get_text_features(**text_inputs)
        txt_features = torch.nn.functional.normalize(txt_features, dim=-1)

    similarities = torch.matmul(txt_features, img_features.T).squeeze()
    top_indices = torch.topk(similarities, k=3).indices
    top_findings = [reference_findings[i] for i in top_indices]

    st.markdown('<div class="section-title">🔍 Top Similar Clinical Findings</div>', unsafe_allow_html=True)
    st.info(f"• {top_findings[0]}\n\n• {top_findings[1]}\n\n• {top_findings[2]}")

    # Step 3: Draft Report via TinyLlama
    llama_prompt = (
        f"🧠 Caption: {caption}\n"
        f"📚 Retrieved Reports:\n" +
        "\n".join(f"- {f}" for f in top_findings) +
        "\n\nGenerate a clinical-style radiology report:"
    )
    llama_inputs = llama_tok(llama_prompt, return_tensors="pt").to(device)
    llama_output = llama_model.generate(**llama_inputs, max_new_tokens=300, temperature=0.7)
    draft_report = llama_tok.decode(llama_output[0], skip_special_tokens=True)

    # Step 4–5: Refinements via GPT-2
    gpt2_input_1 = f"Caption: {caption}\nDraft: {draft_report}\nRefine this into a structured radiology report:"
    gpt2_inputs_1 = gpt2_tok(gpt2_input_1, return_tensors="pt").to(device)
    gpt2_output_1 = gpt2_model.generate(**gpt2_inputs_1, max_new_tokens=250, do_sample=True)
    refined_1 = gpt2_tok.decode(gpt2_output_1[0], skip_special_tokens=True)

    gpt2_input_2 = f"{refined_1}\nRefine the new report:"
    gpt2_inputs_2 = gpt2_tok(gpt2_input_2, return_tensors="pt").to(device)
    gpt2_output_2 = gpt2_model.generate(**gpt2_inputs_2, max_new_tokens=200, do_sample=True)
    refined_2 = gpt2_tok.decode(gpt2_output_2[0], skip_special_tokens=True)

    # Display Report
    st.markdown('<div class="section-title">📄 Final Structured Radiology Report</div>', unsafe_allow_html=True)
    st.text_area("Refined Report", refined_2, height=600)
