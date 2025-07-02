import streamlit as st
import torch
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM,
    GPT2Tokenizer, GPT2LMHeadModel
)

st.set_page_config(page_title="🩺 Radiology Report Generator", layout="centered")
st.title("🖼️ Upload Chest X-ray for Report")

uploaded_file = st.file_uploader("Upload a PNG/JPG image", type=["png", "jpg", "jpeg"])
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    tinyllama_tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tinyllama = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(device)

    gpt2_tok = GPT2Tokenizer.from_pretrained("distilgpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

    return clip, clip_proc, blip, blip_proc, tinyllama_tok, tinyllama, gpt2_tok, gpt2

clip_model, clip_proc, blip_model, blip_proc, llama_tok, llama, gpt2_tok, gpt2 = load_models()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("## 🧠 Image Caption")
    blip_inputs = blip_proc(images=image, return_tensors="pt").to(device)
    caption_ids = blip_model.generate(**blip_inputs)
    caption = blip_proc.decode(caption_ids[0], skip_special_tokens=True)
    st.success(caption)

    # 🔍 Text database for similarity match
    reference_findings = [
        "There is evidence of right lower lobe consolidation.",
        "No acute cardiopulmonary abnormality.",
        "Mild cardiomegaly with clear lungs.",
        "Findings consistent with pneumonia.",
        "Chronic interstitial changes noted."
    ]

    # CLIP retrieval
    img_inputs = clip_proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_feats = clip_model.get_image_features(**img_inputs)
        image_feats = torch.nn.functional.normalize(image_feats, dim=-1)

    text_inputs = clip_proc(text=reference_findings, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_feats = clip_model.get_text_features(**text_inputs)
        text_feats = torch.nn.functional.normalize(text_feats, dim=-1)

    cosine_scores = torch.matmul(text_feats, image_feats.T).squeeze()
    topk = torch.topk(cosine_scores, k=3)
    top_reports = [reference_findings[i] for i in topk.indices.tolist()]

    st.markdown("## 📚 Retrieved Reports")
    for report in top_reports:
        st.markdown(f"- {report}")

    # ✏️ Drafting with TinyLlama
    st.markdown("## 📝 Drafting Report")
    llama_prompt = (
        f"🧠 Caption: {caption}\n"
        f"📚 Retrieved Reports:\n"
        + "\n".join(f"- {r}" for r in top_reports)
        + "\n\nGenerate a clinical-style radiology report:"
    )

    inputs = llama_tok(llama_prompt, return_tensors="pt").to(device)
    output_ids = llama.generate(**inputs, max_new_tokens=300, temperature=0.8)
    draft = llama_tok.decode(output_ids[0], skip_special_tokens=True)
    st.text_area("Draft Report", draft, height=200)

    # 🧾 Final Refinement with GPT-2
    st.markdown("## ✅ Final Radiology Report")
    final_prompt = f"Caption: {caption}\nDraft: {draft}\nRefine this into a structured radiology report:"
    gpt2_inputs = gpt2_tok(final_prompt, return_tensors="pt").to(device)
    gpt2_output = gpt2.generate(**gpt2_inputs, max_new_tokens=200)
    final = gpt2_tok.decode(gpt2_output[0], skip_special_tokens=True)
    st.success(final)
