import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM
)
import evaluate

# 🧠 Initialize once
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    falcon_model = "tiiuae/falcon-7b-instruct"
    falcon_tok = AutoTokenizer.from_pretrained(falcon_model)
    falcon = AutoModelForCausalLM.from_pretrained(falcon_model, torch_dtype=torch.float16, device_map="auto").eval()

    refiner_model = "airesearch/wizard-7b-v1.0"
    r_tok = AutoTokenizer.from_pretrained(refiner_model)
    refiner = AutoModelForCausalLM.from_pretrained(refiner_model, torch_dtype=torch.float16, device_map="auto").eval()

    return {
        "device": device,
        "clip_model": clip_model, "clip_proc": clip_proc,
        "blip_model": blip_model, "blip_proc": blip_proc,
        "falcon": falcon, "falcon_tok": falcon_tok,
        "refiner": refiner, "r_tok": r_tok
    }

models = load_models()

# 🖼️ UI: upload image
st.title("🔎 Radiology Report Generator")
uploaded = st.file_uploader("Upload a medical image (jpg/png)", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Upload an image to proceed")
    st.stop()

img = Image.open(uploaded).convert("RGB").resize((224,224))
st.image(img, caption="Uploaded Image", use_column_width=True)

device = models["device"]

# 1️⃣ CLIP extraction + Retrieval
clip_input = models["clip_proc"](images=img, return_tensors="pt").to(device)
img_feat = models["clip_model"].get_image_features(**clip_input)
img_feat = F.normalize(img_feat, dim=-1)

report_db = [
    "There is evidence of right lower lobe consolidation.",
    "No acute cardiopulmonary abnormality.",
    "Mild cardiomegaly with clear lungs.",
    "Findings consistent with pneumonia.",
    "Chronic interstitial changes noted."
]
txt_input = models["clip_proc"](text=report_db, return_tensors="pt", padding=True).to(device)
txt_feat = models["clip_model"].get_text_features(**txt_input)
txt_feat = F.normalize(txt_feat, dim=-1)

scores = (txt_feat @ img_feat.T).squeeze()
topk = torch.topk(scores, k=3).indices
retrieved_reports = [report_db[i] for i in topk]

st.subheader("📚 Retrieved Similar Reports")
for r in retrieved_reports:
    st.write(f"- {r}")

# 2️⃣ BLIP captioning
blip_input = models["blip_proc"](images=img, return_tensors="pt").to(device)
cap_ids = models["blip_model"].generate(**blip_input)
caption = models["blip_proc"].decode(cap_ids[0], skip_special_tokens=True)
st.subheader("🖼️ Image Caption")
st.write(caption)

# 3️⃣ Draft report generation
prompt = f"Caption: {caption}\nSimilar prior reports:\n- " + "\n- ".join(retrieved_reports) + "\n\nWrite a draft radiology report:"
inp = models["falcon_tok"](prompt, return_tensors="pt").to(device)
draft_ids = models["falcon"].generate(**inp, max_new_tokens=150)
draft = models["falcon_tok"].decode(draft_ids[0], skip_special_tokens=True)
st.subheader("✍️ Draft Report")
st.write(draft)

# 4️⃣ Refined report
prompt2 = f"Caption: {caption}\nDraft: {draft}\n\nRefine this into a concise radiology report:"
r_inp = models["r_tok"](prompt2, return_tensors="pt").to(device)
refine_ids = models["refiner"].generate(**r_inp, max_new_tokens=100)
refined = models["r_tok"].decode(refine_ids[0], skip_special_tokens=True)
st.subheader("✅ Refined Final Report")
st.write(refined)

# 5️⃣ BLEU/ROUGE/BERTScore evaluation (dummy example)
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

dummy_ref = "No acute cardiopulmonary abnormality."
scores_bleu = bleu.compute(predictions=[refined], references=[[dummy_ref]])
scores_rouge = rouge.compute(predictions=[refined], references=[dummy_ref])
scores_bert = bertscore.compute(predictions=[refined], references=[dummy_ref], lang="en")

st.subheader("📊 Evaluation Scores (vs dummy reference)")
st.write(scores_bleu, scores_rouge, {k: round(v,4) for k,v in scores_bert.items()})
