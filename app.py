# 📦 Imports
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import io
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM
)
import warnings

warnings.filterwarnings('ignore')

# 🔁 Device setup with memory optimization
# This will be run once when the app starts
device = "cuda" if torch.cuda.is_available() else "cpu"
if 'cuda' in device:
    torch.backends.cuda.matmul.allow_tf32 = True

# --- Model Loading Functions (using st.cache_resource for efficiency) ---
@st.cache_resource
@torch.no_grad()
def load_all_models():
    """
    Loads all necessary models into memory.
    Uses st.cache_resource to load models only once across Streamlit reruns.
    """
    print("🔄 Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        device_map="auto",
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32
    )
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    st.write("✅ CLIP model loaded.")

    print("\n🔄 Loading BLIP model...")
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        device_map="auto",
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32
    )
    st.write("✅ BLIP model loaded.")

    print("\n🔄 Loading Falcon-7B model...")
    falcon_tok = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    falcon = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b-instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    ).eval()
    st.write("✅ Falcon model loaded.")

    print("\n🔄 Loading Wizard-7B model...")
    r_tok = AutoTokenizer.from_pretrained("airesearch/wizard-7b-v1.0")
    refiner = AutoModelForCausalLM.from_pretrained(
        "airesearch/wizard-7b-v1.0",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    ).eval()
    st.write("✅ Wizard model loaded.")
    st.success("All AI models are ready!")
    return clip_model, clip_proc, blip_proc, blip_model, falcon, falcon_tok, refiner, r_tok

# Load all models once when the app starts
clip_model, clip_proc, blip_proc, blip_model, falcon, falcon_tok, refiner, r_tok = load_all_models()

# ---------------------------
# Helper Functions for Inference
# ---------------------------
def load_image_from_uploaded_file(uploaded_file):
    """
    Loads an image from Streamlit's UploadedFile object.
    """
    try:
        img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB").resize((224, 224))
        return img
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        return None

def get_clip_features(img):
    """Extracts image and text features using CLIP and finds top matching reports."""
    report_db = [
        "Consolidation RLL", "No acute findings",
        "Mild cardiomegaly", "Pneumonia",
        "Interstitial changes", "Pleural effusion",
        "Pulmonary edema", "Atelectasis"
    ]
    with torch.no_grad():
        img_inputs = clip_proc(images=img, return_tensors="pt").to(device)
        img_feat = clip_model.get_image_features(**img_inputs)
        img_feat = F.normalize(img_feat, dim=-1)

        txt_inputs = clip_proc(text=report_db, return_tensors="pt", padding=True).to(device)
        txt_feat = clip_model.get_text_features(**txt_inputs)
        txt_feat = F.normalize(txt_feat, dim=-1)

        scores = (txt_feat @ img_feat.T).squeeze()
        retrieved = [report_db[i] for i in scores.topk(3).indices.tolist()]
    return retrieved

def generate_blip_caption(img):
    """Generates a caption for the given image using BLIP."""
    with torch.no_grad():
        inputs = blip_proc(images=img, return_tensors="pt").to(device)
        cap_ids = blip_model.generate(**inputs)
        caption = blip_proc.decode(cap_ids[0], skip_special_tokens=True)
    return caption

def generate_text(model, tokenizer, prompt, max_tokens=150):
    """Generates text using a given language model."""
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ---------------------------
# Streamlit UI and Application Logic
# ---------------------------
st.set_page_config(page_title="Medical Report Generator", layout="centered")

st.title("🩺 AI-Powered Medical Report Generator")
st.markdown(
    """
    Upload a chest X-ray image (JPG or PNG) and let our AI generate a professional medical report.
    """
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, PNG"
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Load the image for processing
    img = load_image_from_uploaded_file(uploaded_file)

    if img is not None:
        with st.spinner("Generating report... This may take a moment."):
            try:
                # --- CLIP: Feature Extraction and Retrieval ---
                st.subheader("1. Retrieving Similar Cases")
                retrieved_reports = get_clip_features(img)
                st.write("Top 3 matching reports from database:")
                for i, report in enumerate(retrieved_reports, 1):
                    st.write(f"- {report}")

                # --- BLIP: Captioning ---
                st.subheader("2. Generating Image Caption")
                caption = generate_blip_caption(img)
                st.write(f"**Image Caption:** {caption}")

                # --- Falcon: Draft Generation ---
                st.subheader("3. Generating Draft Report (Falcon)")
                prompt_draft = f"Caption: {caption}\nReports:\n- "+ "\n- ".join(retrieved_reports) + "\nDraft:"
                draft_report = generate_text(falcon, falcon_tok, prompt_draft, 150)
                st.write(f"**Draft Report:** {draft_report}")

                # --- Wizard: Refined Report Generation ---
                st.subheader("4. Refining Report (Wizard)")
                prompt_refined = f"Caption: {caption}\nDraft: {draft_report}\nRefine this medical report to be more professional:"
                refined_report = generate_text(refiner, r_tok, prompt_refined, 100)
                st.success("Report generation complete!")

                st.markdown("---")
                st.subheader("✨ Final Professional Medical Report:")
                st.text_area("Final Report", refined_report, height=200)
                st.markdown("---")

            except Exception as e:
                st.error(f"An error occurred during report generation: {e}")
                st.info("Please try uploading another image or check the console for more details.")
    else:
        st.warning("Please upload a valid image file to proceed.")
else:
    st.info("Upload a chest X-ray image above to get started!")
