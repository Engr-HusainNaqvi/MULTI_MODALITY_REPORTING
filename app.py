# 📦 Imports
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
from flask import Flask, request, jsonify # Import Flask components
from flask_cors import CORS # Import CORS for cross-origin requests

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# 🔁 Device setup with memory optimization
device = "cuda" if torch.cuda.is_available() else "cpu"
# Enable faster matmul on modern GPUs if CUDA is available
if 'cuda' in device:
    torch.backends.cuda.matmul.allow_tf32 = True

# Global variables to store loaded models
# Models will be loaded once when the server starts
clip_model, clip_proc = None, None
blip_proc, blip_model = None, None
falcon, falcon_tok = None, None
refiner, r_tok = None, None

# --- Model Loading Functions (modified to load once) ---
@torch.no_grad()
def load_all_models():
    """Loads all necessary models into global variables."""
    global clip_model, clip_proc, blip_proc, blip_model, falcon, falcon_tok, refiner, r_tok

    print("🔄 Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        device_map="auto",
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32
    )
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model loaded.")

    print("\n🔄 Loading BLIP model...")
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        device_map="auto",
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32
    )
    print("BLIP model loaded.")

    print("\n🔄 Loading Falcon-7B model...")
    falcon_tok = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    falcon = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b-instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    ).eval()
    print("Falcon model loaded.")

    print("\n🔄 Loading Wizard-7B model...")
    r_tok = AutoTokenizer.from_pretrained("airesearch/wizard-7b-v1.0")
    refiner = AutoModelForCausalLM.from_pretrained(
        "airesearch/wizard-7b-v1.0",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    ).eval()
    print("Wizard model loaded.")
    print("\nAll models loaded successfully!")


# ---------------------------
# Helper Functions (adapted for server use)
# ---------------------------
def load_image_from_bytes(image_bytes):
    """
    Loads an image from bytes data and preprocesses it.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        return img
    except Exception as e:
        print(f"❌ Error loading image from bytes: {e}")
        return None

def get_clip_features(img):
    """Extracts image and text features using CLIP and finds top matching reports."""
    report_db = [
        "Consolidation RLL", "No acute findings",
        "Mild cardiomegaly", "Pneumonia",
        "Interstitial changes", "Pleural effusion",
        "Pulmonary edema", "Atelectasis"
    ]
    # Process image
    img_inputs = clip_proc(images=img, return_tensors="pt").to(device)
    img_feat = clip_model.get_image_features(**img_inputs)
    img_feat = F.normalize(img_feat, dim=-1)

    # Process text database
    txt_inputs = clip_proc(text=report_db, return_tensors="pt", padding=True).to(device)
    txt_feat = clip_model.get_text_features(**txt_inputs)
    txt_feat = F.normalize(txt_feat, dim=-1)

    # Calculate similarity scores
    scores = (txt_feat @ img_feat.T).squeeze()
    retrieved = [report_db[i] for i in scores.topk(3).indices.tolist()]
    return retrieved

def generate_blip_caption(img):
    """Generates a caption for the given image using BLIP."""
    inputs = blip_proc(images=img, return_tensors="pt").to(device)
    cap_ids = blip_model.generate(**inputs)
    caption = blip_proc.decode(cap_ids[0], skip_special_tokens=True)
    return caption

def generate_text(model, tokenizer, prompt, max_tokens=150):
    """Generates text using a given language model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ---------------------------
# Flask API Endpoint
# ---------------------------
@app.route('/generate_report', methods=['POST'])
def generate_report_endpoint():
    """
    API endpoint to receive an image, process it, and return a medical report.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type. Only JPG/PNG images are allowed.'}), 400

    try:
        img_bytes = file.read()
        img = load_image_from_bytes(img_bytes)

        if img is None:
            return jsonify({'error': 'Could not process image.'}), 500

        # --- CLIP: Feature Extraction and Retrieval ---
        retrieved_reports = get_clip_features(img)

        # --- BLIP: Captioning ---
        caption = generate_blip_caption(img)

        # --- Falcon: Draft Generation ---
        prompt_draft = f"Caption: {caption}\nReports:\n- "+ "\n- ".join(retrieved_reports) + "\nDraft:"
        draft_report = generate_text(falcon, falcon_tok, prompt_draft, 150)

        # --- Wizard: Refined Report Generation ---
        prompt_refined = f"Caption: {caption}\nDraft: {draft_report}\nRefine this medical report to be more professional:"
        refined_report = generate_text(refiner, r_tok, prompt_refined, 100)

        return jsonify({'report': refined_report})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

# ---------------------------
# Main execution for Flask app
# ---------------------------
if __name__ == "__main__":
    # Load all models when the Flask application starts
    load_all_models()
    # Run the Flask app
    # Host on 0.0.0.0 to make it accessible from other machines/containers
    # Debug=True is for development only; set to False in production
    app.run(host='0.0.0.0', port=5000, debug=True)
