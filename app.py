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
warnings.filterwarnings('ignore')

# 🔁 Device setup with memory optimization
device = "cuda" if torch.cuda.is_available() else "cpu"
# Enable faster matmul on modern GPUs if CUDA is available
if 'cuda' in device:
    torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------
# 1️⃣ Image Loading + Feature Extraction + Retrieval
# ---------------------------
def load_image(image_path):
    """
    Loads an image from a given path and preprocesses it.
    In a web application, this might come from a file upload.
    """
    try:
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        return img
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

@torch.no_grad()
def load_clip_model():
    """Loads the CLIP model and processor."""
    print("🔄 Loading CLIP model...")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        device_map="auto",
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32
    )
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def get_clip_features(img, clip_model, clip_proc, report_db):
    """Extracts image and text features using CLIP and finds top matching reports."""
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

# ---------------------------
# 2️⃣ Vision Captioning (BLIP)
# ---------------------------
@torch.no_grad()
def load_blip_model():
    """Loads the BLIP model and processor."""
    print("\n🔄 Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        device_map="auto",
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32
    )
    return processor, model

def generate_blip_caption(img, blip_proc, blip_model):
    """Generates a caption for the given image using BLIP."""
    inputs = blip_proc(images=img, return_tensors="pt").to(device)
    cap_ids = blip_model.generate(**inputs)
    caption = blip_proc.decode(cap_ids[0], skip_special_tokens=True)
    return caption

# ---------------------------
# 3️⃣ Draft + Refiner Agents
# ---------------------------
def generate_text(model, tokenizer, prompt, max_tokens=150):
    """Generates text using a given language model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def load_falcon_model():
    """Loads the Falcon-7B-Instruct model."""
    print("\n🔄 Loading Falcon-7B model...")
    falcon_tok = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    falcon = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b-instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    ).eval()
    return falcon, falcon_tok

def load_wizard_model():
    """Loads the Wizard-7B-v1.0 model."""
    print("\n🔄 Loading Wizard-7B model...")
    r_tok = AutoTokenizer.from_pretrained("airesearch/wizard-7b-v1.0")
    refiner = AutoModelForCausalLM.from_pretrained(
        "airesearch/wizard-7b-v1.0",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    ).eval()
    return refiner, r_tok

# Main execution flow
if __name__ == "__main__":
    # --- Image Input ---
    # For a standalone script, you'd typically pass an image path as an argument
    # or prompt the user for one. For demonstration, let's assume an image path.
    # Replace 'path/to/your/image.jpg' with an actual image file path.
    image_path = 'sample_chest_xray.jpg' # <<< IMPORTANT: Change this to your image path
    img = load_image(image_path)

    if img is None:
        print("Exiting due to image loading error.")
    else:
        # --- CLIP: Feature Extraction and Retrieval ---
        clip_model, clip_proc = load_clip_model()
        report_db = [
            "Consolidation RLL", "No acute findings",
            "Mild cardiomegaly", "Pneumonia",
            "Interstitial changes", "Pleural effusion",
            "Pulmonary edema", "Atelectasis"
        ]
        retrieved_reports = get_clip_features(img, clip_model, clip_proc, report_db)
        print("\n📌 Top 3 matching reports:")
        for i, report in enumerate(retrieved_reports, 1):
            print(f"{i}. {report}")

        # Clear CLIP from memory
        del clip_model, clip_proc
        torch.cuda.empty_cache()

        # --- BLIP: Captioning ---
        blip_proc, blip_model = load_blip_model()
        caption = generate_blip_caption(img, blip_proc, blip_model)
        print(f"\n📝 Image caption: {caption}")

        # Clear BLIP from memory
        del blip_proc, blip_model
        torch.cuda.empty_cache()

        # --- Falcon: Draft Generation ---
        falcon, falcon_tok = load_falcon_model()
        prompt_draft = f"Caption: {caption}\nReports:\n- "+ "\n- ".join(retrieved_reports) + "\nDraft:"
        draft_report = generate_text(falcon, falcon_tok, prompt_draft, 150)
        print(f"\n✍️ Draft report:\n{draft_report}")

        # Clear Falcon from memory
        del falcon, falcon_tok
        torch.cuda.empty_cache()

        # --- Wizard: Refined Report Generation ---
        refiner, r_tok = load_wizard_model()
        prompt_refined = f"Caption: {caption}\nDraft: {draft_report}\nRefine this medical report to be more professional:"
        refined_report = generate_text(refiner, r_tok, prompt_refined, 100)
        print(f"\n💎 Refined report:\n{refined_report}")

        # Clear Wizard from memory
        del refiner, r_tok
        torch.cuda.empty_cache()

        # ✅ Final Output
        print("\n✨ Final Structured Report:")
        print("-"*50)
        print(refined_report)
        print("-"*50)
