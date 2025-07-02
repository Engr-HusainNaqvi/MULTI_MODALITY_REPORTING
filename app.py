# app.py

import torch
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)

def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    return file_path

def load_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    img.show()
    return img

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Running on: {device}")

    image_path = select_image()
    if not image_path:
        print("❌ No image selected.")
        return
    img = load_image(image_path)

    # --- CLIP: Visual-Text Embedding ---
    print("📥 Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    img_inputs = clip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**img_inputs)
    image_features = torch.nn.functional.normalize(image_features, dim=-1)

    retrieved_reports = [
        "There is evidence of right lower lobe consolidation.",
        "No acute cardiopulmonary abnormality.",
        "Mild cardiomegaly with clear lungs.",
        "Findings consistent with pneumonia.",
        "Chronic interstitial changes noted."
    ]

    text_inputs = clip_processor(text=retrieved_reports, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)

    cosine_scores = torch.matmul(text_features, image_features.T).squeeze()
    top_indices = torch.topk(cosine_scores, k=3).indices
    retrieved_top = [retrieved_reports[i] for i in top_indices]

    print("\n📚 Retrieved Reports:", retrieved_top)

    # --- BLIP: Image Captioning ---
    print("🖼️ Generating caption...")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    blip_inputs = blip_processor(images=img, return_tensors="pt").to(device)
    blip_output = blip_model.generate(**blip_inputs)
    caption = blip_processor.decode(blip_output[0], skip_special_tokens=True)

    print("\n📝 Caption:", caption)

    # --- GPT2-Large for both Draft and Final ---
    print("🧠 Loading GPT2-large for draft + final...")
    gpt2_model_name = "gpt2-large"
    gpt2_tok = AutoTokenizer.from_pretrained(gpt2_model_name)
    gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name).to(device)

    # Draft generation
    draft_prompt = (
        f"🧠 Caption: {caption}\n"
        f"📚 Retrieved Reports:\n" + "\n".join(f"- {r}" for r in retrieved_top) +
        "\n\nGenerate a clinical-style radiology report:"
    )
    draft_inputs = gpt2_tok(draft_prompt, return_tensors="pt").to(device)
    draft_outputs = gpt2_model.generate(
        **draft_inputs, max_new_tokens=200, do_sample=True, temperature=0.7
    )
    draft_report = gpt2_tok.decode(draft_outputs[0], skip_special_tokens=True)
    print("\n📝 Draft Report:\n", draft_report)

    # Refinement step
    final_prompt = (
        f"Caption: {caption}\n"
        f"Draft: {draft_report}\n"
        f"Refine this into a structured radiology report:"
    )
    final_inputs = gpt2_tok(final_prompt, return_tensors="pt").to(device)
    final_outputs = gpt2_model.generate(
        **final_inputs, max_new_tokens=150, do_sample=True
    )
    final_report = gpt2_tok.decode(final_outputs[0], skip_special_tokens=True)
    print("\n✅ Final Radiology Report:\n", final_report)

if __name__ == "__main__":
    main()
