# app.py

import torch
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel
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
    print(f"🖥️ Using device: {device}")

    # Step 1: Load image
    image_path = select_image()
    if not image_path:
        print("❌ No image selected.")
        return
    img = load_image(image_path)

    # Step 2: CLIP - image and text embedding
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

    # Step 3: BLIP - image captioning
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    blip_inputs = blip_processor(images=img, return_tensors="pt").to(device)
    blip_output = blip_model.generate(**blip_inputs)
    caption = blip_processor.decode(blip_output[0], skip_special_tokens=True)

    print("\n🖼️ Generated Caption:", caption)

    # Step 4: Falcon-7B - generate draft report
    falcon_model_name = "tiiuae/falcon-7b-instruct"
    falcon_tok = AutoTokenizer.from_pretrained(falcon_model_name)
    falcon_model = AutoModelForCausalLM.from_pretrained(
        falcon_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device).eval()

    draft_prompt = (
        f"🧠 Caption: {caption}\n"
        f"📚 Retrieved Reports:\n" + "\n".join(f"- {report}" for report in retrieved_top) +
        "\n\nGenerate a clinical-style radiology report:"
    )

    falcon_inputs = falcon_tok(draft_prompt, return_tensors="pt").to(device)
    falcon_outputs = falcon_model.generate(
        **falcon_inputs, max_new_tokens=256, do_sample=True, temperature=0.7
    )
    draft_report = falcon_tok.decode(falcon_outputs[0], skip_special_tokens=True)

    print("\n📝 Draft Report:\n", draft_report)

    # Step 5: GPT-2 - refine final report
    gpt2_tok = GPT2Tokenizer.from_pretrained("distilgpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

    final_prompt = (
        f"Caption: {caption}\n"
        f"Draft: {draft_report}\n"
        f"Refine this into a structured radiology report:"
    )

    gpt2_inputs = gpt2_tok(final_prompt, return_tensors="pt").to(device)
    gpt2_outputs = gpt2_model.generate(
        **gpt2_inputs, max_new_tokens=150, do_sample=True
    )
    final_report = gpt2_tok.decode(gpt2_outputs[0], skip_special_tokens=True)

    print("\n✅ Final Radiology Report:\n", final_report)


if __name__ == "__main__":
    main()
