from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import time
import pandas as pd
from tqdm import tqdm

# ---------------- config ----------------
BATCH_SIZE = 1
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# System/User prompts (English, strict, line-focused)
PROMPT_SYSTEM = (
    "You are a handwriting transcription assistant (HTR). Read the provided IMAGE of a single text line "
    "and output only the exact transcription in Norwegian.\n"
    "Rules:\n"
    "- Do not add explanations, descriptions, metadata, labels, or translations.\n"
    "- If a letter/word is unclear, transcribe as accurately as possible without inventing extra words.\n"
    "- Preserve original spelling, abbreviations, capitalization, and punctuation.\n"
    "- Output a single line of plain text with no newline characters."
)
PROMPT_USER = "Transcribe this handwritten line. Respond only with the text in Norwegian."

# Data (images already cropped)
dataset_filename = 'norhand/test_data/textlines.csv'   # columns: file, textline, bbox (bbox is unused)
dataset_main_path = 'norhand/test_data/textlines'

# ---------------- load data ----------------
dataset = pd.read_csv(dataset_filename)
if 'prediction' not in dataset.columns:
    dataset['prediction'] = ""

print(f"Total rows: {len(dataset)} | Batch size: {BATCH_SIZE}")

# ---------------- speed/precision knobs ----------------
# Avoid flash-attn headaches; SDPA is stable and fast on recent PyTorch
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"

# Allow TF32 (Ampere+ GPUs) for a bit more speed without quality loss
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---------------- model & processor ----------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,          # If your GPU prefers fp16, switch to torch.float16
    attn_implementation="sdpa",
    device_map="auto",
)

# Give the vision tower a bit more pixel budget for fine strokes
min_pixels = 256 * 28 * 28
max_pixels = 2048 * 28 * 28   # bump to 2048–3072 if VRAM allows
processor = AutoProcessor.from_pretrained(
    MODEL_NAME, min_pixels=min_pixels, max_pixels=max_pixels
)

# ---------------- run ----------------
for start in tqdm(range(0, len(dataset), BATCH_SIZE), total=len(dataset)):
    end = min(start + BATCH_SIZE, len(dataset))
    batch_df = dataset.iloc[start:end]

    # Build messages (system + user) and pack images
    messages_list = []
    for idx, row in batch_df.iterrows():
        file_name = row['file']
        file_path = os.path.join(dataset_main_path, file_name)
        gt_text = row.get('textline', "")  # ground truth from CSV

        print(f"[{idx}] file: {file_name}")
        print(f"  GT: {gt_text}")  # <-- print ground truth here

        messages = [
            {"role": "system", "content": [{"type": "text", "text": PROMPT_SYSTEM}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": file_path},
                    {"type": "text", "text": PROMPT_USER},
                ],
            },
        ]
        messages_list.append(messages)

    # Prepare text + image batches (no videos at all)
    texts, image_inputs_list = [], []
    for messages in messages_list:
        t = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(t)
        imgs, _ = process_vision_info(messages)
        image_inputs_list.append(imgs)

    inputs = processor(
        text=texts,
        images=image_inputs_list,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Inference
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            num_beams=5,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    outputs = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    outputs = [o.strip().splitlines()[0] if "\n" in o else o.strip() for o in outputs]

    # Print ground truth + prediction together
    for (row_idx, row), pred in zip(batch_df.iterrows(), outputs):
        gt = row.get('textline', "")
        print(f"  PRED: {pred}")
        print("----")
        dataset.at[row_idx, 'prediction'] = pred

# Save
out_path = 'norhand/test_data/textlines_predictions.csv'
dataset.to_csv(out_path, index=False)
print(f"Saved predictions -> {out_path}")
