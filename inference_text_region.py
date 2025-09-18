from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import pandas as pd
from tqdm import tqdm

# ---------------- config ----------------
BATCH_SIZE = 1
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# System/User prompts (text REGION; preserve exact line breaks)
PROMPT_SYSTEM = (
    "You are a handwriting transcription assistant (HTR). Read the provided IMAGE containing a text REGION with "
    "multiple handwritten lines, and output ONLY the transcription in Norwegian.\n"
    "Rules:\n"
    "- Preserve the original line breaks exactly as they appear in the image (top-to-bottom).\n"
    "- Separate lines using a single newline character (\\n). Do not add extra blank lines at the start or end.\n"
    "- Do not number lines or add explanations, labels, quotes, or metadata.\n"
    "- Preserve original spelling, abbreviations, capitalization, hyphenation, and punctuation; do not translate.\n"
    "- If a word/letter is unclear, transcribe as faithfully as possible without inventing words."
)
PROMPT_USER = (
    "Transcribe the handwritten text REGION. Respond with the lines in reading order separated by \\n, "
    "preserving the line breaks exactly as in the image. Do not add or remove lines."
)

# ---------------- data ----------------
# CSV columns expected: file (image filename). If you have ground-truth, put it in 'textlines' (optional).
dataset_filename = 'norhand/test_data/regions.csv'
dataset_main_path = 'norhand/test_data/regions'

# ---------------- load data ----------------
dataset = pd.read_csv(dataset_filename)
if 'prediction' not in dataset.columns:
    dataset['prediction'] = ""

print(f"Total rows: {len(dataset)} | Batch size: {BATCH_SIZE}")

# ---------------- speed/precision knobs ----------------
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"
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

# Larger pixel budget so the vision tower renders bigger images (tune for your VRAM)
min_pixels = 256 * 28 * 28
max_pixels = 3072 * 28 * 28
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
        gt_text = row.get('textlines', "")  # optional ground-truth

        print(f"[{idx}] file: {file_name}")
        if isinstance(gt_text, str) and gt_text:
            print("  GT (csv):")
            print(gt_text)

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

    # Prepare text + image batches (no videos)
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

    # Inference — allow enough tokens for multi-line regions
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,                # generous for regions
            do_sample=False,
            num_beams=5,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # Remove prompt and decode
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Post-process: keep EXACT line breaks, trim only outer whitespace
    outputs = []
    for o in decoded:
        text = o.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        # Avoid collapsing internal blank lines—preserve as is
        # Also strip trailing spaces on each line (but keep internal spaces)
        lines = [ln.rstrip() for ln in text.split("\n")] if text else []
        outputs.append("\n".join(lines))

    # Print and save
    for (row_idx, row), pred in zip(batch_df.iterrows(), outputs):
        print("  PRED:")
        print(pred)
        print("----")
        dataset.at[row_idx, 'prediction'] = pred

# Save
out_path = 'norhand/test_data/regions_predictions.csv'
dataset.to_csv(out_path, index=False)
print(f"Saved predictions -> {out_path}")
