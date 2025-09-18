from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import time
import pandas as pd

# ------------------ config ------------------
BATCH_SIZE = 2
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

dataset_filename = 'norhand/test_data/textlines.csv'   # columns: file, textline, bbox
dataset_main_path = 'norhand/test_data/textlines'

# Prompts
SYSTEM_PROMPT = (
    "Du er en assistent for transkripsjon av håndskrift (HTR). Les nøye den håndskrevne "
    "teksten i bildet og lever kun den nøyaktige transkripsjonen på norsk.\n"
    "Regler:\n"
    "- Ikke legg til forklaringer, beskrivelser, metadata, eller oversettelser.\n"
    "- Hvis et ord/bokstav er uklar, transkriber så godt som mulig uten å gjette ekstra ord.\n"
    "- Behold original staving, forkortelser og tegnsetting slik de står i håndskriften.\n"
    "- Ikke bruk anførselstegn, markdown eller etiketter. Svar kun med teksten."
)
USER_PROMPT = "Transkriber håndskriften i bildet. Svar kun med teksten (norsk)."

# Disable FlashAttention2 (stick to SDPA to avoid nvcc/wheel issues)
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"

# ------------------ load data ------------------
dataset = pd.read_csv(dataset_filename)
if 'prediction' not in dataset.columns:
    dataset['prediction'] = ""

# ------------------ load model & processor ------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)

# Optionally bound visual tokens for VRAM
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    model_name, min_pixels=min_pixels, max_pixels=max_pixels
)

print(f"Total rows: {len(dataset)} | Batch size: {BATCH_SIZE}")

# ------------------ batching loop ------------------
for start in range(0, len(dataset), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(dataset))
    batch_df = dataset.iloc[start:end]

    # Build messages per sample (system + user[image,text])
    messages_list = []
    for idx, row in batch_df.iterrows():
        file_path = os.path.join(dataset_main_path, row['file'])
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": file_path},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]
        messages_list.append(messages)

    # Convert conversations to chat template + gather image inputs
    texts = []
    image_inputs_list = []
    video_inputs_list = []  # images-only, but we’ll keep the list to check presence
    for messages in messages_list:
        t = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(t)
        imgs, vids = process_vision_info(messages)
        image_inputs_list.append(imgs)
        video_inputs_list.append(vids)  # will be None for pure images

    # Only pass videos if any are present
    has_any_video = any(v is not None and v != [] for v in video_inputs_list)
    proc_kwargs = dict(
        text=texts,
        images=image_inputs_list,
        padding=True,
        return_tensors="pt",
    )
    if has_any_video:
        proc_kwargs["videos"] = video_inputs_list

    inputs = processor(**proc_kwargs).to("cuda")

    # ------------------ inference ------------------
    print(f"Generating for rows {start}..{end-1} ...")
    t0 = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    print(f"Batch time: {time.time() - t0:.2f}s")

    # Trim prompts off and decode
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    outputs = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Save predictions
    for (row_idx, _), pred in zip(batch_df.iterrows(), outputs):
        dataset.at[row_idx, 'prediction'] = pred.strip()

# ------------------ save ------------------
out_path = 'norhand/test_data/textlines_predictions.csv'
dataset.to_csv(out_path, index=False)
print(f"Saved predictions -> {out_path}")
