from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import time
import pandas as pd

BATCH_SIZE = 1

# ---- Prompts ----
PROMPT_SYSTEM = (
    "You are a handwriting transcription assistant (HTR). Carefully read the handwritten text in the image and return only the exact transcription in Norwegian."
    "teksten i bildet og lever kun den nøyaktige transkripsjonen på norsk.\n"
    "Rules:\n"
    "- Do not add explanations, descriptions, metadata, or translations.\n"
    "- If a word or letter is unclear, transcribe it as accurately as possible without guessing additional words.\n"
    "- Preserve the original spelling, abbreviations, and punctuation as they appear in the handwriting."
    "- Do not use quotation marks, markdown, or labels. Output only the text."
)
PROMPT_USER = "Transcribe the handwriting in the image. Respond only with the text (in Norwegian)."

# Load the dataset - has columns: file, textline, bbox
dataset_filename = 'norhand/test_data/textlines.csv'
dataset = pd.read_csv(dataset_filename)
dataset_main_path = 'norhand/test_data/textlines'

# Ensure column for predictions exists
if 'prediction' not in dataset.columns:
    dataset['prediction'] = ""

# Disable FlashAttention2
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)

# Processor (keep min/max pixels if you need them)
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    model_name, min_pixels=min_pixels, max_pixels=max_pixels
)

print(f"Total rows: {len(dataset)} | Batch size: {BATCH_SIZE}")

for start in range(0, len(dataset), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(dataset))
    batch_df = dataset.iloc[start:end]

    # Build conversations and collect vision inputs
    messages_list = []
    for idx, row in batch_df.iterrows():
        file_name = row['file']
        textline = row.get('textline', "")
        bbox_coords = row.get('bbox', "")
        print(f"[{idx}] file: {file_name} | textline: {textline} | bbox: {bbox_coords}")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": PROMPT_SYSTEM}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": os.path.join(dataset_main_path, file_name)},
                    {"type": "text", "text": PROMPT_USER},
                ],
            },
        ]
        messages_list.append(messages)

    # Prepare text + image/video batches
    texts = []
    image_inputs_list = []
    video_inputs_list = []
    for messages in messages_list:
        t = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(t)
        imgs, vids = process_vision_info(messages)
        image_inputs_list.append(imgs)
        video_inputs_list.append(vids)  # will be None for images-only

    # ---- only include `videos` if present ----
    has_any_video = any(v is not None and v != [] for v in video_inputs_list)
    processor_kwargs = dict(
        text=texts,
        images=image_inputs_list,
        padding=True,
        return_tensors="pt",
    )
    if has_any_video:
        processor_kwargs["videos"] = video_inputs_list

    inputs = processor(**processor_kwargs).to("cuda")

    # Inference
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

    # Trim prompts and decode
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    outputs = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print("Predictions:")
    for o in outputs:
        print(f"  -> {o.strip()}")

    # Save back
    for (row_idx, _), pred in zip(batch_df.iterrows(), outputs):
        dataset.at[row_idx, 'prediction'] = pred.strip()

# Save
out_path = 'norhand/test_data/textlines_predictions_batch4.csv'
dataset.to_csv(out_path, index=False)
print(f"Saved predictions -> {out_path}")
