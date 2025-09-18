from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import time
import pandas as pd

BATCH_SIZE = 2

# Load the dataset - has columns: file, textline, bbox
dataset_filename = 'norhand/test_data/textlines.csv'
dataset = pd.read_csv(dataset_filename)
dataset_main_path = 'norhand/test_data/textlines'

# Ensure column for predictions exists
if 'prediction' not in dataset.columns:
    dataset['prediction'] = ""

# Hard-disable FlashAttention2 so Transformers won't try to import it
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)

# Optional: tune visual token budget (keep if you need it)
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    model_name, min_pixels=min_pixels, max_pixels=max_pixels
)

print(f"Total rows: {len(dataset)} | Batch size: {BATCH_SIZE}")

for start in range(0, len(dataset), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(dataset))
    batch_df = dataset.iloc[start:end]

    # Build messages per sample
    messages_list = []
    for idx, row in batch_df.iterrows():
        file_name = row['file']
        textline = row.get('textline', "")
        bbox_coords = row.get('bbox', "")
        print(f"[{idx}] file: {file_name} | textline: {textline} | bbox: {bbox_coords}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": os.path.join(dataset_main_path, file_name)},
                    {
                        "type": "text",
                        "text": (
                            "Read the handwritten text in the image. "
                            "The language is Norwegian. Only output the text without any other explanation."
                        ),
                    },
                ],
            }
        ]
        messages_list.append(messages)

    # Turn each conversation into a chat template string, and collect image/video inputs
    texts = []
    image_inputs_list = []
    video_inputs_list = []
    for messages in messages_list:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        image_inputs, video_inputs = process_vision_info(messages)
        image_inputs_list.append(image_inputs)
        video_inputs_list.append(video_inputs)

    # Flatten image/video inputs as the processor expects a list aligned with texts
    # (Each element corresponds to one sample)
    inputs = processor(
        text=texts,
        images=image_inputs_list,
        videos=video_inputs_list,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Inference
    print(f"Generating for rows {start}..{end-1} ...")
    start_time = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,           # deterministic; set True + temperature for sampling
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time
    print(f"Batch time: {elapsed:.2f}s")

    # Trim prompts off and decode
    trimmed = []
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
        trimmed.append(out_ids[len(in_ids):])
    outputs = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Store predictions back into the dataset
    for (row_idx, _), pred in zip(batch_df.iterrows(), outputs):
        dataset.at[row_idx, 'prediction'] = pred.strip()

# Save the results to a new CSV file
out_path = 'norhand/test_data/textlines_predictions.csv'
dataset.to_csv(out_path, index=False)
print(f"Saved predictions -> {out_path}")
