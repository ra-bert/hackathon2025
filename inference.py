from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import time
import pandas as pd

#BATCH_SIZE = 1

# Load the dataset - has column file	textline	bbox
dataset_filename = 'norhand/test_data/textlines.csv'
dataset = pd.read_csv(dataset_filename)
dataset_path = 'norhand/test_data/textlines'

# Hard-disable FlashAttention2 so Transformers won't try to import it
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-32B-Instruct", dtype="auto", device_map="auto"
# )
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
# Use SDPA instead of FlashAttention2 and rename torch_dtype -> dtype
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,                 # <— was torch_dtype
    attn_implementation="sdpa",           # <— force SDPA
    device_map="auto",
)

# default processor
processor = AutoProcessor.from_pretrained(model_name)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
# Make it ready for a reading all dataset textline 
for idx in range(len(dataset)):
    file_name = dataset.iloc[idx]['file']
    text = dataset.iloc[idx]['textline']
    # bbox_coords = dataset.iloc[idx]['bbox']
    print(f"Processing idx {idx}, file_name: {file_name}, text: {text}, line: {line}, bbox_coords: {bbox_coords}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": file_name},
                {"type": "text", "text": "Read the handwritten text in the image. The language is Norwegian. Only output the text without any other explanation."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    start_time = time.time()

    # Inference: Generation of the output
    print("Start to generate...")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output_text)
    end_time = time.time()
    print(f"Time used: {end_time - start_time} seconds")
    dataset.at[idx, 'prediction'] = output_text[0]
# Save the results to a new CSV file
dataset.to_csv('norhand/test_data/textlines_predictions.csv', index=False)