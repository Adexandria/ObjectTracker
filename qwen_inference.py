import torch
from decord import VideoReader
from transformers import  Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np


def inference(video_path, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video = VideoReader(video_path)

    total_frames = len(video)

    # Simulate inference results
    messages =[
        {"role": "user",
        "content": [
             { 
                "video": video_path,
                "total_pixels": 20480 * 32 * 32,
                "min_pixels": 64 * 32 * 32,
                "max_frames": total_frames,
                "samples_fps": 2},
            {"type": "text", "content": prompt}
        ],
        }
    ]
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",torch_dtype=torch.float16,device_map = device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)

    if video_kwargs:
        for key, value in video_kwargs.items():
            if isinstance(value, list) and len(value) > 0:
                video_kwargs[key] = value[0]
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,**video_kwargs, return_tensors="pt")

    inputs = inputs.to(device)

    output_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_token=False, clean_up_tokenization_spaces=False)
    return output_text[0]