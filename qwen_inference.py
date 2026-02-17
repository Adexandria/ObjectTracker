from qwen_vl_utils import process_vision_info
import re
import numpy as np

def inference(model, processor, video_path, prompt, device):

    # Simulate inference results
    messages =[
        {"role": "user",
        "content": [
             { 
               "video" : video_path,
                "max_pixels": 360 *420,
                "fps": 2.0
               },
            {"type": "text", "text": prompt}
        ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info([messages])

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]

    output_text = processor.batch_decode(generated_ids, skip_special_token=True, clean_up_tokenization_spaces=True)

    return extract_item(output_text[0])
    

def extract_item(response):

    clean_text = response.split("<|im_end|>")[0].strip()

    patterns = [
        r"is (?:a|an|the) (.*)", 
        r"lifted is (.*)",
        r"object is (.*)"
    ]
    for pattern in patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            return match.group(1).strip(" .")
    
    return clean_text.strip("")