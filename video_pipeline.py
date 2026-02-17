import torch
from qwen_inference import inference
from transformers import  Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import numpy as np
import argparse

def main(video_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = """
    Identify the object being lifted in this video. Provide in detail the name of the object.
    """

    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map = device)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    output_text = inference(model, processor, video_path, prompt, device)

    print(f"Extracted item: {output_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run video inference")

    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video")

    args = parser.parse_args()

    main(args.video_path)