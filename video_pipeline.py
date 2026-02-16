import numpy as np
from qwen_inference import inference

def main():
    video_path = r"C:\Users\T14\Documents\Local_Git\Semantics\non_anonymized\non_anonymized\video_env_1\ap_0016.mp4"
    prompt = "Highlight the item that is being lifted in the video?"
    output_text = inference(video_path, prompt)
    print(output_text)

if __name__ == "__main__":
    main()