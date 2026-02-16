import numpy as np
from qwen_inference import inference

def main():
    video_path = r"non_anonymized\non_anonymized\video_env_1\ap_0016.mp4"
    prompt = "Identify the person that is interacting with the object in the video and if the interaction is lifting or dropping, the format of output should be like {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"Person\"}"
    output_text = inference(video_path, prompt)
    print(output_text)

if __name__ == "__main__":
    main()