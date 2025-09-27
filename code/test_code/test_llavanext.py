import os
import json
import argparse
from tqdm import tqdm
import time
from PIL import Image
import numpy as np
import av
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

def get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return 'video'
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
        return 'image'
    else:
        raise ValueError(f"Unsupported file format: {ext} in file {file_path}")

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    if not frames:
        raise ValueError("Could not decode specified frames from the video.")
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def process_file(dataset_json_path: str, model, processor, result_suffix: str, device: str):
    json_filename = os.path.basename(dataset_json_path)
    result_json_path = os.path.join(
        os.path.dirname(dataset_json_path),
        f"{os.path.splitext(json_filename)[0]}{result_suffix}"
    )

    if os.path.exists(result_json_path):
        print(f"[INFO] Result file '{os.path.basename(result_json_path)}' already exists. Skipping.")
        return

    try:
        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Failed to read or parse JSON file {dataset_json_path}: {e}")
        return

    all_results = []
    base_path = os.path.dirname(dataset_json_path)

    for item in tqdm(data, desc=f"  Inferring on {json_filename}"):
        start_time = time.time()
        model_output = "N/A"
        
        try:
            prompt_text = item['conversations'][0]['value']
            ground_truth = item['conversations'][1]['value']
            media_path_key = 'image' if 'image' in item else 'video'
            media_relative_path = item.get(media_path_key)
            if not media_relative_path:
                raise ValueError("JSON item is missing 'image' or 'video' key.")
            
            media_full_path = os.path.join(base_path, media_relative_path)
            if not os.path.exists(media_full_path):
                raise FileNotFoundError(f"Media file not found: {media_full_path}")
            
            media_type = get_media_type(media_full_path)
            clean_prompt = prompt_text.replace("<image>", "").replace("<video>", "").strip()

            conversation = [
                {"role": "user", "content": [
                    {"type": "text", "text": clean_prompt},
                    {"type": media_type},
                ]},
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

            if media_type == 'image':
                raw_image = Image.open(media_full_path)
                inputs = processor(text=prompt, images=raw_image, return_tensors='pt').to(device, torch.float16)
            
            elif media_type == 'video':
                container = av.open(media_full_path)
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / 8).astype(int)
                clip = read_video_pyav(container, indices)
                inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(device, torch.float16)

            output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            decoded_output = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
            
            assistant_marker = "ASSISTANT:"
            if assistant_marker in decoded_output:
                model_output = decoded_output.split(assistant_marker)[-1].strip()
            else:
                model_output = decoded_output

        except Exception as e:
            model_output = f"ERROR: {str(e)}"
        
        end_time = time.time()
        all_results.append({
            "id": item.get('id', 'N/A'), 
            "prompt": prompt_text, 
            "model_output": model_output, 
            "ground_truth": ground_truth, 
            "processing_time_seconds": round(end_time - start_time, 2)
        })

    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f" Processing complete. Results saved to: {result_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch inference with a local LLaVA-NeXT-Video model.")
    parser.add_argument("--model-path", required=True, help="Full path to the local model directory.")
    parser.add_argument("--result-suffix", required=True, help="Suffix for the generated result files (e.g., '_result.json').")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on (e.g., 'cuda:0' or 'cpu').")
    args = parser.parse_args()

    try:
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
        ).to(args.device)
        processor = LlavaNextVideoProcessor.from_pretrained(args.model_path)
        print(" Model and processor loaded successfully.")
    except Exception as e:
        print(f"Failed to load model from '{args.model_path}'. Error: {e}")
        return

    current_dir = os.getcwd()
    source_json_files = [
        f for f in os.listdir(current_dir) 
        if f.endswith('.json') and not f.endswith(args.result_suffix)
    ]

    if not source_json_files:
        print(f"\n[INFO] No source JSON files: {current_dir}")
    else:
        for json_filename in sorted(source_json_files):
            process_file(
                dataset_json_path=os.path.join(current_dir, json_filename),
                model=model,
                processor=processor,
                result_suffix=args.result_suffix,
                device=args.device
            )
            

if __name__ == "__main__":
    main()