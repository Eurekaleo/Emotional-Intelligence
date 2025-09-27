import os
import json
import argparse
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

try:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import av
except ImportError as e:
    print(f"Original error: {e}")
    exit(1)

# --- Configuration ---
DEFAULT_MODEL_PATH = "example/model/Qwen2.5-VL-model"

def get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return 'video'
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
        return 'image'
    else:
        raise ValueError(f"Unsupported file format: {ext} in file {file_path}")

def run_inference_on_file(
    json_path: str,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    result_suffix: str,
    fps: float,
    max_pixels: int,
    total_pixels: Optional[int],
    gen_tokens: int
):


    result_json_path = f"{os.path.splitext(json_path)[0]}{result_suffix}"

    if os.path.exists(result_json_path):
        print(f"  [INFO] Result file '{os.path.basename(result_json_path)}' already exists. Skipping.")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Could not read or parse JSON file {json_path}: {e}")
        return
    from qwen_vl_utils import process_vision_info

    all_results = []
    for item in tqdm(data, desc=f"  Inferring on {os.path.basename(json_path)}"):
        start_time = time.time()
        model_output = "N/A"
        
        try:
            prompt_text = item['conversations'][0]['value']
            ground_truth = item['conversations'][1]['value']
            media_path_key = 'image' if 'image' in item else 'video'
            media_relative_path = item.get(media_path_key)

            if not media_relative_path:
                raise ValueError("JSON entry is missing 'image' or 'video' key.")

            base_dir = os.path.dirname(json_path)
            media_full_path = os.path.join(base_dir, media_relative_path)
            
            if not os.path.exists(media_full_path):
                raise FileNotFoundError(f"Media file not found: {media_full_path}")
            
            media_type = get_media_type(media_full_path)
            clean_prompt = prompt_text.replace("<image>", "").replace("<video>", "").strip()

            content: List[Dict[str, Any]] = []
            media_abs_path = os.path.abspath(media_full_path)
            
            if media_type == 'image':
                content.append({"type": "image", "image": media_abs_path})
            else:  # video
                video_item = {
                    "type": "video",
                    "video": media_abs_path,
                    "fps": float(fps),
                    "max_pixels": int(max_pixels),
                }
                if total_pixels is not None and total_pixels > 0:
                    video_item["total_pixels"] = int(total_pixels)
                content.append(video_item)

            content.append({"type": "text", "text": clean_prompt})
            messages = [{"role": "user", "content": content}]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt", **video_kwargs,
            )
            inputs = inputs.to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=gen_tokens, do_sample=False)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            model_output = (output_text[0] if output_text else "").strip()

        except Exception as e:
            model_output = f"ERROR: {str(e)}\n{traceback.format_exc()}"
        
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


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Batch Inference (High-Performance Mode)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the local model directory.")
    parser.add_argument("--result-suffix", default="_result.json", help="Suffix for result files.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame rate for video sampling.")
    parser.add_argument("--max-pixels", type=int, default=360*420, help="Maximum pixels per frame.")
    parser.add_argument("--total-pixels", type=int, default=0, help="Total pixel limit for a video (0 for unlimited).")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum number of new tokens to generate.")
    args = parser.parse_args()

    if not args.model_path or args.model_path == "path/to/your/Qwen2.5-VL-model":
        exit(1)

    print(f"Model Path: {args.model_path}")

    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
        print("Flash Attention 2 detected. Using for better performance.")
    except ImportError:
        attn_implementation = "eager"
        print("Flash Attention 2 not found. ")

    print(f"Loading model with bfloat16 + {attn_implementation}...")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map="auto",
    )
    
    processor = AutoProcessor.from_pretrained(args.model_path)
    current_dir = os.getcwd()
    source_files = [
        f for f in os.listdir(current_dir)
        if f.endswith('.json') and not f.endswith(args.result_suffix)
    ]

    if not source_files:
        print(f"\nNo source JSON files.")
    else:
        print(f"\nFound {len(source_files)} JSON file(s) to process.")
        
        for json_filename in sorted(source_files):
            json_full_path = os.path.join(current_dir, json_filename)
            run_inference_on_file(
                json_full_path, model, processor, args.result_suffix,
                fps=args.fps, max_pixels=args.max_pixels,
                total_pixels=(args.total_pixels if args.total_pixels > 0 else None),
                gen_tokens=args.max_new_tokens
            )
            

if __name__ == "__main__":
    main()