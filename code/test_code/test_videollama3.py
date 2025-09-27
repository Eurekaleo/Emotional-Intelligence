import os
import json
import argparse
from tqdm import tqdm
import time
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

LEVEL_DIRS = ["level1", "level2", "level3"]
GENERIC_RESULT_PATTERN = "_result.json"

def get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return 'video'
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
        return 'image'
    else:
        raise ValueError(f"Unsupported file extension: {ext} in file {file_path}")

def process_task(task_path: str, model, processor, result_suffix: str, device: str):


    source_json_files = [
        f for f in os.listdir(task_path)
        if f.endswith('.json') and GENERIC_RESULT_PATTERN not in f
    ]

    if not source_json_files:
        return

    for json_filename in source_json_files:
        dataset_json_path = os.path.join(task_path, json_filename)
        result_json_path = os.path.join(task_path, f"{os.path.splitext(json_filename)[0]}{result_suffix}")

        if os.path.exists(result_json_path):
            continue

        try:
            with open(dataset_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            continue

        all_results = []
        for item in tqdm(data, desc=f"  Processing {json_filename}"):
            start_time = time.time()
            model_output = "N/A"

            try:
                prompt_text = item['conversations'][0]['value']
                ground_truth = item['conversations'][1]['value']
                media_path_key = 'image' if 'image' in item else 'video'
                media_relative_path = item.get(media_path_key)

                if not media_relative_path:
                    raise ValueError("Missing 'image' or 'video' key in JSON entry.")

                media_full_path = os.path.join(task_path, media_relative_path)
                if not os.path.exists(media_full_path):
                    raise FileNotFoundError(f"Media file not found: {media_full_path}")

                media_type = get_media_type(media_full_path)
                clean_prompt = prompt_text.replace("<image>", "").replace("<video>", "").strip()

                if media_type == 'image':
                    media_content = {"type": "image", "image": {"image_path": media_full_path}}
                else:
                    media_content = {"type": "video", "video": {"video_path": media_full_path, "fps": 1, "max_frames": 128}}
                
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            media_content,
                            {"type": "text", "text": clean_prompt},
                        ]
                    },
                ]

                inputs = processor(conversation=conversation, return_tensors="pt")
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                response = processor.batch_decode(output, skip_special_tokens=True)[0].strip()

                last_turn_markers = ['assistant\n', 'assistant:']
                for marker in last_turn_markers:
                    if marker in response.lower():
                        response = response.split(marker, 1)[-1].strip()
                        break
                model_output = response

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


def main():
    parser = argparse.ArgumentParser(description="Batch processing script for a local multimodal model using the Transformers library.")

    parser.add_argument("--model-path", required=True, help="Full path to the local model directory")
    parser.add_argument("--result-suffix", required=True, help="Suffix for the generated result files")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model")

    args = parser.parse_args()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=args.device
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        return

    for level_dir in LEVEL_DIRS:
        level_path = os.path.join(os.getcwd(), level_dir)
        if not os.path.isdir(level_path):
            continue

        task_dirs = sorted([d.path for d in os.scandir(level_path) if d.is_dir()])
        for task_path in task_dirs:
            process_task(task_path, model, processor, args.result_suffix, args.device)


if __name__ == "__main__":
    main()