import os
import json
import base64
from openai import OpenAI
from tqdm import tqdm
import time
import sys

# --- Configuration ---
MODEL_ENDPOINT = "http://localhost:8000/v1"
MODEL_NAME = "minicpm" 
RESULT_SUFFIX = "_minicpm_result.json"   

GENERIC_RESULT_PATTERN = "_result.json"

def get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return 'video'
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
        return 'image'
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def encode_media_to_base64(media_path: str) -> str:
    try:
        with open(media_path, "rb") as media_file:
            return base64.b64encode(media_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Media file not found at: {media_path}")
        raise
    except Exception as e:
        raise IOError(f"Could not read or encode file {media_path}: {e}")

def process_directory(client: OpenAI, model_name: str, result_suffix: str):
    current_dir = os.getcwd()
    source_json_files = [
        f for f in os.listdir(current_dir)
        if f.endswith('.json') and GENERIC_RESULT_PATTERN not in f
    ]

    if not source_json_files:
        print(f"[Info] No source JSON files to process in {current_dir}.")
        return

    for json_filename in source_json_files:
        process_single_json(current_dir, json_filename, client, model_name, result_suffix)

def process_single_json(directory: str, json_filename: str, client: OpenAI, model_name: str, result_suffix: str):
    dataset_json_path = os.path.join(directory, json_filename)
    result_json_path = os.path.join(directory, f"{os.path.splitext(json_filename)[0]}{result_suffix}")

    if os.path.exists(result_json_path):
        print(f"Result file '{os.path.basename(result_json_path)}' already exists, skipping.")
        return

    print(f"Reading and processing dataset: {json_filename}")
    try:
        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return

    all_results = []
    for item in tqdm(data, desc=f"  Processing {json_filename}", unit="item"):
        start_time = time.time()
        model_output = ""
        try:
            prompt = item['conversations'][0]['value']
            ground_truth = item['conversations'][1]['value']
            media_path_key = 'image' if 'image' in item else 'video'
            media_relative_path = item.get(media_path_key)
            if not media_relative_path:
                raise ValueError("JSON item is missing 'image' or 'video' key.")

            media_full_path = os.path.join(directory, media_relative_path)
        
            media_type = get_media_type(media_full_path)
            media_base64 = encode_media_to_base64(media_full_path)
            clean_prompt = prompt.replace("<image>", "").replace("<video>", "").strip()
            
            content = [{"type": "text", "text": clean_prompt}]
            if media_type == 'image':
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{media_base64}"}})
            else: # video
                content.append({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{media_base64}"}})
            
            messages = [{"role": "user", "content": content}]

            response = client.chat.completions.create(model=model_name, messages=messages, max_tokens=1024, temperature=0.0)
            model_output = response.choices[0].message.content

        except Exception as e:
            model_output = f"ERROR: {str(e)}"
        
        end_time = time.time()
      
        all_results.append({
            "id": item.get('id', 'N/A'),
            "prompt": prompt,
            "model_output": model_output,
            "ground_truth": ground_truth,
            "processing_time_seconds": round(end_time - start_time, 2)
        })

    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"Task complete. Results saved to: {result_json_path}")


def main():
    try:
        client = OpenAI(base_url=MODEL_ENDPOINT, api_key="EMPTY")
    except Exception as e:
        print(f"[Fatal] Could not initialize OpenAI client: {e}")
        sys.exit(1) 

    process_directory(client, MODEL_NAME, RESULT_SUFFIX)



if __name__ == "__main__":
    main()