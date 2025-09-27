import os
import json
import argparse
import base64
from openai import OpenAI
from tqdm import tqdm
import time


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
        raise
    except Exception as e:
        raise IOError(f"Failed to read or encode file {media_path}: {e}")

def process_file(dataset_json_path: str, client: OpenAI, model_name: str, result_suffix: str):
    json_filename = os.path.basename(dataset_json_path)
    result_json_path = os.path.join(
        os.path.dirname(dataset_json_path),
        f"{os.path.splitext(json_filename)[0]}{result_suffix}"
    )

    if os.path.exists(result_json_path):
        print(f"Result file '{os.path.basename(result_json_path)}' already exists. Skipping.")
        return

    try:
        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Failed to read or parse JSON file {dataset_json_path}: {e}")
        return

    all_results = []
    base_path = os.path.dirname(dataset_json_path)

    for item in tqdm(data, desc=f"  Querying API for {json_filename}"):
        start_time = time.time()
        model_output = "N/A"
        try:
            prompt = item['conversations'][0]['value']
            ground_truth = item['conversations'][1]['value']
            media_path_key = 'image' if 'image' in item else 'video'
            media_relative_path = item.get(media_path_key)
            if not media_relative_path:
                raise ValueError("JSON item is missing 'image' or 'video' key.")
            
            media_full_path = os.path.join(base_path, media_relative_path)
            if not os.path.exists(media_full_path):
                raise FileNotFoundError(f"Media file not found: {media_full_path}")
            
            media_type = get_media_type(media_full_path)
            media_base64 = encode_media_to_base64(media_full_path)
            clean_prompt = prompt.replace("<image>", "").replace("<video>", "").strip()
            
            if media_type == 'image':
                messages = [{"role": "user", "content": [{"type": "text", "text": clean_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{media_base64}"}}]}]
            else: 
                messages = [{"role": "user", "content": [{"type": "text", "text": clean_prompt}, {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{media_base64}"}}]}]

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
    print(f"  [SUCCESS] Processing complete. Results saved to: {result_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch inference for multimodal models using an OpenAI-compatible API.")
    parser.add_argument("--model-endpoint", default="http://localhost:8004/v1", help="The API endpoint of the model server.")
    parser.add_argument("--model-name", default="llavaonevision7b", help="The name of the model to use.")
    parser.add_argument("--result-suffix", default="_result.json", help="Suffix for the generated result files.")
    args = parser.parse_args()

    try:
        client = OpenAI(base_url=args.model_endpoint, api_key="EMPTY")
    except Exception as e:
        print(f"Could not initialize OpenAI client: {e}")
        return

    current_dir = os.getcwd()
    source_json_files = [
        f for f in os.listdir(current_dir) 
        if f.endswith('.json') and not f.endswith(GENERIC_RESULT_PATTERN)
    ]

    if not source_json_files:
        print(f"\nNo source JSON files: {current_dir}")
    else:
        for json_filename in sorted(source_json_files):
            process_file(
                dataset_json_path=os.path.join(current_dir, json_filename),
                client=client,
                model_name=args.model_name,
                result_suffix=args.result_suffix
            )
            

if __name__ == "__main__":
    main()