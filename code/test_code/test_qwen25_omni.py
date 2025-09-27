import os
import json
import argparse
import time
import sys
import base64
import signal
import contextlib
from typing import Dict, Any, List
from tqdm import tqdm
from openai import OpenAI


class TimeoutError(Exception):
    pass

@contextlib.contextmanager
def timeout(seconds: int, error_message: str = 'Function call timed out'):
    def _handle_timeout(signum, frame):
        raise TimeoutError(error_message)
    
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

def get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return 'video'
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
        return 'image'
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def create_text_message(prompt: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

def create_multimodal_message(prompt: str, media_path: str) -> List[Dict[str, Any]]:
    media_type = get_media_type(media_path)
    
    try:
        with open(media_path, "rb") as media_file:
            media_base64 = base64.b64encode(media_file.read()).decode('utf-8')
    except IOError as e:
        raise IOError(f"Could not read or encode file {media_path}: {e}")

    content = [{"type": "text", "text": prompt}]
    if media_type == 'image':
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{media_base64}"}})
    elif media_type == 'video':
        content.append({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{media_base64}"}})
    
    return [{"role": "user", "content": content}]

def get_model_response(client: OpenAI, model_name: str, messages: List[Dict[str, Any]]) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=1024,
        temperature=0.0
    )
    return response.choices[0].message.content

def text_only_fallback(client: OpenAI, model_name: str, prompt_text: str) -> str:
    print("  [INFO] Executing text-only fallback...", file=sys.stderr)
    try:
        clean_prompt = prompt_text.replace("<image>", "").replace("<video>", "").strip()
        messages = create_text_message(clean_prompt)
        return get_model_response(client, model_name, messages)
    except Exception as e:
        return f"ERROR in text-only fallback: {str(e)}"

def process_file(client: OpenAI, model_name: str, result_suffix: str, json_filename: str):
    result_json_path = f"{os.path.splitext(json_filename)[0]}{result_suffix}"
    if os.path.exists(result_json_path):
        print(f"[INFO] Skipping already processed file: {json_filename}")
        return
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_results = []
    for item in tqdm(data, desc=f"  Processing {json_filename}", file=sys.stdout):
        start_time = time.time()
        model_output = ""
        prompt_text = ""
        ground_truth = ""

        try:
            prompt_text = item['conversations'][0]['value']
            ground_truth = item['conversations'][1]['value']
            clean_prompt = prompt_text.replace("<image>", "").replace("<video>", "").strip()

            media_relative_path = item.get('image') or item.get('video')

            if not media_relative_path:
                print(f"\n No media key found for item {item.get('id', 'N/A')}. Falling back to text-only.", file=sys.stderr)
                model_output = text_only_fallback(client, model_name, prompt_text)
            else:
                media_full_path = os.path.abspath(media_relative_path)
                if not os.path.exists(media_full_path):
                    raise FileNotFoundError(f"Media file not found: {media_full_path}")
                
                try:
                    with timeout(seconds=300):
                        messages = create_multimodal_message(clean_prompt, media_full_path)
                        model_output = get_model_response(client, model_name, messages)
                except TimeoutError:
                    print(f"\n Processing timed out for item {item.get('id', 'N/A')}. Falling back to text-only.", file=sys.stderr)
                    model_output = text_only_fallback(client, model_name, prompt_text)

        except Exception as e:
            error_message = f"ERROR: {str(e)}"
            model_output = error_message
            print(f"\n Failed to process item {item.get('id', 'N/A')}: {e}", file=sys.stderr)
        
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
    print(f" File processing complete. Results saved to: {result_json_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on JSON files in the current directory using a specified model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model-endpoint",
        default="http://127.0.0.1:8000/v1",
        help="The API endpoint for the model (e.g., 'http://127.0.0.1:8000/v1')."
    )
    parser.add_argument(
        "--model-name",
        default="qwen2.5-omni-7b",
        help="The name of the model to use for inference."
    )
    parser.add_argument(
        "--result-suffix",
        default="_result.json",
        help="The suffix to append to result filenames."
    )
    args = parser.parse_args()

    try:
        client = OpenAI(base_url=args.model_endpoint, api_key="EMPTY")
    except Exception as e:
        print(f"Could not create OpenAI client. Please check the endpoint: {e}", file=sys.stderr)
        sys.exit(1)

    current_dir = os.getcwd()
    source_files = sorted([
        f for f in os.listdir(current_dir)
        if f.endswith('.json') and not f.endswith(args.result_suffix)
    ])

    if not source_files:
        print("No source JSON files.", file=sys.stderr)
        return

    for json_filename in source_files:
        process_file(client, args.model_name, args.result_suffix, json_filename)
            


if __name__ == "__main__":
    main()