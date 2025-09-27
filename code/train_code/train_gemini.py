import os
import json
import argparse
import time
from typing import Optional
from tqdm import tqdm
from google import genai
import random
from collections import deque
#config
GENERIC_RESULT_PATTERN = "_result.json"

INLINE_SIZE_LIMIT_BYTES = 20 * 1024 * 1024

MODEL_NAME = "gemini-2.5-pro"
RESULT_SUFFIX = "_gemini_2.5_pro_result.json"

REQUESTS_PER_MINUTE = 30  
MIN_REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5
MAX_RETRY_DELAY = 120


class RateLimiter:
    
    def __init__(self, requests_per_minute):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.request_times = deque(maxlen=requests_per_minute)
    
    def wait_if_needed(self):
        current_time = time.time()
        
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
            current_time = time.time()
        
        minute_ago = current_time - 60
        while self.request_times and self.request_times[0] < minute_ago:
            self.request_times.popleft()
        
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0]) + 0.1
            if sleep_time > 0:
                print(f"      [RATE LIMIT] Waiting {sleep_time:.1f}s to respect rate limits...")
                time.sleep(sleep_time)
                current_time = time.time()
        
        self.last_request_time = current_time
        self.request_times.append(current_time)


def get_mime_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv", ".3gp", ".3gpp", ".flv"]:
        return "video/mp4"

    audio_types = {
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".aac": "audio/aac",
        ".aiff": "audio/aiff",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg"
    }
    if ext in audio_types:
        return audio_types[ext]

    image_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif"
    }
    if ext in image_types:
        return image_types[ext]

    return "application/octet-stream"


def _poll_file_ready(client: genai.Client, file_obj, sleep_s: float = 2.0, max_wait_s: float = 300.0) -> Optional[object]:
    """Poll file processing status."""
    start = time.time()
    name = getattr(file_obj, "name", None)
    state = getattr(file_obj, "state", None)
    state_name = getattr(state, "name", None) or str(state)

    while state_name and state_name.upper() in ("PROCESSING", "PENDING"):
        if time.time() - start > max_wait_s:
            return None
        time.sleep(sleep_s)
        try:
            file_obj = client.files.get(name=name)
        except Exception:
            time.sleep(sleep_s)
        state = getattr(file_obj, "state", None)
        state_name = getattr(state, "name", None) or str(state)

    return file_obj


def process_single_sample_with_retry(
    client: genai.Client, 
    media_full_path: str, 
    prompt_text: str,
    rate_limiter: RateLimiter,
    item_id: str = "N/A"
) -> str:
    """Process a single sample with retry logic and rate limiting."""
    retry_delay = INITIAL_RETRY_DELAY
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            rate_limiter.wait_if_needed()
            
            return process_single_sample(client, media_full_path, prompt_text)
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            if any(x in error_str.lower() for x in ["503", "overloaded", "rate", "quota", "429"]):
                if attempt < MAX_RETRIES - 1:
                    jitter = random.uniform(0, retry_delay * 0.3)
                    sleep_time = retry_delay + jitter
                    
                    print(f"\n      [RETRY] Item {item_id}: API overloaded/rate limited. "
                          f"Waiting {sleep_time:.1f}s... (Attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(sleep_time)
                    
                    retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                    
                    if "503" in error_str:
                        rate_limiter.min_interval *= 1.2 
                        print(f"      [THROTTLE] Slowing down to {60/rate_limiter.min_interval:.1f} RPM")
                    continue
            
            print(f"\n Item {item_id}: {error_str}")
            return f"ERROR: {error_str}"
    
    print(f"\n    Item {item_id}: Max retries exceeded")
    return f"ERROR: Max retries exceeded - {last_error}"


def process_single_sample(client: genai.Client, media_full_path: str, prompt_text: str) -> str:
    clean_prompt = prompt_text.replace("<image>", "").replace("<video>", "").strip()
    file_size = os.path.getsize(media_full_path)
    mime_type = get_mime_type(media_full_path)

    if file_size < INLINE_SIZE_LIMIT_BYTES:
        with open(media_full_path, "rb") as f:
            file_bytes = f.read()

        media_part = genai.types.Part(
            inline_data=genai.types.Blob(data=file_bytes, mime_type=mime_type)
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                media_part,
                genai.types.Part(text=clean_prompt),
            ],
        )
        text_response = getattr(response, "text", None)
        if text_response is None:
            return "ERROR: Empty response from model"
        return text_response

    else:
        uploaded_file = None
        try:
            print(f"\n      [UPLOAD] Uploading: {os.path.basename(media_full_path)} ({file_size/1024**2:.1f} MB)...")
            uploaded_file = client.files.upload(file=media_full_path)

            uploaded_file = _poll_file_ready(client, uploaded_file)
            if uploaded_file is None:
                raise RuntimeError("File processing timeout in Files API.")

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    uploaded_file,
                    genai.types.Part(text=clean_prompt)
                ],
            )
            text_response = getattr(response, "text", None)
            if text_response is None:
                return "ERROR: Empty response from model"
            return text_response
        finally:
            try:
                if uploaded_file and getattr(uploaded_file, "name", None):
                    client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


def save_result_immediately(result_json_path: str, all_results: list):
    os.makedirs(os.path.dirname(result_json_path) or ".", exist_ok=True)
    
    temp_path = result_json_path + f".tmp_{os.getpid()}"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        
        os.replace(temp_path, result_json_path)
        
        os.sync() if hasattr(os, 'sync') else None
        
    except Exception as e:
        print(f"\n  Failed to save results: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def load_existing_results(result_json_path: str) -> tuple[list, set]:
    if os.path.exists(result_json_path):
        try:
            with open(result_json_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            processed_ids = {item.get("id") for item in existing_results if item.get("id")}
            return existing_results, processed_ids
        except Exception as e:
            print(f"Could not load existing results: {e}")
    return [], set()


def process_json_file(
    json_path: str, 
    client: genai.Client, 
    resume: bool = True,
    requests_per_minute: int = REQUESTS_PER_MINUTE
):
    json_filename = os.path.basename(json_path)
    working_dir = os.path.dirname(json_path) or "."
    result_json_path = os.path.join(
        working_dir, 
        f"{os.path.splitext(json_filename)[0]}{RESULT_SUFFIX}"
    )

    rate_limiter = RateLimiter(requests_per_minute)

    all_results = []
    processed_ids = set()
    if resume:
        all_results, processed_ids = load_existing_results(result_json_path)
        if processed_ids:
            print(f"Found {len(processed_ids)} already processed items")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} total items from source file")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f" Could not read JSON file {json_path}: {e}")
        return

    items_to_process = []
    for i, item in enumerate(data):
        item_id = item.get("id", f"item_{i}")
        if "id" not in item:
            item["id"] = item_id
        if resume and item_id in processed_ids:
            continue
        items_to_process.append(item)
    
    if not items_to_process:
        print(f"All items already processed!")
        return

    if not os.path.exists(result_json_path):
        save_result_immediately(result_json_path, all_results)
        print(f" Result file: {result_json_path}\n")
    
    with tqdm(items_to_process, desc="Progress", ncols=100) as pbar:
        for idx, item in enumerate(pbar):
            start_time = time.time()
            item_id = item.get("id", f"item_{idx}")
            
            pbar.set_description(f"Processing {item_id}")
            
            try:
                prompt = item["conversations"][0]["value"]
                ground_truth = item["conversations"][1]["value"]

                media_path_key = "image" if "image" in item else "video"
                media_relative_path = item.get(media_path_key)
                if not media_relative_path:
                    raise ValueError("Missing 'image' or 'video' key")

                media_full_path = os.path.join(working_dir, media_relative_path)
                if not os.path.exists(media_full_path):
                    raise FileNotFoundError(f"Media file not found: {media_full_path}")

                model_output = process_single_sample_with_retry(
                    client, media_full_path, prompt, rate_limiter, item_id
                )

            except Exception as e:
                model_output = f"ERROR: {str(e)}"
                print(f"\n[ERROR] Failed item {item_id}: {e}")
                prompt = item.get("conversations", [{}])[0].get("value", "")
                ground_truth = item.get("conversations", [{}, {}])[1].get("value", "")

            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            if model_output is None:
                model_output = "ERROR: Null response from model"
            
            result_item = {
                "id": item_id,
                "prompt": prompt,
                "model_output": model_output,
                "ground_truth": ground_truth,
                "processing_time_seconds": processing_time,
            }
            
            all_results.append(result_item)
            
            save_result_immediately(result_json_path, all_results)
            
            current_rpm = 60 / rate_limiter.min_interval

            success_count = sum(1 for r in all_results 
                              if r.get("model_output") and not str(r["model_output"]).startswith("ERROR"))
            error_count = len(all_results) - success_count
            
            pbar.set_postfix({
                "Saved": len(all_results),
                "OK": success_count,
                "Err": error_count,
                "RPM": f"{current_rpm:.1f}",
                "Time": f"{processing_time:.1f}s"
            })
    
    success_count = sum(1 for r in all_results 
                        if r.get("model_output") and not str(r["model_output"]).startswith("ERROR"))
    error_count = len(all_results) - success_count
    

def main():
    parser = argparse.ArgumentParser(
        description=f"Process task with {MODEL_NAME} - saves after each item."
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GOOGLE_API_KEY", "API_KEY"),
        help="Google Gemini API key (or set env GOOGLE_API_KEY)."
    )
    parser.add_argument(
        "--json-file",
        type=str,
        help="Specific JSON file to process."
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch (ignore existing results)."
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=REQUESTS_PER_MINUTE,
        help=f"Requests per minute limit (default: {REQUESTS_PER_MINUTE})."
    )
    args = parser.parse_args()

    if not args.api_key:
        print("\nPlease provide your Google Gemini API key.")
        return

    try:
        client = genai.Client(api_key=args.api_key)
    except Exception as e:
        print(f"\n Failed to initialize Gemini client: {e}")
        return

    if args.json_file:
        if not os.path.exists(args.json_file):
            print(f"\n JSON file not found: {args.json_file}")
            return
        process_json_file(args.json_file, client, not args.no_resume, args.rpm)
    else:
        current_dir = os.getcwd()
        json_files = [
            f for f in os.listdir(current_dir)
            if f.endswith(".json") and GENERIC_RESULT_PATTERN not in f
        ]
        
        if not json_files:
            print(f"\nNo source JSON files found in current directory.")
            return
        for f in json_files:
            print(f"  - {f}")
        
        for json_file in json_files:
            json_path = os.path.join(current_dir, json_file)
            process_json_file(json_path, client, not args.no_resume, args.rpm)

if __name__ == "__main__":
    main()