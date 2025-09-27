import os
import json
import argparse
import time
from typing import Optional
from tqdm import tqdm
from google import genai

# --- Configuration ---
LEVEL_DIRS = ["level1", "level2", "level3"]
GENERIC_RESULT_PATTERN = "_result.json"
PROMPT_IMAGE_PLACEHOLDER = "<image>"
PROMPT_VIDEO_PLACEHOLDER = "<video>"

INLINE_SIZE_LIMIT_BYTES = 20 * 1024 * 1024

MODEL_NAME = "gemini-2.5-pro"
RESULT_SUFFIX = "_gemini_2.5_pro_result.json"


def get_mime_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    # --- Video ---
    if ext in [".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv", ".3gp", ".3gpp", ".flv"]:
        return "video/mp4"

    # --- Audio ---
    if ext in [".mp3", ".wav", ".aac", ".aiff", ".flac", ".ogg"]:
        return f"audio/{ext[1:]}"

    # --- Image ---
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext in [".png", ".webp", ".gif"]:
        return f"image/{ext[1:]}"

    return "application/octet-stream"


def _poll_file_ready(client: genai.Client, file_obj, sleep_s: float = 2.0, max_wait_s: float = 300.0) -> Optional[object]:
    start_time = time.time()
    name = getattr(file_obj, "name", None)
    if not name:
        return file_obj 

    while time.time() - start_time < max_wait_s:
        state = getattr(file_obj, "state", None)
        state_name = getattr(state, "name", None) or str(state)

        if state_name.upper() not in ("PROCESSING", "PENDING"):

            return file_obj

        time.sleep(sleep_s)
        try:
            file_obj = client.files.get(name=name)
        except Exception:
            
            time.sleep(sleep_s)
    
    return None 

def process_single_sample(client: genai.Client, media_full_path: str, prompt_text: str) -> str:

    clean_prompt = prompt_text.replace(PROMPT_IMAGE_PLACEHOLDER, "").replace(PROMPT_VIDEO_PLACEHOLDER, "").strip()
    file_size = os.path.getsize(media_full_path)
    mime_type = get_mime_type(media_full_path)

    try:
        if file_size < INLINE_SIZE_LIMIT_BYTES:
            print(f"\n      [INFO] File size ({file_size / 1024**2:.2f} MB) is under limit. Using inline method.")
            with open(media_full_path, "rb") as f:
                media_part = genai.types.Part(
                    inline_data=genai.types.Blob(data=f.read(), mime_type=mime_type)
                )

            contents = [media_part, genai.types.Part(text=clean_prompt)]
            response = client.models.generate_content(model=MODEL_NAME, contents=contents)
            return getattr(response, "text", str(response))

        else:
            print(f"\n     File size ({file_size / 1024**2:.2f} MB) exceeds limit. Using File API.")
            uploaded_file = None
            try:
                print(f"    Uploading: {os.path.basename(media_full_path)}...")
                uploaded_file = client.files.upload(file_path=media_full_path)

                uploaded_file = _poll_file_ready(client, uploaded_file)
                if uploaded_file is None:
                    raise RuntimeError("File processing timed out in Files API.")

                print("   File is ready. Generating content...")
                contents = [uploaded_file, genai.types.Part(text=clean_prompt)]
                response = client.models.generate_content(model=MODEL_NAME, contents=contents)
                return getattr(response, "text", str(response))
            finally:
                if uploaded_file and getattr(uploaded_file, "name", None):
                    try:
                        print(f"   Deleting uploaded file: {uploaded_file.name}")
                        client.files.delete(name=uploaded_file.name)
                    except Exception as e:
                        print(f"   Failed to delete uploaded file: {e}")

    except Exception as e:
        print(f"  An error occurred during Gemini processing: {e}")
        return f"ERROR: {str(e)}"


def process_task(task_path: str, client: genai.Client):

    source_json_files = [
        f for f in os.listdir(task_path)
        if f.endswith(".json") and GENERIC_RESULT_PATTERN not in f
    ]
    if not source_json_files:
        print(f" No source JSON files found in {task_path}.")
        return

    for json_filename in source_json_files:
        dataset_json_path = os.path.join(task_path, json_filename)
        result_json_path = os.path.join(task_path, f"{os.path.splitext(json_filename)[0]}{RESULT_SUFFIX}")

        if os.path.exists(result_json_path):
            print(f"  Result file already exists, skipping: {os.path.basename(result_json_path)}")
            continue

        print(f" Reading and processing dataset: {json_filename}")
        try:
            with open(dataset_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Could not read or parse JSON file {dataset_json_path}: {e}")
            continue

        all_results = []
        for item in tqdm(data, desc=f"  Processing {json_filename}"):
            start_time = time.time()
            model_output = ""
            prompt = ""
            ground_truth = ""
            try:
                prompt = item["conversations"][0]["value"]
                ground_truth = item["conversations"][1]["value"]

                media_path_key = "image" if "image" in item else "video"
                media_relative_path = item.get(media_path_key)
                if not media_relative_path:
                    raise ValueError("Missing 'image' or 'video' key in JSON item.")

                media_full_path = os.path.join(task_path, media_relative_path)
                if not os.path.exists(media_full_path):
                    raise FileNotFoundError(f"Media file not found: {media_full_path}")

                model_output = process_single_sample(client, media_full_path, prompt)

            except Exception as e:
                model_output = f"ERROR: {str(e)}"
                print(f" Failed to process item {item.get('id', 'N/A')}: {e}")

            end_time = time.time()
            all_results.append({
                "id": item.get("id", "N/A"),
                "prompt": prompt,
                "model_output": model_output,
                "ground_truth": ground_truth,
                "processing_time_seconds": round(end_time - start_time, 2),
            })

        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f" Task complete. Results saved to: {result_json_path}")


def main():

    parser = argparse.ArgumentParser(
        description=f"Run batch inference on datasets using the Google Gemini '{MODEL_NAME}' model."
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        help="Google Gemini API key. Can also be set via the GOOGLE_API_KEY environment variable."
    )
    args = parser.parse_args()

    if not args.api_key or args.api_key == "GEMINI_API_KEY":
        return

    try:
        genai.configure(api_key=args.api_key)
        client = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        return

    dataset_dir = os.getcwd()
    print(f"Running in directory: {dataset_dir}")

    for level_dir in LEVEL_DIRS:
        level_path = os.path.join(dataset_dir, level_dir)
        if not os.path.isdir(level_path):
            continue
        
        task_dirs = sorted([d.path for d in os.scandir(level_path) if d.is_dir()])
        for task_path in task_dirs:
            process_task(task_path, client)


if __name__ == "__main__":
    main()