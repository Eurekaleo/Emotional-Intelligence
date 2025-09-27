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

INLINE_SIZE_LIMIT_BYTES = 20 * 1024 * 1024

MODEL_NAME = "gemini-2.5-flash"
RESULT_SUFFIX = f"_{MODEL_NAME.replace('.', '_')}_result.json"


def get_mime_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    # ---- Video ----
    if ext in [".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv", ".3gp", ".3gpp", ".flv"]:
        return "video/mp4"

    # ---- Audio ----
    if ext in [".mp3", ".wav", ".aac", ".aiff", ".flac", ".ogg"]:
        if ext == ".mp3":
            return "audio/mp3"
        if ext == ".wav":
            return "audio/wav"
        if ext == ".aac":
            return "audio/aac"
        if ext == ".aiff":
            return "audio/aiff"
        if ext == ".flac":
            return "audio/flac"
        if ext == ".ogg":
            return "audio/ogg"

    # ---- Image ----
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"

    return "application/octet-stream"


def _poll_file_ready(client: genai.Client, file_obj, sleep_s: float = 2.0, max_wait_s: float = 300.0) -> Optional[object]:
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


def process_single_sample(client: genai.Client, media_full_path: str, prompt_text: str) -> str:
    clean_prompt = prompt_text.replace("<image>", "").replace("<video>", "").strip()
    file_size = os.path.getsize(media_full_path)
    mime_type = get_mime_type(media_full_path)

    try:
        if file_size < INLINE_SIZE_LIMIT_BYTES:
            print(f"\n  File size ({file_size / 1024**2:.2f} MB) is under limit. Using inline method.")
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
            return getattr(response, "text", str(response))

        else:
            print(f"\n   File size ({file_size / 1024**2:.2f} MB) exceeds limit. Using File API.")
            uploaded_file = None
            try:
                print(f"   Uploading: {os.path.basename(media_full_path)} ...")
                uploaded_file = client.files.upload(file=media_full_path)

                uploaded_file = _poll_file_ready(client, uploaded_file)
                if uploaded_file is None:
                    raise RuntimeError("File processing timeout in Files API.")

                print("   File is ready. Generating content ...")
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[
                        uploaded_file,                     
                        genai.types.Part(text=clean_prompt) 
                    ],
                )
                return getattr(response, "text", str(response))
            finally:
                try:
                    if uploaded_file and getattr(uploaded_file, "name", None):
                        print(f" Deleting uploaded file: {uploaded_file.name}")
                        client.files.delete(name=uploaded_file.name)
                except Exception as _e:
                    print(f" Failed to delete uploaded file: {_e}")

    except Exception as e:
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
            print(f" Result file already exists, skipping: {os.path.basename(result_json_path)}")
            continue

        try:
            with open(dataset_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f" Could not read or parse JSON file {dataset_json_path}: {e}")
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
        description=f"Test emotion.hallucination with {MODEL_NAME}, auto-selecting media strategy (google-genai)."
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GOOGLE_API_KEY", "KEY"),
        help="Google Gemini API key (or set the GOOGLE_API_KEY environment variable)."
    )
    args = parser.parse_args()

    if not args.api_key or args.api_key == "KEY":
        print("\nPlease provide your Google Gemini API key via the --api-key argument or by setting the GOOGLE_API_KEY environment variable.")
        return

    try:
        client = genai.Client(api_key=args.api_key)
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        return

    dataset_dir = os.getcwd()

    for level_dir in LEVEL_DIRS:
        level_path = os.path.join(dataset_dir, level_dir)
        if not os.path.isdir(level_path):
            continue

        task_dirs = sorted([d.path for d in os.scandir(level_path) if d.is_dir()])
        for task_path in task_dirs:
            process_task(task_path, client)


if __name__ == "__main__":
    main()