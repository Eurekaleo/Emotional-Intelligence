import os
import json
import argparse
import base64
import cv2
import tempfile
import subprocess
import shutil
import io
import time
import glob
from openai import OpenAI
from tqdm import tqdm
from PIL import Image

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
        raise IOError(f"Could not read or encode file {media_path}: {e}")

def extract_keyframes(video_path: str, max_frames: int = 20) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / video_fps if video_fps > 0 else 0

    if duration <= 20:
        target_frames = min(max_frames, int(duration))
        frame_interval = max(1, int(video_fps))  # 1 frame per second
    else:
        target_frames = max_frames
        frame_interval = max(1, int(total_frames / max_frames))


    keyframes = []
    frame_count = 0
    sampled_count = 0

    while cap.isOpened() and sampled_count < target_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            keyframes.append(pil_image)
            sampled_count += 1

        frame_count += 1

    cap.release()
    return keyframes

def extract_audio_to_text(video_path: str, client: OpenAI) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        command = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'mp3',
            '-ar', '16000', '-ac', '1', '-y', temp_audio_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"FFmpeg audio extraction failed: {result.stderr}")
            os.unlink(temp_audio_path)
            return "Audio extraction failed."
        with open(temp_audio_path, 'rb') as audio_file:

            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=(os.path.basename(temp_audio_path), audio_file.read()),
                response_format="text"
            )
        
        os.unlink(temp_audio_path)
        return transcription

    except Exception as e:
        print(f"Audio processing failed: {e}")
 
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return "Audio processing failed."

def process_single_sample(client: OpenAI, media_full_path: str, prompt_text: str) -> str:

    clean_prompt = prompt_text.replace("<image>", "").replace("<video>", "").strip()
    media_type = get_media_type(media_full_path)

    try:
        content = [{"type": "text", "text": clean_prompt}]

        if media_type == 'image':
            base64_media = encode_media_to_base64(media_full_path)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_media}"}})
        
        elif media_type == 'video':
            key_frames = extract_keyframes(media_full_path)
            audio_text = ""
            if shutil.which('ffmpeg'):
                try:
                    audio_text = extract_audio_to_text(media_full_path, client)
                except Exception as audio_e:
                    print(f" Could not extract audio from video: {audio_e}")
            else:
                print(" ffmpeg not found, skipping audio extraction.")

            for frame in key_frames:
                with io.BytesIO() as buffer:
                    frame.save(buffer, format='JPEG', quality=85)
                    frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}})
            
            if audio_text:
                full_prompt_with_audio = f"{clean_prompt}\n\n--- Video Audio Transcription ---\n{audio_text}"
                content[0] = {"type": "text", "text": full_prompt_with_audio}

        messages = [{"role": "user", "content": content}]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1024,
            temperature=0,
            timeout=120.0
        )
        return response.choices[0].message.content

    except Exception as e:
        try:
            messages = [{"role": "user", "content": clean_prompt}]
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1024,
                temperature=0,
                timeout=60.0
            )
            return response.choices[0].message.content
        except Exception as fallback_e:
            print(f"  Text-only fallback also failed: {fallback_e}")
            return f"ERROR: Fallback failed. Details: {fallback_e}"

def process_dataset_file(dataset_path: str, client: OpenAI, result_suffix: str):

    result_path = f"{os.path.splitext(dataset_path)[0]}{result_suffix}"
    if os.path.exists(result_path):
        print(f"Result file '{os.path.basename(result_path)}' already exists, skipping.")
        return

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f" Could not read or parse JSON file {dataset_path}: {e}")
        return

    all_results = []
    base_dir = os.path.dirname(dataset_path)

    for item in tqdm(data, desc=f"  Processing {os.path.basename(dataset_path)}"):
        start_time = time.time()
        model_output = ""
        try:
            prompt = item['conversations'][0]['value']
            ground_truth = item['conversations'][1]['value']
            media_path_key = 'image' if 'image' in item else 'video'
            media_relative_path = item.get(media_path_key)

            if not media_relative_path:
                raise ValueError("JSON item is missing 'image' or 'video' key.")
            
            media_full_path = os.path.join(base_dir, media_relative_path)
            if not os.path.exists(media_full_path):
                raise FileNotFoundError(f"Media file not found: {media_full_path}")
            
            model_output = process_single_sample(client, media_full_path, prompt)
            
        except Exception as e:
            model_output = f"ERROR: {str(e)}"
            print(f" Failed to process item {item.get('id', 'N/A')}: {e}")
        
        end_time = time.time()
        all_results.append({
            "id": item.get('id', 'N/A'), 
            "prompt": prompt, 
            "model_output": model_output, 
            "ground_truth": ground_truth, 
            "processing_time_seconds": round(end_time - start_time, 2)
        })

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f" Processing complete. Results saved to: {result_path}")

def main():
    parser = argparse.ArgumentParser(description="Run inference on a dataset using GPT-4o with multimodal capabilities.")
    parser.add_argument("--api-key", required=True, help="OpenAI API key.")
    parser.add_argument("--model-name", default="gpt-4o", help="The model to use (default: gpt-4o).")
    parser.add_argument("--result-suffix", default="_gpt4o_result.json", help="Suffix for the output result files.")
    
    args = parser.parse_args()

    try:
        client = OpenAI(api_key=args.api_key)
    except Exception as e:
        print(f"[Could not initialize OpenAI client: {e}")
        return

    if not shutil.which('ffmpeg'):
        print("ffmpeg not found. Video audio extraction will be disabled.")

    source_json_files = [
        f for f in glob.glob('*.json') 
        if not f.endswith(args.result_suffix) and GENERIC_RESULT_PATTERN not in f
    ]

    if not source_json_files:
        print("\n No source JSON files.")
    else:
        print(f"\nFound {len(source_json_files)} dataset(s) to process.")

    for json_file in sorted(source_json_files):
        process_dataset_file(json_file, client, args.result_suffix)
            
if __name__ == "__main__":
    main()