import json
import cv2
import base64
import os
import time
from openai import OpenAI
from pathlib import Path

API_KEY = os.environ.get("OPENAI_API_KEY", "api-key")
client = OpenAI(api_key=API_KEY)
MODEL_NAME = "gpt-5"                          

INPUT_FILE = "example.json"                     
OUTPUT_FILE = "gpt5_results.json"             

TEST_MODE = True                             
TEST_ITEMS = 5                             
VIDEO_SAMPLE_INTERVAL = 25                    

MAX_RETRY = 3                                 
SLEEP_BETWEEN_RETRY = 3                        
REASONING_EFFORT = "high"                      

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None

def process_video(video_path, sample_interval=VIDEO_SAMPLE_INTERVAL):
    try:
        video = cv2.VideoCapture(video_path)
        base64_frames = []
        frame_count = 0
        
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            frame_count += 1

            if (frame_count - 1) % sample_interval == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        
        video.release()
        print(f"Processed video: {frame_count} total frames, sampled {len(base64_frames)} frames")
        return base64_frames
    
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

def determine_media_type(file_path):
    ext = Path(file_path).suffix.lower()
    
    if ext in VIDEO_EXTENSIONS:
        return 'video'
    elif ext in IMAGE_EXTENSIONS:
        return 'image'
    else:
        print(f" Unknown extension {ext}, treating as video")
        return 'video'

def call_gpt5_with_retry(prompt, media_path, attempt_num=1):
    media_type = determine_media_type(media_path)
    
    if not os.path.exists(media_path):
        print(f" Warning: File {media_path} not found")
        return None
    
    for retry in range(MAX_RETRY):
        try:
            print(f"  Attempt {attempt_num}.{retry + 1}: Processing {media_type}...")
            
            if media_type == 'video':
                frames = process_video(media_path)
                if not frames:
                    print(f"   No frames extracted from {media_path}")
                    return None
                
                content = [
                    {"type": "input_text", "text": prompt},
                    *[
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{frame}"
                        }
                        for frame in frames
                    ]
                ]
            else:
                base64_image = encode_image_to_base64(media_path)
                if not base64_image:
                    return None
                    
                content = [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            
            response = client.responses.create(
                model=MODEL_NAME,
                reasoning={"effort": REASONING_EFFORT},
                input=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            return response.output_text
        
        except Exception as e:
            print(f" Error on attempt {retry + 1}: {str(e)}")
            if retry < MAX_RETRY - 1:
                print(f" Retrying in {SLEEP_BETWEEN_RETRY} seconds...")
                time.sleep(SLEEP_BETWEEN_RETRY)
            else:
                print(f" Failed after {MAX_RETRY} attempts")
                return None
    
    return None

def extract_prompt_and_truth(conversations):
    prompt = None
    ground_truth = None
    
    if conversations and len(conversations) > 0 and conversations[0]["from"] == "human":
        full_value = conversations[0]["value"]
        prompt = full_value.replace("<video>", "").replace("<image>", "").strip()
    
    if len(conversations) >= 2 and conversations[1]["from"] == "gpt":
        ground_truth = conversations[1]["value"]
    
    return prompt, ground_truth

def main():

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found")
        return
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items from {INPUT_FILE}")
    
    items_to_process = data[:TEST_ITEMS] if TEST_MODE else data
    results = []
    
    
    for idx, item in enumerate(items_to_process, 1):
        item_id = item.get("id")
        media_path = item.get("video") or item.get("image")
        conversations = item.get("conversations", [])
        
        print(f"\n[{idx}/{len(items_to_process)}] Processing ID: {item_id}")
        print(f"  Media: {media_path}")
        
        if not media_path or not conversations:
            print(f"Skipping: missing media path or conversations")
            continue

        prompt, ground_truth = extract_prompt_and_truth(conversations)
        
        if not prompt:
            print(f"Skipping: no prompt found")
            continue

        response1 = call_gpt5_with_retry(prompt, media_path, attempt_num=1)

        response2 = call_gpt5_with_retry(prompt, media_path, attempt_num=2)

        result = {
            "id": item_id,
            "media_path": media_path,
            "response_1": response1,
            "response_2": response2,
            "ground_truth": ground_truth
        }
        results.append(result)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()