import os
import json
import argparse
from tqdm import tqdm
import time
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 448
DEFAULT_VIDEO_SEGMENTS = 8
DEFAULT_MAX_PATCHES_PER_FRAME = 1  
DEFAULT_MAX_PATCHES_PER_IMAGE = 6  

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def get_frame_indices(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_frame_indices(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    valid_indices = [i for i in frame_indices if i < len(vr)]
    if not valid_indices:
        raise ValueError(f"No valid frames could be sampled from video {video_path}.")

    frames = vr.get_batch(valid_indices).asnumpy()

    for frame_np in frames:
        img = Image.fromarray(frame_np).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
        
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return 'video'
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
        return 'image'
    else:
        raise ValueError(f"Unsupported file format: {ext} in file {file_path}")

def process_file(dataset_json_path: str, model, tokenizer, result_suffix: str):
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

    generation_config = dict(num_beams=1, max_new_tokens=2048, do_sample=False)
    device = next(model.parameters()).device
    all_results = []
    
    base_path = os.path.dirname(dataset_json_path)

    for item in tqdm(data, desc=f"  Inferring on {json_filename}"):
        start_time = time.time()
        model_output = "N/A"
        
        try:
            prompt_text = item['conversations'][0]['value']
            ground_truth = item['conversations'][1]['value']
            media_path_key = 'image' if 'image' in item else 'video'
            media_relative_path = item.get(media_path_key)
            if not media_relative_path:
                raise ValueError("JSON item is missing 'image' or 'video' key.")
            
            media_full_path = os.path.join(base_path, media_relative_path)
            if not os.path.exists(media_full_path):
                raise FileNotFoundError(f"Media file not found: {media_full_path}")
            
            media_type = get_media_type(media_full_path)
            clean_prompt = prompt_text.replace("<image>", "").replace("<video>", "").strip()

            pixel_values, num_patches_list, question = None, None, None

            if media_type == 'image':
                image = Image.open(media_full_path).convert('RGB')
                transform = build_transform(input_size=DEFAULT_IMAGE_SIZE)
                patches = dynamic_preprocess(image, image_size=DEFAULT_IMAGE_SIZE, use_thumbnail=True, max_num=DEFAULT_MAX_PATCHES_PER_IMAGE)
                pixel_values = torch.stack([transform(p) for p in patches])
                num_patches_list = [len(patches)]
                question = f"<image>\n{clean_prompt}"
            
            elif media_type == 'video':
                pixel_values, num_patches_list = load_video(
                    media_full_path,
                    num_segments=DEFAULT_VIDEO_SEGMENTS,
                    max_num=DEFAULT_MAX_PATCHES_PER_FRAME,
                    input_size=DEFAULT_IMAGE_SIZE
                )
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                question = f"{video_prefix}{clean_prompt}"

            pixel_values = pixel_values.to(torch.bfloat16).to(device)

            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                num_patches_list=num_patches_list,
                history=None
            )
            model_output = response.strip()

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
    parser = argparse.ArgumentParser(description="Batch inference with InternVL model on local JSON datasets.")
    parser.add_argument("--model-path", required=True, help="Full path to the local model directory.")
    parser.add_argument("--result-suffix", default="_result.json", help="Suffix for the generated result files.")
    args = parser.parse_args()

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    except Exception as e:
        print(f"Failed to load the model from {args.model_path}. Error: {e}")
        return
    
    current_dir = os.getcwd()
    source_json_files = [
        f for f in os.listdir(current_dir)
        if f.endswith('.json') and not f.endswith(args.result_suffix)
    ]

    if not source_json_files:
        print(f"\nNo source JSON files: {current_dir}")
    else:
        for json_filename in sorted(source_json_files):
            process_file(os.path.join(current_dir, json_filename), model, tokenizer, args.result_suffix)


if __name__ == "__main__":
    main()