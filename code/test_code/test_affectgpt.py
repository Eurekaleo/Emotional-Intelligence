import os
import json
import argparse
import time
import sys
from typing import Dict, Any
from tqdm import tqdm
import torch
import cv2
import tempfile
import torch.backends.cudnn as cudnn
import decord
decord.bridge.set_bridge('torch')
import torch.multiprocessing as mp

# --- Worker Initialization for Multiprocessing Pool ---
worker_chat = None
worker_cfg = None

def init_worker():
    global worker_chat, worker_cfg
    setup_seeds(42)
    worker_chat, worker_cfg = load_affectgpt_model()
    print(f"[Worker PID: {os.getpid()}] Model loaded successfully.")

sys.path.append('AffectGPT')

from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry
from my_affectgpt.conversation.conversation_video import Chat

# --- Configuration ---
LEVEL_DIRS = ["level1", "level2", "level3"]
GENERIC_RESULT_PATTERN = "_result.json"

def get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return 'video'
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
        return 'image'
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def setup_seeds(seed=42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def load_affectgpt_model():
    print("Loading AffectGPT model...")
    import config
    config.PATH_TO_LLM['Qwen25'] = 'Qwen25'
    config.PATH_TO_VISUAL['CLIP_VIT_LARGE'] = 'CLIP_VIT_LARGE'
    config.PATH_TO_AUDIO['HUBERT_LARGE'] = 'HUBRT_LARGE'
    cfg_path = "CFG_PATH"
    class Args:
        def __init__(self):
            self.cfg_path = cfg_path
            self.options = ["inference.test_epoch=60"]
    args = Args()
    cfg = Config(args)
    model_cfg = cfg.model_cfg
    device = 'cuda:7'
    ckpt_path = "CKPT_PATH"
    model_cfg.ckpt_3 = ckpt_path
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)
    model = model.to(device).eval()
    chat = Chat(model, model_cfg, device=device)
    print("AffectGPT model loaded!")
    return chat, cfg

def create_complete_dataset(cfg):
    from my_affectgpt.processors import BaseProcessor
    class CompleteDataset:
        def __init__(self, cfg):
            self.vis_processor = BaseProcessor()
            self.img_processor = BaseProcessor()
            self.n_frms = 8
            inference_cfg = cfg.inference_cfg
            vis_processor_cfg = inference_cfg.get("vis_processor")
            img_processor_cfg = inference_cfg.get("img_processor")
            if vis_processor_cfg is not None:
                self.vis_processor = registry.get_processor_class(vis_processor_cfg.train.name).from_config(vis_processor_cfg.train)
            if img_processor_cfg is not None:
                self.img_processor = registry.get_processor_class(img_processor_cfg.train.name).from_config(img_processor_cfg.train)
            self.n_frms = cfg.model_cfg.vis_processor.train.n_frms
            
        def read_frame_face_audio_text(self, video_path, face_npy, audio_path, image_path):
            sample_data = {}
            frame, raw_frame = None, None
            if video_path is not None:
                from my_affectgpt.processors.video_processor import load_video
                raw_frame, msg = load_video(video_path=video_path, n_frms=self.n_frms, height=224, width=224, sampling="uniform", return_msg=True)
                frame = self.vis_processor.transform(raw_frame)
            sample_data['frame'] = frame
            sample_data['raw_frame'] = raw_frame
            sample_data['face'] = None
            sample_data['raw_face'] = None
            audio, raw_audio = None, None
            if audio_path is not None and os.path.exists(audio_path):
                from my_affectgpt.models.ImageBind.data import load_audio, transform_audio
                raw_audio = load_audio([audio_path], "cpu", clips_per_video=8)[0]
                audio = transform_audio(raw_audio, "cpu")
            sample_data['audio'] = audio
            sample_data['raw_audio'] = raw_audio
            image, raw_image = None, None
            if image_path is not None and os.path.exists(image_path):
                from PIL import Image as PILImage
                raw_image = PILImage.open(image_path).convert('RGB')
                image = self.img_processor(raw_image)
            sample_data['image'] = image
            sample_data['raw_image'] = raw_image
            return sample_data
        def get_prompt_for_multimodal(self, face_or_frame, subtitle, user_message):
            prompt = f"<FrameHere>{user_message}"
            if subtitle:
                prompt = f"Context: {subtitle}\n{prompt}"
            return prompt
    return CompleteDataset(cfg)

def process_single_sample(chat, cfg, media_full_path, prompt_text):
    try:
        media_type = get_media_type(media_full_path)
        if media_type == 'image':
            temp_video = tempfile.mktemp(suffix='.mp4')
            img = cv2.imread(media_full_path)
            height, width, layers = img.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_video, fourcc, 1.0, (width, height))
            for _ in range(30):
                video_writer.write(img)
            video_writer.release()
            video_path, audio_path, image_path = temp_video, None, None
        else:
            video_path, audio_path, image_path = media_full_path, None, None
        
        dataset_cls = create_complete_dataset(cfg)
        sample_data = dataset_cls.read_frame_face_audio_text(video_path, None, audio_path, image_path)
        _, audio_llms = chat.postprocess_audio(sample_data)
        _, frame_llms = chat.postprocess_frame(sample_data)
        _, face_llms = chat.postprocess_face(sample_data)
        _, image_llms = chat.postprocess_image(sample_data)
        
        img_list = {'audio': audio_llms, 'frame': frame_llms, 'face': face_llms, 'image': image_llms, 'multi': None}
        
        user_message = prompt_text.replace("<image>", "").replace("<video>", "").strip()
        prompt = dataset_cls.get_prompt_for_multimodal('frame', "", user_message)
        
        response = chat.answer_sample(prompt=prompt, img_list=img_list, num_beams=1, temperature=0.1, do_sample=False, top_p=0.9, max_new_tokens=512, max_length=2000)
        
        if media_type == 'image' and os.path.exists(temp_video):
            os.remove(temp_video)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"

def text_only_fallback_logic(chat, cfg, prompt_text: str) -> str:
    print("Executing text-only fallback in worker...")
    try:
        dataset_cls = create_complete_dataset(cfg)
        user_message = prompt_text.replace("<image>", "").replace("<video>", "").strip()
        prompt = dataset_cls.get_prompt_for_multimodal('frame', "", user_message)
        img_list = {'audio': None, 'frame': None, 'face': None, 'image': None, 'multi': None}
        response = chat.answer_sample(prompt=prompt, img_list=img_list, num_beams=1, temperature=0.1, do_sample=False, top_p=0.9, max_new_tokens=512, max_length=2000)
        return response
    except Exception as e:
        return f"ERROR in text-only fallback: {str(e)}"

# --- Worker Task Functions ---
def run_inference_task(media_full_path, prompt_text):
    global worker_chat, worker_cfg
    if worker_chat is None or worker_cfg is None: return "ERROR: Worker model not initialized."
    return process_single_sample(worker_chat, worker_cfg, media_full_path, prompt_text)

def run_fallback_task(prompt_text):
    global worker_chat, worker_cfg
    if worker_chat is None or worker_cfg is None: return "ERROR: Worker model not initialized."
    return text_only_fallback_logic(worker_chat, worker_cfg, prompt_text)

def process_task(task_path: str, result_suffix: str, pool_ref: list):
    source_json_files = [f for f in os.listdir(task_path) if f.endswith('.json') and GENERIC_RESULT_PATTERN not in f]
    if not source_json_files:
        print(f"   No source JSON files found in {task_path}.")
        return

    for json_filename in source_json_files:
        dataset_json_path = os.path.join(task_path, json_filename)
        result_json_path = os.path.join(task_path, f"{os.path.splitext(json_filename)[0]}{result_suffix}")
        if os.path.exists(result_json_path):
            print(f"  [INFO] Result file '{os.path.basename(result_json_path)}' already exists, skipping.")
            continue

        print(f"Reading and processing dataset: {json_filename}")
        try:
            with open(dataset_json_path, 'r', encoding='utf-8') as f: data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  Could not read or parse JSON file {dataset_json_path}: {e}")
            continue

        all_results = []
        for item in tqdm(data, desc=f"  Processing {json_filename}"):
            start_time = time.time()
            model_output, prompt, ground_truth = "", "", ""
            try:
                prompt = item['conversations'][0]['value']
                ground_truth = item['conversations'][1]['value']
                media_relative_path = item.get('image') or item.get('video')
                if not media_relative_path and 'conversations' in item and isinstance(item.get('conversations')[0], dict):
                    media_relative_path = item['conversations'][0].get('image') or item['conversations'][0].get('video')

                if not media_relative_path:
                    model_output = pool_ref[0].apply(run_fallback_task, args=(prompt,))
                else:
                    media_full_path = os.path.join(task_path, media_relative_path)
                    if not os.path.exists(media_full_path):
                        model_output = pool_ref[0].apply(run_fallback_task, args=(prompt,))
                    else:
                        async_result = pool_ref[0].apply_async(run_inference_task, args=(media_full_path, prompt))
                        try:
                            result = async_result.get(timeout=60)
                            if isinstance(result, str) and result.startswith("ERROR:"):
                                pool_ref[0].terminate()
                                pool_ref[0].join()
                                pool_ref[0] = mp.Pool(processes=1, initializer=init_worker)
                                model_output = pool_ref[0].apply(run_fallback_task, args=(prompt,))
                            else:
                                model_output = result
                        except mp.TimeoutError:
                            pool_ref[0].terminate()
                            pool_ref[0].join()
                            pool_ref[0] = mp.Pool(processes=1, initializer=init_worker)
                            model_output = pool_ref[0].apply(run_fallback_task, args=(prompt,))
            except Exception as e:
                model_output = f"ERROR: Main loop error: {e}"
                print(f"\n {model_output}")

            end_time = time.time()
            all_results.append({
                "id": item.get('id', 'N/A'), "prompt": prompt, "model_output": model_output,
                "ground_truth": ground_truth, "processing_time_seconds": round(end_time - start_time, 2)
            })

        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f"  Task complete. Results saved to: {result_json_path}")

def main():
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Test emotion.hallucination with AffectGPT.")
    parser.add_argument("--result-suffix", default="_affectgpt_result.json", help="Result file suffix")
    args = parser.parse_args()

    pool = mp.Pool(processes=1, initializer=init_worker)
    pool_ref = [pool] 

    try:
        dataset_dir = "emotion_task"
        os.chdir(dataset_dir)
        for level_dir in LEVEL_DIRS:
            level_path = os.path.join(dataset_dir, level_dir)
            if not os.path.isdir(level_path): continue
            
            task_dirs = sorted([d.path for d in os.scandir(level_path) if d.is_dir()])
            for task_path in task_dirs:
                process_task(task_path, args.result_suffix, pool_ref)
    finally:
        pool_ref[0].close()
        pool_ref[0].join()

if __name__ == "__main__":
    main()