import os
import json
import sys
import time
import traceback
from typing import Dict, Any

from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from transformers import BertTokenizer

worker_model_objects: Dict[str, Any] = {}

def init_worker(model_path: str, bert_path: str, humanomni_project_path: str, device: str):
    global worker_model_objects
    
    if humanomni_project_path and humanomni_project_path not in sys.path:
        sys.path.append(humanomni_project_path)

    try:
        from humanomni import model_init, mm_infer
        from humanomni.utils import disable_torch_init
    except ImportError:
        print(f"[Worker PID: {os.getpid()}] ERROR: Failed to import HumanOmni. Ensure the humanomni_path is set correctly.", file=sys.stderr)
        return

    disable_torch_init()
    
    model, processor, tokenizer = model_init(model_path, device=device)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
    
    worker_model_objects = {
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        "bert_tokenizer": bert_tokenizer,
        "mm_infer": mm_infer,
    }

def get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return 'video'
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
        return 'image'
    else:
        return 'unknown'

def process_single_sample(media_full_path: str, prompt_text: str) -> str:
    global worker_model_objects
    try:
        model = worker_model_objects['model']
        processor = worker_model_objects['processor']
        tokenizer = worker_model_objects['tokenizer']
        bert_tokenizer = worker_model_objects['bert_tokenizer']
        mm_infer = worker_model_objects['mm_infer']
        
        media_type = get_media_type(media_full_path)
        if media_type == 'unknown':
            raise ValueError(f"Unsupported media type for file: {media_full_path}")
        
        clean_prompt = prompt_text.replace("<image>", "").replace("<video>", "").strip()

        media_tensor, audio_tensor, modal_str = None, None, ""

        if media_type == 'video':
            media_tensor = processor['video'](media_full_path)
            audio_tensor = processor['audio'](media_full_path)[0]
            modal_str = 'video_audio'
        elif media_type == 'image':
            media_tensor = processor['image'](media_full_path)
            if media_tensor.ndim == 3:
                media_tensor = media_tensor.unsqueeze(0)
            audio_tensor = None
            modal_str = 'image'

        output = mm_infer(
            media_tensor,
            instruct=clean_prompt,
            model=model,
            tokenizer=tokenizer,
            modal=modal_str,
            question=clean_prompt,
            bert_tokeni=bert_tokenizer,
            do_sample=False,
            audio=audio_tensor
        )
        return output
    except Exception as e:
        return f"ERROR: {e}\n{traceback.format_exc()}"

def text_only_fallback(prompt_text: str) -> str:
    global worker_model_objects
    try:
        model = worker_model_objects['model']
        tokenizer = worker_model_objects['tokenizer']
        
        clean_prompt = prompt_text.replace("<image>", "").replace("<video>", "").strip()
        inputs = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
        
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=512,
            do_sample=False
        )
        
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        if response.startswith(clean_prompt):
             return response[len(clean_prompt):].strip()
        else:
             return response

    except Exception as e:
        return f"ERROR in text-only fallback: {str(e)}"

def run_inference_task(media_full_path, prompt_text):
    if not worker_model_objects: return "ERROR: Worker model not initialized."
    return process_single_sample(media_full_path, prompt_text)

def run_fallback_task(prompt_text):
    if not worker_model_objects: return "ERROR: Worker model not initialized."
    return text_only_fallback(prompt_text)

def main():
    class Config:
        model_path = "example/model/HumanOmni_7B"
        
        bert_model_path = "example/model/bert-base-uncased"
        
        humanomni_path = "example/moodel"
        
        input_dir = "example"

        device = "cuda:7"
        
        result_suffix = "_humanomni_result.json"
        
        timeout = 60
    
    config = Config()
    
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    worker_device = config.device
    if "cuda" in config.device:
        gpu_id = config.device.split(':')[-1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        worker_device = "cuda:0"

    init_args = (config.model_path, config.bert_model_path, config.humanomni_path, worker_device)
    pool_ref = [mp.Pool(processes=1, initializer=init_worker, initargs=init_args)]

    try:
        source_json_files = [
            f for f in os.listdir(config.input_dir) 
            if f.endswith(".json") and not f.endswith(config.result_suffix)
        ]

        if not source_json_files:
            print("No source JSON files.")
            return

        for json_filename in source_json_files:
            dataset_json_path = os.path.join(config.input_dir, json_filename)
            result_json_path = os.path.join(config.input_dir, f"{os.path.splitext(json_filename)[0]}{config.result_suffix}")
            
            if os.path.exists(result_json_path):
                continue

            try:
                with open(dataset_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                continue

            all_results = []
            for item in tqdm(data, desc=f"  Inferring on {json_filename}", unit="item"):
                start_time = time.time()
                model_output, prompt, ground_truth = "", "", ""
                
                try:
                    prompt = item["conversations"][0]["value"]
                    ground_truth = item["conversations"][1]["value"]
                    media_relative_path = item.get('image') or item.get('video')

                    if not media_relative_path:
                        model_output = pool_ref[0].apply(run_fallback_task, args=(prompt,))
                    else:
                        media_full_path = os.path.join(config.input_dir, media_relative_path)
                        if not os.path.exists(media_full_path):
                            model_output = pool_ref[0].apply(run_fallback_task, args=(prompt,))
                        else:
                            media_type = get_media_type(media_full_path)
                            
                            if media_type == 'image':
                                model_output = pool_ref[0].apply(run_fallback_task, args=(prompt,))
                            elif media_type == 'video':
                                async_result = pool_ref[0].apply_async(run_inference_task, args=(media_full_path, prompt))
                                try:
                                    model_output = async_result.get(timeout=config.timeout)
                                except (mp.TimeoutError, Exception) as e:
                                    print(f"\n Worker task failed for item {item.get('id', 'N/A')}. Reason: {type(e).__name__}. Restarting and falling back.", file=sys.stderr)
                                    pool_ref[0].terminate()
                                    pool_ref[0].join()
                                    pool_ref[0] = mp.Pool(processes=1, initializer=init_worker, initargs=init_args)
                                    model_output = pool_ref[0].apply(run_fallback_task, args=(prompt,))
                            else: # Unknown media type
                                model_output = pool_ref[0].apply(run_fallback_task, args=(prompt,))

                except Exception as e:
                    model_output = f"ERROR: An unexpected error occurred in the main loop: {e}"

                all_results.append({
                    "id": item.get("id", "N/A"),
                    "prompt": prompt,
                    "model_output": model_output,
                    "ground_truth": ground_truth,
                    "processing_time_seconds": round(time.time() - start_time, 2),
                })

            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            print(f"Results saved to: {result_json_path}")

    finally:
        if pool_ref and pool_ref[0]:
            pool_ref[0].close()
            pool_ref[0].join()

if __name__ == "__main__":
    if torch.cuda.is_available():
        try:
            mp.set_start_method('spawn', force=True)
            print("Multiprocessing start method set to 'spawn'.")
        except RuntimeError:
            pass
        
    main()