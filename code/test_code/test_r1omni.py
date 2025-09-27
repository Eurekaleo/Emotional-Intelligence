
import os
import json
import argparse
import time
import sys
import glob
from typing import Dict, Any, List
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from transformers import BertTokenizer

from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init

worker_model_objects: Dict[str, Any] = {}

def init_worker(model_path: str, bert_path: str, device: str):
    global worker_model_objects
    try:
        disable_torch_init()
        model, processor, tokenizer = model_init(model_path, device=device)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
        worker_model_objects = {
            "model": model,
            "processor": processor,
            "tokenizer": tokenizer,
            "bert_tokenizer": bert_tokenizer,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

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
            modal_str = 'image'

        output = mm_infer(
            media=media_tensor,
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
        import traceback
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
        return response
    except Exception as e:
        return f"ERROR in text-only fallback: {str(e)}"

def run_inference_task(media_full_path: str, prompt_text: str) -> str:
    if not worker_model_objects: return "ERROR: Worker model not initialized."
    return process_single_sample(media_full_path, prompt_text)

def run_fallback_task(prompt_text: str) -> str:
    if not worker_model_objects: return "ERROR: Worker model not initialized."
    return text_only_fallback(prompt_text)

def process_json_file(
    dataset_json_path: str,
    result_suffix: str,
    pool_ref: List[mp.Pool],
    model_path: str,
    bert_path: str,
    device: str
):

    base_dir = os.path.dirname(dataset_json_path)
    json_filename = os.path.basename(dataset_json_path)
    result_json_path = os.path.join(base_dir, f"{os.path.splitext(json_filename)[0]}{result_suffix}")

    if os.path.exists(result_json_path):
        return

    try:
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return

    all_results = []
    for item in tqdm(data, desc=f"  Inferring on {json_filename}", unit="item"):
        start_time = time.time()
        model_output, prompt, ground_truth = "", "", ""
        pool = pool_ref[0]

        try:
            prompt = item["conversations"][0]["value"]
            ground_truth = item["conversations"][1]["value"]
            media_relative_path = item.get('image') or item.get('video')

            if not media_relative_path:
                model_output = pool.apply(run_fallback_task, args=(prompt,))
            else:
                media_full_path = os.path.join(base_dir, media_relative_path)
                if not os.path.exists(media_full_path):
                    model_output = pool.apply(run_fallback_task, args=(prompt,))
                else:
                    async_result = pool.apply_async(run_inference_task, args=(media_full_path, prompt))
                    try:
                        model_output = async_result.get(timeout=60)
                    except (mp.TimeoutError, Exception) as e:
                        pool.terminate()
                        pool.join()
                        pool_ref[0] = mp.Pool(processes=1, initializer=init_worker, initargs=(model_path, bert_path, device))
                        model_output = pool_ref[0].apply(run_fallback_task, args=(prompt,))

        except Exception as e:
            model_output = f"ERROR: Main loop error: {e}"
            print(f"\n {model_output}")

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

def main():
    parser = argparse.ArgumentParser(description="Batch inference with R1-Omni model on local JSON datasets.")
    parser.add_argument("--model-path", required=True, help="Path to the R1-Omni model directory.")
    parser.add_argument("--bert-path", required=True, help="Path to the bert-base-uncased tokenizer directory.")
    parser.add_argument("--input-dir", default=".", help="Directory containing JSON datasets and media files.")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on (e.g., 'cuda:0', 'cpu').")
    parser.add_argument("--result-suffix", default="_r1omni_result.json", help="Suffix for result JSON files.")
    args = parser.parse_args()
    
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    worker_device = args.device
    if "cuda" in args.device:
        gpu_id = args.device.split(':')[-1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        worker_device = "cuda:0"

    pool_ref = [mp.Pool(
        processes=1,
        initializer=init_worker,
        initargs=(args.model_path, args.bert_path, worker_device)
    )]

    try:
        source_json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
        source_json_files = [f for f in source_json_files if not f.endswith(args.result_suffix)]

        if not source_json_files:
            return

        for json_path in sorted(source_json_files):
            process_json_file(
                json_path,
                args.result_suffix,
                pool_ref,
                args.model_path,
                args.bert_path,
                worker_device
            )
    finally:
        pool_ref[0].close()
        pool_ref[0].join()


if __name__ == "__main__":
    if torch.cuda.is_available():
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    main()