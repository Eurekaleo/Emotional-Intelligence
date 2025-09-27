import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import json
import argparse
import time
import sys
from typing import List, Dict, Any

from tqdm import tqdm
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import signal
import contextlib

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

class TimeoutError(Exception):
    pass


EMOTION_LLAMA_PATH = "PATH_TO_EMOTION_LLAMA_PROJECT"
if EMOTION_LLAMA_PATH not in sys.path:
    sys.path.append(EMOTION_LLAMA_PATH)

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, Chat, SeparatorStyle


from minigpt4.datasets.builders import *   # noqa
from minigpt4.models import *              # noqa
from minigpt4.processors import *          # noqa
from minigpt4.runners import *             # noqa
from minigpt4.tasks import *               # noqa

# --- Configuration ---
LEVEL_DIRS = ["level1", "level2", "level3"]
GENERIC_RESULT_PATTERN = "_result.json"
RESULT_SUFFIX = "_emotionllama_result.json"


_TAGS = [
    r"<s>\s*[INST]\s*", r"[/INST]",                 
    r"<image>.*?</image>", r"<img>.*?</img>",            
    r"<video>.*?</video>", r"<feature>.*?</feature>",    
    r"<VideoHere>", r"<FeatureHere>",                    
    r"<image>", r"</image>", r"<video>", r"</video>", r"<feature>", r"</feature>",
]
_TAGS_RE = re.compile("|".join(_TAGS), flags=re.IGNORECASE | re.DOTALL)

def clean_prompt_text(s: str) -> str:
    s = _TAGS_RE.sub("", s).strip()
    tail = '\nRespond ONLY with: {"emotion":"neutral|negative|positive"}'
    if "Respond ONLY with" not in s:
        s += tail
    return s

_JSON_RE = re.compile(r'\{\s*"emotion"\s*:\s*"(neutral|negative|positive)"\s*\}', re.IGNORECASE)
def extract_emotion_json(text: str) -> str:
    m = _JSON_RE.search(text)
    if m:
        return json.dumps({"emotion": m.group(1).lower()}, ensure_ascii=False)
    low = text.lower()
    if "negative" in low:
        return json.dumps({"emotion": "negative"}, ensure_ascii=False)
    if "positive" in low:
        return json.dumps({"emotion": "positive"}, ensure_ascii=False)
    return json.dumps({"emotion": "neutral"}, ensure_ascii=False)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def get_first_frame_pil(video_path: str):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"Cannot read frame from video file: {video_path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in VID_EXTS:
        return 'video'
    elif ext in IMG_EXTS:
        return 'image'
    else:
        return 'unknown'

@torch.inference_mode()
def process_single_sample(chat: Chat, media_full_path: str, prompt_text: str) -> str:
    try:
        chat_state = Conversation(
            system="",
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep=""
        )
        img_list = []
        media_type = get_media_type(media_full_path)
        if media_type == 'unknown':
            raise ValueError(f"Unsupported media type: {media_full_path}")

        if media_type == 'video':
            pil_image = get_first_frame_pil(media_full_path)
        else:  # image
            from PIL import Image
            pil_image = Image.open(media_full_path).convert("RGB")
        
        chat.upload_img(pil_image, chat_state, img_list)
        
        if len(img_list) > 0:
            chat.encode_img(img_list)

        clean_prompt = prompt_text.replace("<image>", "").replace("<video>", "").strip()
        chat.ask(clean_prompt, chat_state)
        
        model_output = chat.answer(conv=chat_state, img_list=img_list, temperature=0.1, max_new_tokens=500, max_length=2000)[0]
        return model_output
    except Exception as e:
        return f"ERROR: {str(e)}"

def text_only_fallback(chat: Chat, prompt_text: str) -> str:

    print("  [INFO] Executing text-only fallback...")
    try:
        img_list = [Image.new('RGB', (1, 1), 'black')]
        
        chat_state = Conversation(
            system="",
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep=""
        )
        
        chat.encode_img(img_list)

        clean_prompt = prompt_text.replace("<image>", "").replace("<video>", "").strip()
        chat.ask(clean_prompt, chat_state)
        
        model_output = chat.answer(conv=chat_state, img_list=[], temperature=0.1, max_new_tokens=500, max_length=2000)[0]
        return model_output
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"ERROR in text-only fallback: {str(e)}"

def process_task(task_path: str, chat: Chat):
    print(f"\n--- Processing Task: {os.path.basename(task_path)} ---")

    source_json_files = [
        f for f in os.listdir(task_path)
        if f.endswith(".json") and GENERIC_RESULT_PATTERN not in f and not f.endswith(RESULT_SUFFIX)
    ]
    if not source_json_files:
        print(f"   No source JSON files found in {task_path}.")
        return

    for json_filename in source_json_files:
        dataset_json_path = os.path.join(task_path, json_filename)
        result_json_path = os.path.join(task_path, f"{os.path.splitext(json_filename)[0]}{RESULT_SUFFIX}")

        if os.path.exists(result_json_path):
            print(f"  Result file already exists, skipping: {os.path.basename(result_json_path)}")
            continue

        print(f"  Reading and processing dataset: {json_filename}")
        try:
            with open(dataset_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  Could not read or parse JSON file {dataset_json_path}: {e}")
            continue

        all_results: List[Dict[str, Any]] = []
        for item in tqdm(data, desc=f"  Processing {json_filename}"):
            start_time = time.time()
            model_output = ""
            prompt = ""
            ground_truth = ""
            try:
                prompt = item["conversations"][0]["value"]
                ground_truth = item["conversations"][1]["value"]

                media_relative_path = None

                if 'image' in item:
                    media_relative_path = item.get('image')
                elif 'video' in item:
                    media_relative_path = item.get('video')
              
                elif 'conversations' in item and item['conversations'] and isinstance(item['conversations'][0], dict):
                    conv0 = item['conversations'][0]
                    if 'image' in conv0:
                        media_relative_path = conv0.get('image')
                    elif 'video' in conv0:
                        media_relative_path = conv0.get('video')


                if not media_relative_path:
                    print(f"\n  Could not find media key for item {item.get('id', 'N/A')}. Falling back to text-only.")
                    model_output = text_only_fallback(chat=chat, prompt_text=prompt)
                else:
                    media_full_path = os.path.join(task_path, media_relative_path)
                    if not os.path.exists(media_full_path):
                        raise FileNotFoundError(f"Media file not found: {media_full_path}")

                    try:
                        with timeout(seconds=300):
                            model_output = process_single_sample(
                                chat=chat,
                                media_full_path=media_full_path,
                                prompt_text=prompt,
                            )
                    except TimeoutError:
                        print(f"\n  Processing timed out for item {item.get('id', 'N/A')}. Falling back to text-only.")
                        model_output = text_only_fallback(
                            chat=chat,
                            prompt_text=prompt
                        )
            except Exception as e:
                model_output = f"ERROR: {str(e)}"

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
        print(f"Task complete. Results saved to: {result_json_path}")

def init_model(cfg_path: str, device: str):

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        sys.exit(1)

    args = argparse.Namespace(cfg_path=cfg_path, options=None)
    cfg = Config(args)
    model_config = cfg.model_cfg

    model_config.low_resource = False

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)


    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import contextlib
    @contextlib.contextmanager
    def _fp16_autocast_cm():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            yield
    model.maybe_autocast = lambda *a, **k: _fp16_autocast_cm()

    # Initialize visual processor
    try:
        vis_processor_cfg = cfg.datasets_cfg.feature_face_caption.vis_processor.train
    except Exception:
        vis_processor_cfg = cfg.datasets_cfg.cc_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    model.eval()
    chat = Chat(model, vis_processor, device=device)

    if hasattr(chat, "answer_prepare"):
        _orig_answer_prepare = chat.answer_prepare

        def _answer_prepare_sane(*args, **kwargs):
            out = _orig_answer_prepare(*args, **kwargs)
            if isinstance(out, dict) and "inputs_embeds" in out:
                emb = out["inputs_embeds"]
                if isinstance(emb, torch.Tensor):
                    ref_param = next(model.llama_model.parameters())
                    target_device = ref_param.device
                    target_dtype  = ref_param.dtype  
                    emb = emb.to(device=target_device, dtype=target_dtype).contiguous()
                    out["inputs_embeds"] = emb

            for k in ("do_sample", "top_p", "repetition_penalty", "length_penalty", "num_beams"):
                out.pop(k, None)
            return out

        chat.answer_prepare = _answer_prepare_sane
    return chat, cfg

# ---------------- Main Function ----------------
def main():
    parser = argparse.ArgumentParser(description="Batch inference for task with Emotion-LLaMA.")
    parser.add_argument(
        "--cfg-path",
        default=os.path.join(EMOTION_LLAMA_PATH, " example_config/demo.yaml"),
        help="Path to the Emotion-LLaMA configuration file.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0", help="Device to run on.")
    args = parser.parse_args()

    if "cuda" in args.device:
        if not torch.cuda.is_available():
            print(f"CUDA device '{args.device}' is not available.")
            sys.exit(1)
        torch.cuda.set_device(args.device)

    chat, cfg = init_model(args.cfg_path, args.device)


    dataset_dir = os.getcwd()
    print(f"Running in directory: {dataset_dir}")

    for level_dir in LEVEL_DIRS:
        level_path = os.path.join(dataset_dir, level_dir)
        if not os.path.isdir(level_path):
            continue

        task_dirs = sorted([d.path for d in os.scandir(level_path) if d.is_dir()])
        for task_path in task_dirs:
            process_task(
                task_path,
                chat,
            )


if __name__ == "__main__":
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    main()