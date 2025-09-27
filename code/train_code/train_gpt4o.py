import json
import os
import time
from openai import OpenAI


API_KEY = os.environ.get("OPENAI_API_KEY", "api-key")
client = OpenAI(api_key=API_KEY)
MODEL_NAME = "gpt-4o"                          

INPUT_FILE = "example.json"                      
OUTPUT_FILE = "gpt4o_results.json"             

TEST_MODE = True                             
TEST_ITEMS = 5                                 

MAX_RETRY = 3                                  
SLEEP_BETWEEN_RETRY = 3                   
REASONING_EFFORT = "high"                      

def call_gpt4o_with_retry(prompt, attempt_num=1):
    for retry in range(MAX_RETRY):
        try:

            response = client.responses.create(
                model=MODEL_NAME,
                reasoning={"effort": REASONING_EFFORT},
                input=[{
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}]
                }]
            )

            return response.output_text

        except Exception as e:
            print(f" Error on attempt {retry + 1}: {str(e)}")
            if retry < MAX_RETRY - 1:
                print(f" Retrying in {SLEEP_BETWEEN_RETRY} seconds...")
                time.sleep(SLEEP_BETWEEN_RETRY)
            else:
                print(f"Failed after {MAX_RETRY} attempts")
                return None
    return None


def extract_prompt_and_truth(conversations):
    prompt = None
    ground_truth = None

    if conversations and len(conversations) > 0 and conversations[0]["from"] == "human":
        prompt = conversations[0]["value"].strip()

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
        conversations = item.get("conversations", [])

        print(f"\n[{idx}/{len(items_to_process)}] Processing ID: {item_id}")

        if not conversations:
            print(f"Skipping: missing conversations")
            continue

        prompt, ground_truth = extract_prompt_and_truth(conversations)

        if not prompt:
            print(f"Skipping: no prompt found")
            continue

        print(f"  Ground Truth: {ground_truth}")

        response1 = call_gpt4o_with_retry(prompt, attempt_num=1)

        response2 = call_gpt4o_with_retry(prompt, attempt_num=2)

        result = {
            "id": item_id,
            "response_1": response1,
            "response_2": response2,
            "ground_truth": ground_truth
        }
        results.append(result)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()