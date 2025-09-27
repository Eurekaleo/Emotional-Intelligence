import json
import re
import openai
from collections import Counter, defaultdict
from datetime import datetime
import time

def extract_numbered_list(text):

    if not text:
        return []


    patterns = [
        r'^\s*(\d+)\.\s*(.+)$',
        r'^\s*(\d+)\)\s*(.+)$',
        r'^\s*\((\d+)\)\s*(.+)$',
    ]

    items = []
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        matched = False
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                number = int(match.group(1))
                content = match.group(2).strip()
                items.append(content)
                matched = True
                break


        if not matched and len(line) > 5:
            items.append(line)

    return items

def evaluate_with_gpt4(prediction_text, ground_truth_text, emotion, client):

    try:
        prompt = f"""You are evaluating emotion interpretation explanations.

Task: Compare the predicted explanation with the reference explanation for the emotion "{emotion}".

Reference explanation:
{ground_truth_text}

Predicted explanation:
{prediction_text}

Evaluate if the predicted explanation is reasonable and accurate compared to the reference.

Respond with only "CORRECT" or "INCORRECT" based on whether the predicted explanation is of acceptable quality compared to the reference."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )

        result = response.choices[0].message.content.strip().upper()
        is_correct = result == "CORRECT"

        return {
            "is_correct": is_correct,
            "llm_response": result
        }

    except Exception as e:
        return {
            "is_correct": False,
            "llm_response": f"Error: {str(e)}"
        }

def extract_emotion_from_prompt(prompt):
 
    emotion_match = re.search(r"Emotion to explain:\s*(\w+)", prompt)
    if emotion_match:
        return emotion_match.group(1)
    return "Unknown"

def evaluate_emotion_interpretation(result_file_path, api_key):



    client = openai.OpenAI(api_key=api_key)


    with open(result_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    detailed_results = []
    extraction_errors = defaultdict(list)

    correct_count = 0
    total_valid = 0


    for i, item in enumerate(results):
        item_id = item['id']
        prompt = item['prompt']
        model_output = item['model_output']
        ground_truth = item['ground_truth']

        print(f"Processing sample {i+1}/{len(results)}: {item_id}")

        emotion = extract_emotion_from_prompt(prompt)


        pred_items = extract_numbered_list(model_output)
        gt_items = extract_numbered_list(ground_truth)


        has_valid_format = len(pred_items) > 0


        detailed_item = {
            'id': item_id,
            'emotion': emotion,
            'model_output': model_output,
            'ground_truth': ground_truth,
            'extracted_prediction': pred_items,
            'extracted_ground_truth': gt_items,
            'prediction_count': len(pred_items),
            'ground_truth_count': len(gt_items),
            'has_valid_format': has_valid_format,
            'llm_evaluation': None,
            'is_correct': False
        }


        if has_valid_format:
            llm_eval = evaluate_with_gpt4(model_output, ground_truth, emotion, client)
            detailed_item['llm_evaluation'] = llm_eval
            detailed_item['is_correct'] = llm_eval['is_correct']

            if llm_eval['is_correct']:
                correct_count += 1
            total_valid += 1


            time.sleep(0.5)
        else:
            extraction_errors['invalid_format'].append(item_id)

        detailed_results.append(detailed_item)


    format_success_rate = total_valid / len(results) if len(results) > 0 else 0
    accuracy = correct_count / total_valid if total_valid > 0 else 0


    emotion_distribution = Counter([item['emotion'] for item in detailed_results])


    pred_lengths = [item['prediction_count'] for item in detailed_results if item['has_valid_format']]
    gt_lengths = [item['ground_truth_count'] for item in detailed_results]

    avg_pred_length = sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0
    avg_gt_length = sum(gt_lengths) / len(gt_lengths) if gt_lengths else 0


    evaluation_result = {
        'task_info': {
            'task_name': 'emotion.interpretation',
            'dataset': 'EIBench',
            'evaluation_time': datetime.now().isoformat(),
            'total_samples': len(results),
            'valid_samples': total_valid,
            'format_success_rate': round(format_success_rate, 4)
        },
        'metrics': {
            'LLM_ACC': round(accuracy, 4),
            'Correct_Count': correct_count,
            'Total_Valid': total_valid
        },
        'content_analysis': {
            'avg_prediction_length': round(avg_pred_length, 2),
            'avg_ground_truth_length': round(avg_gt_length, 2),
            'emotion_distribution': dict(emotion_distribution)
        },
        'error_analysis': {
            'extraction_errors': {
                error_type: {
                    'count': len(sample_ids),
                    'sample_ids': sample_ids
                } for error_type, sample_ids in extraction_errors.items()
            }
        }
    }


    emotion_analysis = {}
    for emotion in emotion_distribution.keys():
        emotion_samples = [item for item in detailed_results if item['emotion'] == emotion and item['has_valid_format']]
        if emotion_samples:
            emotion_correct = sum(1 for item in emotion_samples if item['is_correct'])
            emotion_total = len(emotion_samples)
            emotion_acc = emotion_correct / emotion_total if emotion_total > 0 else 0

            emotion_analysis[emotion] = {
                'sample_count': emotion_total,
                'correct_count': emotion_correct,
                'accuracy': round(emotion_acc, 4),
                'avg_prediction_length': round(sum([item['prediction_count'] for item in emotion_samples]) / len(emotion_samples), 2)
            }

    evaluation_result['emotion_analysis'] = emotion_analysis


    base_name = result_file_path.replace('.json', '')


    eval_output_file = f"{base_name}_evaluation.json"
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)


    detailed_output_file = f"{base_name}_detailed_results.json"
    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)


    incorrect_samples = [item for item in detailed_results if item['has_valid_format'] and not item['is_correct']]
    if incorrect_samples:
        incorrect_report_file = f"{base_name}_incorrect_samples.json"
        with open(incorrect_report_file, 'w', encoding='utf-8') as f:
            json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)


    print(f"\nEvaluation complete: {len(results)} samples")
    print(f"Key metric: LLM_ACC={evaluation_result['metrics']['LLM_ACC']}")
    print(f"Correct count: {evaluation_result['metrics']['Correct_Count']}/{evaluation_result['metrics']['Total_Valid']}")
    print(f"Format success rate: {evaluation_result['task_info']['format_success_rate']}")
    print(f"Average prediction length: {evaluation_result['content_analysis']['avg_prediction_length']}")
    print(f"Results saved to: {eval_output_file}")
    if incorrect_samples:
        print(f"Incorrect samples: {len(incorrect_samples)}; see {incorrect_report_file} for details")


    if emotion_analysis:
        print("\nPerformance by emotion type:")
        for emotion, analysis in emotion_analysis.items():
            print(f"  {emotion}: ACC={analysis['accuracy']}, Samples={analysis['sample_count']}")

    return evaluation_result


if __name__ == "__main__":
    result_file = "model_result.json"
    api_key = "xxx"

    try:
        evaluation_result = evaluate_emotion_interpretation(result_file, api_key)

    except FileNotFoundError:
        print(f"Error: file not found {result_file}")
    except json.JSONDecodeError:
        print(f"Error: invalid format for {result_file}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")

