import json
import re
import openai
from collections import Counter, defaultdict
from datetime import datetime
import time

def check_required_format(text):

    if not text:
        return False, ""


    pattern = r'^The audience laughed because\s+'
    match = re.search(pattern, text, re.IGNORECASE)

    if match:

        content = text[match.end():].strip()
        return True, content
    else:
        return False, text.strip()

def evaluate_with_gpt4(prediction_text, ground_truth_text, client):

    try:
        prompt = f"""Compare the predicted explanation with the reference explanation for why the audience laughed.

Reference explanation:
{ground_truth_text}

Predicted explanation:
{prediction_text}

Does the predicted explanation correctly identify the reason for laughter? The prediction doesn't need to be identical to the reference, but should accurately capture why the audience laughed.

Respond with only "CORRECT" or "INCORRECT"."""

        response = client.chat.completions.create(
            model="gpt-4",
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

def evaluate_laughter_reasoning(result_file_path, api_key):



    client = openai.OpenAI(api_key=api_key)


    with open(result_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    detailed_results = []
    format_errors = defaultdict(list)

    correct_count = 0
    total_valid = 0


    for i, item in enumerate(results):
        item_id = item['id']
        model_output = item['model_output']
        ground_truth = item['ground_truth']

        print(f"Processing sample {i+1}/{len(results)}: {item_id}")

        has_correct_format, cleaned_content = check_required_format(model_output)


        detailed_item = {
            'id': item_id,
            'model_output': model_output,
            'ground_truth': ground_truth,
            'has_correct_format': has_correct_format,
            'cleaned_content': cleaned_content,
            'llm_evaluation': None,
            'is_correct': False
        }


        if has_correct_format and cleaned_content:

            _, gt_cleaned = check_required_format(ground_truth)
            if not gt_cleaned:
                gt_cleaned = ground_truth

            llm_eval = evaluate_with_gpt4(cleaned_content, gt_cleaned, client)
            detailed_item['llm_evaluation'] = llm_eval
            detailed_item['is_correct'] = llm_eval['is_correct']

            if llm_eval['is_correct']:
                correct_count += 1
            total_valid += 1


            time.sleep(0.5)
        else:
            if not has_correct_format:
                format_errors['incorrect_format'].append(item_id)
            else:
                format_errors['empty_content'].append(item_id)

        detailed_results.append(detailed_item)


    format_success_rate = total_valid / len(results) if len(results) > 0 else 0
    accuracy = correct_count / total_valid if total_valid > 0 else 0


    valid_contents = [item['cleaned_content'] for item in detailed_results if item['has_correct_format'] and item['cleaned_content']]
    avg_content_length = sum(len(content.split()) for content in valid_contents) / len(valid_contents) if valid_contents else 0


    correct_samples = [item for item in detailed_results if item['has_correct_format'] and item['is_correct']]
    incorrect_samples = [item for item in detailed_results if item['has_correct_format'] and not item['is_correct']]


    evaluation_result = {
        'task_info': {
            'task_name': 'laughter.reasoning',
            'dataset': 'SMILE',
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
            'avg_content_length_words': round(avg_content_length, 2)
        },
        'error_analysis': {
            'format_errors': {
                error_type: {
                    'count': len(sample_ids),
                    'sample_ids': sample_ids
                } for error_type, sample_ids in format_errors.items()
            }
        }
    }


    if correct_samples:
        correct_avg_length = sum(len(item['cleaned_content'].split()) for item in correct_samples) / len(correct_samples)
        evaluation_result['content_analysis']['correct_samples_avg_length'] = round(correct_avg_length, 2)

    if incorrect_samples:
        incorrect_avg_length = sum(len(item['cleaned_content'].split()) for item in incorrect_samples) / len(incorrect_samples)
        evaluation_result['content_analysis']['incorrect_samples_avg_length'] = round(incorrect_avg_length, 2)


    base_name = result_file_path.replace('.json', '')


    eval_output_file = f"{base_name}_evaluation.json"
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)


    detailed_output_file = f"{base_name}_detailed_results.json"
    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)


    if incorrect_samples:
        incorrect_report_file = f"{base_name}_incorrect_samples.json"
        with open(incorrect_report_file, 'w', encoding='utf-8') as f:
            json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)


    format_error_samples = [item for item in detailed_results if not item['has_correct_format']]
    if format_error_samples:
        format_error_report_file = f"{base_name}_format_error_samples.json"
        with open(format_error_report_file, 'w', encoding='utf-8') as f:
            json.dump(format_error_samples, f, ensure_ascii=False, indent=2)


    print(f"\nEvaluation complete: {len(results)} samples")
    print(f"Key metric: LLM_ACC={evaluation_result['metrics']['LLM_ACC']}")
    print(f"Correct count: {evaluation_result['metrics']['Correct_Count']}/{evaluation_result['metrics']['Total_Valid']}")
    print(f"Format success rate: {evaluation_result['task_info']['format_success_rate']}")
    print(f"Average content length: {evaluation_result['content_analysis']['avg_content_length_words']} words")
    print(f"Results saved to: {eval_output_file}")

    if incorrect_samples:
        print(f"Incorrect samples: {len(incorrect_samples)}; see {incorrect_report_file} for details")
    if format_error_samples:
        print(f"Format-error samples: {len(format_error_samples)}; see {format_error_report_file} for details")

    return evaluation_result


if __name__ == "__main__":
    result_file = "model_result.json"
    api_key = "xxx"

    try:
        evaluation_result = evaluate_laughter_reasoning(result_file, api_key)

    except FileNotFoundError:
        print(f"Error: file not found {result_file}")
    except json.JSONDecodeError:
        print(f"Error: invalid format for {result_file}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")

