import json
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

def extract_humor_label_from_output(model_output):

    if not model_output:
        return None, False, "empty_output"


    valid_labels = ['true', 'false']


    cleaned_output = model_output.strip().lower()


    patterns = [
        r'^(true|false)$',
        r'\b(true|false)\b',
        r'answer[:\s]*(true|false)',
        r'final[:\s]*(true|false)',
        r'(true|false)\.?$',
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned_output)
        if match:
            label = match.group(1).lower()
            if label in valid_labels:
                return label, True, None


    if 'yes' in cleaned_output or 'humor' in cleaned_output:
        if 'no' not in cleaned_output and 'not' not in cleaned_output:
            return 'true', False, "inferred_from_yes_or_humor"
    elif 'no' in cleaned_output or 'not humor' in cleaned_output:
        return 'false', False, "inferred_from_no_or_not_humor"


    if any(word in cleaned_output for word in ['true', 'false']):
        return None, False, "label_found_but_not_extracted"
    else:
        return None, False, "no_label_pattern"

def evaluate_humor_understanding(result_file_path):



    with open(result_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    predictions = []
    ground_truths = []
    detailed_results = []
    extraction_errors = defaultdict(list)
    prediction_errors = defaultdict(list)


    humor_labels = ['false', 'true']


    for item in results:
        item_id = item['id']
        model_output = item['model_output']
        gt_label = item['ground_truth'].lower().strip()


        pred_label, is_valid, error_type = extract_humor_label_from_output(model_output)


        detailed_item = {
            'id': item_id,
            'model_output': model_output,
            'extracted_prediction': pred_label,
            'ground_truth': gt_label,
            'correct': pred_label == gt_label if pred_label else False,
            'valid': is_valid
        }
        detailed_results.append(detailed_item)


        if not is_valid:
            extraction_errors[error_type].append(item_id)
        elif pred_label != gt_label:
            error_pattern = f"{gt_label}_to_{pred_label}"
            prediction_errors[error_pattern].append(item_id)


        if pred_label:
            predictions.append(pred_label)
            ground_truths.append(gt_label)


    if len(predictions) == 0:
        return {
            'error': 'No valid predictions found',
            'total_samples': len(results),
            'extraction_errors': dict(extraction_errors)
        }


    accuracy = accuracy_score(ground_truths, predictions)
    weighted_f1 = f1_score(ground_truths, predictions, average='weighted')
    macro_f1 = f1_score(ground_truths, predictions, average='macro')


    cm = confusion_matrix(ground_truths, predictions, labels=humor_labels)


    class_report = classification_report(ground_truths, predictions,
                                       target_names=humor_labels,
                                       output_dict=True,
                                       zero_division=0)


    true_positives = cm[1, 1]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_negatives = cm[0, 0]

    humor_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    humor_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    humor_f1 = 2 * humor_precision * humor_recall / (humor_precision + humor_recall) if (humor_precision + humor_recall) > 0 else 0


    evaluation_result = {
        'task_info': {
            'task_name': 'humor.understanding',
            'dataset': 'UR-FUNNY',
            'evaluation_time': datetime.now().isoformat(),
            'total_samples': len(results),
            'valid_predictions': len(predictions),
            'extraction_success_rate': round(len(predictions) / len(results), 4)
        },
        'metrics': {
            'ACC': round(accuracy, 4),
            'WAF': round(weighted_f1, 4),
            'Macro_F1': round(macro_f1, 4),
            'Humor_Precision': round(humor_precision, 4),
            'Humor_Recall': round(humor_recall, 4),
            'Humor_F1': round(humor_f1, 4)
        },
        'per_class_metrics': {
            label: {
                'precision': round(class_report[label]['precision'], 4),
                'recall': round(class_report[label]['recall'], 4),
                'f1_score': round(class_report[label]['f1-score'], 4),
                'support': int(class_report[label]['support'])
            } for label in humor_labels if label in class_report
        },
        'confusion_matrix': {
            'labels': humor_labels,
            'matrix': cm.tolist(),
            'detailed': {
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives),
                'true_negatives': int(true_negatives)
            }
        },
        'error_analysis': {
            'extraction_errors': {
                error_type: {
                    'count': len(sample_ids),
                    'sample_ids': sample_ids
                } for error_type, sample_ids in extraction_errors.items()
            },
            'prediction_errors': {
                error_pattern: {
                    'count': len(sample_ids),
                    'sample_ids': sample_ids
                } for error_pattern, sample_ids in prediction_errors.items()
            }
        },
        'distribution': {
            'ground_truth': dict(Counter(ground_truths)),
            'predictions': dict(Counter(predictions))
        }
    }


    base_name = result_file_path.replace('.json', '')


    eval_output_file = f"{base_name}_evaluation.json"
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)


    detailed_output_file = f"{base_name}_detailed_results.json"
    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)


    problem_samples = [item for item in detailed_results if not item['correct']]
    if problem_samples:
        problem_report_file = f"{base_name}_problem_samples.json"
        with open(problem_report_file, 'w', encoding='utf-8') as f:
            json.dump(problem_samples, f, ensure_ascii=False, indent=2)


    print(f"Evaluation complete: {len(results)} samples")
    print(f"Key metrics: ACC={evaluation_result['metrics']['ACC']}, WAF={evaluation_result['metrics']['WAF']}")
    print(f"Humor detection: Precision={evaluation_result['metrics']['Humor_Precision']}, Recall={evaluation_result['metrics']['Humor_Recall']}, F1={evaluation_result['metrics']['Humor_F1']}")
    print(f"Extraction success rate: {evaluation_result['task_info']['extraction_success_rate']}")
    print(f"Results saved to: {eval_output_file}")
    if problem_samples:
        print(f"Problematic samples: {len(problem_samples)}; see {problem_report_file} for details")

    return evaluation_result


if __name__ == "__main__":
    result_file = "model_result.json"

    try:
        evaluation_result = evaluate_humor_understanding(result_file)

    except FileNotFoundError:
        print(f"Error: file not found {result_file}")
    except json.JSONDecodeError:
        print(f"Error: invalid format for {result_file}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")

