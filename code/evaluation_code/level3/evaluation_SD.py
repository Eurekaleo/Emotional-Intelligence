import json
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

def parse_classification_output(output_text):

    if not output_text:
        return None, False, "empty_output"


    cleaned_output = output_text.strip().lower()


    if re.search(r'\btrue\b', cleaned_output):
        return "true", True, None
    elif re.search(r'\bfalse\b', cleaned_output):
        return "false", True, None
    else:
        return None, False, "no_valid_label"

def calculate_weighted_average_f1(y_true, y_pred, labels):

    return f1_score(y_true, y_pred, labels=labels, average='weighted')

def evaluate_sarcasm_detection(result_file_path):


    with open(result_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    predictions = []
    ground_truths = []
    detailed_results = []
    parsing_errors = defaultdict(list)


    for item in results:
        item_id = item['id']
        model_output = item['model_output']
        ground_truth = item['ground_truth']


        pred_label, pred_valid, pred_error = parse_classification_output(model_output)


        gt_label = ground_truth.strip().lower() if isinstance(ground_truth, str) else str(ground_truth).strip().lower()


        detailed_item = {
            'id': item_id,
            'model_output': model_output,
            'ground_truth': ground_truth,
            'extracted_prediction': pred_label,
            'standardized_ground_truth': gt_label,
            'prediction_valid': pred_valid,
            'prediction_error': pred_error
        }
        detailed_results.append(detailed_item)


        if not pred_valid:
            parsing_errors[pred_error].append(item_id)


        if pred_valid and gt_label in ['true', 'false']:
            predictions.append(pred_label)
            ground_truths.append(gt_label)


    if len(predictions) == 0:
        return {
            'error': 'No valid predictions found',
            'total_samples': len(results),
            'parsing_errors': dict(parsing_errors)
        }


    labels = ['true', 'false']


    accuracy = accuracy_score(ground_truths, predictions)
    weighted_f1 = calculate_weighted_average_f1(ground_truths, predictions, labels)


    precision_scores = precision_score(ground_truths, predictions, labels=labels, average=None, zero_division=0)
    recall_scores = recall_score(ground_truths, predictions, labels=labels, average=None, zero_division=0)
    f1_scores = f1_score(ground_truths, predictions, labels=labels, average=None, zero_division=0)


    macro_precision = precision_score(ground_truths, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(ground_truths, predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(ground_truths, predictions, average='macro', zero_division=0)


    cm = confusion_matrix(ground_truths, predictions, labels=labels)


    true_distribution = Counter(ground_truths)
    pred_distribution = Counter(predictions)


    class_metrics = {}
    for i, label in enumerate(labels):
        class_metrics[label] = {
            'precision': round(precision_scores[i], 4),
            'recall': round(recall_scores[i], 4),
            'f1_score': round(f1_scores[i], 4),
            'support': true_distribution.get(label, 0)
        }


    evaluation_result = {
        'task_info': {
            'task_name': 'sarcasm.detection',
            'dataset': 'MUStARD',
            'task_type': '2-CLS',
            'evaluation_time': datetime.now().isoformat(),
            'total_samples': len(results),
            'valid_samples': len(predictions),
            'parsing_success_rate': round(len(predictions) / len(results), 4)
        },
        'metrics': {
            'ACC': round(accuracy, 4),
            'WAF': round(weighted_f1, 4),
            'Macro_Precision': round(macro_precision, 4),
            'Macro_Recall': round(macro_recall, 4),
            'Macro_F1': round(macro_f1, 4)
        },
        'class_metrics': class_metrics,
        'confusion_matrix': {
            'matrix': cm.tolist(),
            'labels': labels
        },
        'distribution_analysis': {
            'ground_truth_distribution': dict(true_distribution),
            'prediction_distribution': dict(pred_distribution)
        },
        'error_analysis': {
            'parsing_errors': {
                error_type: {
                    'count': len(sample_ids),
                    'sample_ids': sample_ids
                } for error_type, sample_ids in parsing_errors.items()
            }
        }
    }


    if len(labels) == 2:

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else [0, 0, 0, 0]


        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        evaluation_result['binary_metrics'] = {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'sensitivity_recall': round(sensitivity, 4),
            'specificity': round(specificity, 4)
        }


    error_samples = []
    correct_samples = []

    for i, (pred, true, item) in enumerate(zip(predictions, ground_truths,
                                              [d for d in detailed_results if d['prediction_valid']])):
        if pred != true:
            error_samples.append({
                'id': item['id'],
                'predicted': pred,
                'ground_truth': true,
                'model_output': item['model_output']
            })
        else:
            correct_samples.append(item['id'])

    evaluation_result['sample_analysis'] = {
        'correct_samples_count': len(correct_samples),
        'error_samples_count': len(error_samples),
        'error_rate': round(len(error_samples) / len(predictions), 4) if predictions else 0
    }


    base_name = result_file_path.replace('.json', '')


    eval_output_file = f"{base_name}_evaluation.json"
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)


    detailed_output_file = f"{base_name}_detailed_results.json"
    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)


    if error_samples:
        error_report_file = f"{base_name}_error_samples.json"
        with open(error_report_file, 'w', encoding='utf-8') as f:
            json.dump(error_samples, f, ensure_ascii=False, indent=2)


    parsing_error_samples = [item for item in detailed_results if not item['prediction_valid']]
    if parsing_error_samples:
        parsing_error_report_file = f"{base_name}_parsing_error_samples.json"
        with open(parsing_error_report_file, 'w', encoding='utf-8') as f:
            json.dump(parsing_error_samples, f, ensure_ascii=False, indent=2)


    print(f"Evaluation complete: {len(results)} samples")
    print(f"Key metrics: ACC={evaluation_result['metrics']['ACC']}, WAF={evaluation_result['metrics']['WAF']}")
    print(f"Macro-averaged metrics: Precision={evaluation_result['metrics']['Macro_Precision']}, Recall={evaluation_result['metrics']['Macro_Recall']}, F1={evaluation_result['metrics']['Macro_F1']}")
    print(f"Parsing success rate: {evaluation_result['task_info']['parsing_success_rate']}")


    print("\nPer-class metrics:")
    for label, metrics in evaluation_result['class_metrics'].items():
        print(f"  {label.upper()}: P={metrics['precision']}, R={metrics['recall']}, F1={metrics['f1_score']}, Support={metrics['support']}")


    print(f"\nConfusion matrix:")
    print(f"        Predicted")
    print(f"Actual  false  true")
    for i, true_label in enumerate(labels):
        row_str = f"{true_label:>5}   "
        for j, pred_label in enumerate(labels):
            row_str += f"{cm[i][j]:>5}  "
        print(row_str)


    print(f"\nLabel distribution:")
    print(f"True labels: {dict(true_distribution)}")
    print(f"Predicted labels: {dict(pred_distribution)}")

    print(f"\nResults saved to: {eval_output_file}")
    if error_samples:
        print(f"Error samples: {len(error_samples)}; see {error_report_file} for details")
    if parsing_error_samples:
        print(f"Parsing-error samples: {len(parsing_error_samples)}; see {parsing_error_report_file} for details")

    return evaluation_result


if __name__ == "__main__":
    result_file = "model_result.json"

    try:
        evaluation_result = evaluate_sarcasm_detection(result_file)

    except FileNotFoundError:
        print(f"Error: file not found {result_file}")
    except json.JSONDecodeError:
        print(f"Error: invalid format for {result_file}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")

