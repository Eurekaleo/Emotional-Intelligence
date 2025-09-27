import json
import re
import ast
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

def extract_stance_from_output(model_output):

    if not model_output:
        return None, False, "empty_output"


    valid_stances = ['support', 'refute', 'comment', 'unrelated']

    try:

        if "{'stance':" in model_output or '{"stance":' in model_output:

            cleaned_output = model_output.strip()


            json_match = re.search(r'\{[^}]*\}', cleaned_output)
            if json_match:
                cleaned_output = json_match.group()


            try:
                parsed = ast.literal_eval(cleaned_output)
            except:

                parsed = json.loads(cleaned_output)

            if 'stance' in parsed and isinstance(parsed['stance'], str):
                stance = parsed['stance'].lower().strip()
                if stance in valid_stances:
                    return stance, True, None
                else:
                    return stance, False, "invalid_stance_label"


        cleaned_output = model_output.lower().strip()


        for stance in valid_stances:
            if cleaned_output == stance:
                return stance, True, None


        for stance in valid_stances:
            if stance in cleaned_output:
                return stance, False, "stance_found_in_text"


        if 'oppose' in cleaned_output or 'against' in cleaned_output or 'deny' in cleaned_output:
            return 'refute', False, "inferred_from_synonyms"
        elif 'agree' in cleaned_output or 'favor' in cleaned_output or 'endorse' in cleaned_output:
            return 'support', False, "inferred_from_synonyms"
        elif 'neutral' in cleaned_output or 'discuss' in cleaned_output or 'mention' in cleaned_output:
            return 'comment', False, "inferred_from_synonyms"
        elif 'irrelevant' in cleaned_output or 'unconnected' in cleaned_output or 'off-topic' in cleaned_output:
            return 'unrelated', False, "inferred_from_synonyms"

        return None, False, "no_stance_pattern"

    except Exception as e:
        return None, False, f"parsing_error_{str(e)}"

def parse_ground_truth_stance(ground_truth):

    if isinstance(ground_truth, dict):
        return ground_truth.get('stance', '').lower().strip()
    elif isinstance(ground_truth, str):

        try:
            parsed = ast.literal_eval(ground_truth)
            if isinstance(parsed, dict) and 'stance' in parsed:
                return parsed['stance'].lower().strip()
        except:
            pass

        return ground_truth.lower().strip()
    else:
        return ''

def evaluate_multimodal_stance_detection(result_file_path):



    with open(result_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    predictions = []
    ground_truths = []
    detailed_results = []
    extraction_errors = defaultdict(list)
    prediction_errors = defaultdict(list)


    stance_labels = ['support', 'refute', 'comment', 'unrelated']


    for item in results:
        item_id = item['id']
        model_output = item['model_output']
        gt_stance = parse_ground_truth_stance(item['ground_truth'])


        pred_stance, is_valid, error_type = extract_stance_from_output(model_output)


        detailed_item = {
            'id': item_id,
            'model_output': model_output,
            'extracted_prediction': pred_stance,
            'ground_truth': gt_stance,
            'correct': pred_stance == gt_stance if pred_stance else False,
            'valid': is_valid
        }
        detailed_results.append(detailed_item)


        if not is_valid:
            extraction_errors[error_type].append(item_id)
        elif pred_stance != gt_stance:
            error_pattern = f"{gt_stance}_to_{pred_stance}"
            prediction_errors[error_pattern].append(item_id)


        if pred_stance:
            predictions.append(pred_stance)
            ground_truths.append(gt_stance)


    if len(predictions) == 0:
        return {
            'error': 'No valid predictions found',
            'total_samples': len(results),
            'extraction_errors': dict(extraction_errors)
        }


    accuracy = accuracy_score(ground_truths, predictions)
    micro_f1 = f1_score(ground_truths, predictions, average='micro')
    macro_f1 = f1_score(ground_truths, predictions, average='macro')
    weighted_f1 = f1_score(ground_truths, predictions, average='weighted')


    cm = confusion_matrix(ground_truths, predictions, labels=stance_labels)


    class_report = classification_report(ground_truths, predictions,
                                       target_names=stance_labels,
                                       output_dict=True,
                                       zero_division=0)


    per_class_metrics = {}
    for i, label in enumerate(stance_labels):
        if label in class_report:
            true_positives = cm[i, i]
            false_positives = np.sum(cm[:, i]) - true_positives
            false_negatives = np.sum(cm[i, :]) - true_positives

            per_class_metrics[label] = {
                'precision': round(class_report[label]['precision'], 4),
                'recall': round(class_report[label]['recall'], 4),
                'f1_score': round(class_report[label]['f1-score'], 4),
                'support': int(class_report[label]['support']),
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives)
            }



    support_refute_predictions = []
    support_refute_ground_truths = []
    for pred, true in zip(predictions, ground_truths):
        if true in ['support', 'refute']:
            support_refute_predictions.append(pred if pred in ['support', 'refute'] else 'other')
            support_refute_ground_truths.append(true)

    support_refute_accuracy = 0
    if len(support_refute_ground_truths) > 0:
        support_refute_accuracy = accuracy_score(support_refute_ground_truths, support_refute_predictions)


    evaluation_result = {
        'task_info': {
            'task_name': 'multimodal.stance.detection',
            'dataset': 'MMWTWT',
            'evaluation_time': datetime.now().isoformat(),
            'total_samples': len(results),
            'valid_predictions': len(predictions),
            'extraction_success_rate': round(len(predictions) / len(results), 4)
        },
        'metrics': {
            'ACC': round(accuracy, 4),
            'Micro_F1': round(micro_f1, 4),
            'Macro_F1': round(macro_f1, 4),
            'Weighted_F1': round(weighted_f1, 4),
            'Support_Refute_ACC': round(support_refute_accuracy, 4)
        },
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': {
            'labels': stance_labels,
            'matrix': cm.tolist(),
            'normalized': (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).round(4).tolist()
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
        },
        'stance_analysis': {
            'support_vs_refute_samples': len(support_refute_ground_truths),
            'comment_samples': ground_truths.count('comment'),
            'unrelated_samples': ground_truths.count('unrelated'),
            'most_confused_pairs': []
        }
    }


    confusion_pairs = []
    for i, label1 in enumerate(stance_labels):
        for j, label2 in enumerate(stance_labels):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'true_stance': label1,
                    'predicted_stance': label2,
                    'count': int(cm[i, j]),
                    'percentage': round(cm[i, j] / np.sum(cm[i, :]) * 100, 2)
                })

    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
    evaluation_result['stance_analysis']['most_confused_pairs'] = confusion_pairs[:5]


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
    print(f"Key metrics: ACC={evaluation_result['metrics']['ACC']}, Micro F1={evaluation_result['metrics']['Micro_F1']}")
    print(f"Support/Refute accuracy: {evaluation_result['metrics']['Support_Refute_ACC']}")
    print(f"Extraction success rate: {evaluation_result['task_info']['extraction_success_rate']}")
    print(f"Results saved to: {eval_output_file}")
    if problem_samples:
        print(f"Problematic samples: {len(problem_samples)}; see {problem_report_file} for details")

    if confusion_pairs:
        print("\nMost confusable stance pairs:")
        for pair in confusion_pairs[:3]:
            print(f"  {pair['true_stance']} → {pair['predicted_stance']}: {pair['count']} times ({pair['percentage']}%)")

    return evaluation_result


if __name__ == "__main__":
    result_file = "model_result.json"

    try:
        evaluation_result = evaluate_multimodal_stance_detection(result_file)

    except FileNotFoundError:
        print(f"Error: file not found {result_file}")
    except json.JSONDecodeError:
        print(f"Error: invalid format for {result_file}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
