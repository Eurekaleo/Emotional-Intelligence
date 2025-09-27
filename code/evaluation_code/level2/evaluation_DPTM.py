import json
import re
import ast
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

def extract_techniques_from_output(model_output):

    if not model_output:
        return [], False, "empty_output"


    valid_techniques = [
        "Appeal to authority", "Appeal to fear/prejudice", "Black-and-white Fallacy/Dictatorship",
        "Causal Oversimplification", "Doubt", "Exaggeration/Minimisation", "Flag-waving",
        "Glittering generalities (Virtue)", "Loaded Language",
        "Misrepresentation of Someone's Position (Straw Man)", "Name calling/Labeling",
        "Obfuscation, Intentional vagueness, Confusion", "Presenting Irrelevant Data (Red Herring)",
        "Reductio ad hitlerum", "Repetition", "Slogans", "Smears", "Thought-terminating cliché",
        "Whataboutism", "Bandwagon", "Transfer", "Appeal to (Strong) Emotions"
    ]

    try:

        if "{'techniques':" in model_output or '{"techniques":' in model_output:

            cleaned_output = model_output.strip()
            if not cleaned_output.startswith('{'):

                json_match = re.search(r'\{[^}]*\}', cleaned_output)
                if json_match:
                    cleaned_output = json_match.group()


            try:
                parsed = ast.literal_eval(cleaned_output)
            except:

                parsed = json.loads(cleaned_output)

            if 'techniques' in parsed and isinstance(parsed['techniques'], list):
                techniques = [tech.strip() for tech in parsed['techniques']]

                valid_techniques_found = [tech for tech in techniques if tech in valid_techniques]
                invalid_techniques = [tech for tech in techniques if tech not in valid_techniques]

                if invalid_techniques:
                    return valid_techniques_found, False, "invalid_technique_labels"
                return valid_techniques_found, True, None


        found_techniques = []
        for technique in valid_techniques:
            if technique in model_output:
                found_techniques.append(technique)

        if found_techniques:
            return found_techniques, False, "techniques_found_but_not_properly_formatted"

        return [], False, "no_techniques_pattern"

    except Exception as e:
        return [], False, f"parsing_error_{str(e)}"

def calculate_multilabel_metrics(y_true_list, y_pred_list, all_labels):


    y_true_binary = []
    y_pred_binary = []

    for y_true, y_pred in zip(y_true_list, y_pred_list):
        true_vector = [1 if label in y_true else 0 for label in all_labels]
        pred_vector = [1 if label in y_pred else 0 for label in all_labels]
        y_true_binary.append(true_vector)
        y_pred_binary.append(pred_vector)

    y_true_binary = np.array(y_true_binary)
    y_pred_binary = np.array(y_pred_binary)


    micro_f1 = f1_score(y_true_binary, y_pred_binary, average='micro')
    macro_f1 = f1_score(y_true_binary, y_pred_binary, average='macro')
    micro_precision = precision_score(y_true_binary, y_pred_binary, average='micro')
    micro_recall = recall_score(y_true_binary, y_pred_binary, average='micro')


    per_label_f1 = f1_score(y_true_binary, y_pred_binary, average=None)
    per_label_precision = precision_score(y_true_binary, y_pred_binary, average=None)
    per_label_recall = recall_score(y_true_binary, y_pred_binary, average=None)

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'per_label_metrics': {
            all_labels[i]: {
                'f1': per_label_f1[i],
                'precision': per_label_precision[i],
                'recall': per_label_recall[i]
            } for i in range(len(all_labels))
        }
    }

def evaluate_persuasion_techniques_detection(result_file_path):



    with open(result_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    predictions = []
    ground_truths = []
    detailed_results = []
    extraction_errors = defaultdict(list)


    all_techniques = [
        "Appeal to authority", "Appeal to fear/prejudice", "Black-and-white Fallacy/Dictatorship",
        "Causal Oversimplification", "Doubt", "Exaggeration/Minimisation", "Flag-waving",
        "Glittering generalities (Virtue)", "Loaded Language",
        "Misrepresentation of Someone's Position (Straw Man)", "Name calling/Labeling",
        "Obfuscation, Intentional vagueness, Confusion", "Presenting Irrelevant Data (Red Herring)",
        "Reductio ad hitlerum", "Repetition", "Slogans", "Smears", "Thought-terminating cliché",
        "Whataboutism", "Bandwagon", "Transfer", "Appeal to (Strong) Emotions"
    ]


    for item in results:
        item_id = item['id']
        model_output = item['model_output']
        gt_techniques = item['ground_truth']['techniques'] if isinstance(item['ground_truth'], dict) else item['ground_truth']


        pred_techniques, is_valid, error_type = extract_techniques_from_output(model_output)


        detailed_item = {
            'id': item_id,
            'model_output': model_output,
            'extracted_prediction': pred_techniques,
            'ground_truth': gt_techniques,
            'exact_match': set(pred_techniques) == set(gt_techniques) if is_valid else False,
            'valid': is_valid
        }
        detailed_results.append(detailed_item)


        if not is_valid:
            extraction_errors[error_type].append(item_id)


        if is_valid:
            predictions.append(pred_techniques)
            ground_truths.append(gt_techniques)


    if len(predictions) == 0:
        return {
            'error': 'No valid predictions found',
            'total_samples': len(results),
            'extraction_errors': dict(extraction_errors)
        }


    metrics = calculate_multilabel_metrics(ground_truths, predictions, all_techniques)


    exact_matches = sum(1 for item in detailed_results if item['exact_match'])
    exact_match_accuracy = exact_matches / len([item for item in detailed_results if item['valid']])


    all_true_labels = [label for labels in ground_truths for label in labels]
    all_pred_labels = [label for labels in predictions for label in labels]


    evaluation_result = {
        'task_info': {
            'task_name': 'detection.of.persuasion.techniques.in.memes',
            'dataset': 'SemEval-2021 Task 6',
            'evaluation_time': datetime.now().isoformat(),
            'total_samples': len(results),
            'valid_predictions': len(predictions),
            'extraction_success_rate': round(len(predictions) / len(results), 4)
        },
        'metrics': {
            'Micro_F1': round(metrics['micro_f1'], 4),
            'Macro_F1': round(metrics['macro_f1'], 4),
            'Micro_Precision': round(metrics['micro_precision'], 4),
            'Micro_Recall': round(metrics['micro_recall'], 4),
            'Exact_Match_Accuracy': round(exact_match_accuracy, 4)
        },
        'per_label_metrics': {
            label: {
                'f1': round(metrics['per_label_metrics'][label]['f1'], 4),
                'precision': round(metrics['per_label_metrics'][label]['precision'], 4),
                'recall': round(metrics['per_label_metrics'][label]['recall'], 4)
            } for label in all_techniques
        },
        'error_analysis': {
            'extraction_errors': {
                error_type: {
                    'count': len(sample_ids),
                    'sample_ids': sample_ids
                } for error_type, sample_ids in extraction_errors.items()
            }
        },
        'label_statistics': {
            'ground_truth_distribution': dict(Counter(all_true_labels)),
            'prediction_distribution': dict(Counter(all_pred_labels)),
            'avg_labels_per_sample': {
                'ground_truth': round(np.mean([len(labels) for labels in ground_truths]), 2),
                'predictions': round(np.mean([len(labels) for labels in predictions]), 2)
            }
        }
    }


    base_name = result_file_path.replace('.json', '')


    eval_output_file = f"{base_name}_evaluation.json"
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)


    detailed_output_file = f"{base_name}_detailed_results.json"
    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)


    problem_samples = [item for item in detailed_results if not item['exact_match']]
    if problem_samples:
        problem_report_file = f"{base_name}_problem_samples.json"
        with open(problem_report_file, 'w', encoding='utf-8') as f:
            json.dump(problem_samples, f, ensure_ascii=False, indent=2)


    print(f"Evaluation complete: {len(results)} samples")
    print(f"Key metrics: Micro F1={evaluation_result['metrics']['Micro_F1']}, Exact Match Acc={evaluation_result['metrics']['Exact_Match_Accuracy']}")
    print(f"Extraction success rate: {evaluation_result['task_info']['extraction_success_rate']}")
    print(f"Results saved to: {eval_output_file}")
    if problem_samples:
        print(f"Problematic samples: {len(problem_samples)}; see {problem_report_file} for details")

    return evaluation_result


if __name__ == "__main__":
    result_file = "model_result.json"

    try:
        evaluation_result = evaluate_persuasion_techniques_detection(result_file)

    except FileNotFoundError:
        print(f"Error: file not found {result_file}")
    except json.JSONDecodeError:
        print(f"Error: invalid format for {result_file}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")

