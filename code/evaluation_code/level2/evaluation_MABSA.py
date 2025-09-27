import json
import re
import ast
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

def extract_targets_from_prompt(prompt):


    targets_match = re.search(r"Targets:\s*([^\n]+)", prompt)
    if targets_match:
        targets_text = targets_match.group(1).strip()

        targets = [target.strip() for target in re.split(r'[,;]', targets_text)]
        return targets
    return []

def extract_sentiment_dict_from_output(model_output):

    if not model_output:
        return {}, False, "empty_output"


    valid_sentiments = ['positive', 'neutral', 'negative']

    try:

        if "{" in model_output and "}" in model_output:

            cleaned_output = model_output.strip()


            json_match = re.search(r'\{[^}]*\}', cleaned_output)
            if json_match:
                cleaned_output = json_match.group()


            try:
                parsed = ast.literal_eval(cleaned_output)
            except:

                parsed = json.loads(cleaned_output)

            if isinstance(parsed, dict):

                sentiment_dict = {}
                all_valid = True

                for target, sentiment in parsed.items():
                    if isinstance(sentiment, str) and sentiment.lower() in valid_sentiments:
                        sentiment_dict[target] = sentiment.lower()
                    else:
                        all_valid = False

                if all_valid and len(sentiment_dict) > 0:
                    return sentiment_dict, True, None
                else:
                    return sentiment_dict, False, "invalid_sentiment_labels"



        pairs = re.findall(r"([^:{},'\"]+):\s*['\"]?(positive|neutral|negative)['\"]?", model_output.lower())
        if pairs:
            sentiment_dict = {}
            for target, sentiment in pairs:
                target = target.strip(' \'"')
                sentiment_dict[target] = sentiment
            return sentiment_dict, False, "extracted_from_text_patterns"

        return {}, False, "no_sentiment_pattern"

    except Exception as e:
        return {}, False, f"parsing_error_{str(e)}"

def parse_ground_truth_dict(ground_truth):

    if isinstance(ground_truth, dict):
        return {k: v.lower() for k, v in ground_truth.items()}
    elif isinstance(ground_truth, str):
        try:
            parsed = ast.literal_eval(ground_truth)
            if isinstance(parsed, dict):
                return {k: v.lower() for k, v in parsed.items()}
        except:
            pass
    return {}

def calculate_multimodal_absa_metrics(predictions, ground_truths):


    all_pred_sentiments = []
    all_true_sentiments = []

    for pred_pairs, true_pairs in zip(predictions, ground_truths):
        pred_dict = dict(pred_pairs)
        true_dict = dict(true_pairs)


        common_targets = set(pred_dict.keys()) & set(true_dict.keys())

        for target in common_targets:
            all_pred_sentiments.append(pred_dict[target])
            all_true_sentiments.append(true_dict[target])

    if len(all_pred_sentiments) == 0:
        return {
            'micro_f1': 0.0,
            'macro_f1': 0.0,
            'micro_precision': 0.0,
            'micro_recall': 0.0,
            'per_class_metrics': {},
            'valid_pairs': 0
        }


    labels = ['positive', 'neutral', 'negative']
    micro_f1 = f1_score(all_true_sentiments, all_pred_sentiments, average='micro')
    macro_f1 = f1_score(all_true_sentiments, all_pred_sentiments, average='macro')
    micro_precision = precision_score(all_true_sentiments, all_pred_sentiments, average='micro')
    micro_recall = recall_score(all_true_sentiments, all_pred_sentiments, average='micro')


    class_report = classification_report(all_true_sentiments, all_pred_sentiments,
                                       target_names=labels,
                                       output_dict=True, zero_division=0)

    per_class_metrics = {}
    for label in labels:
        if label in class_report:
            per_class_metrics[label] = {
                'precision': class_report[label]['precision'],
                'recall': class_report[label]['recall'],
                'f1_score': class_report[label]['f1-score'],
                'support': int(class_report[label]['support'])
            }

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'per_class_metrics': per_class_metrics,
        'valid_pairs': len(all_pred_sentiments)
    }

def evaluate_multimodal_aspect_based_sentiment_analysis(result_file_path):



    with open(result_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    predictions = []
    ground_truths = []
    detailed_results = []
    extraction_errors = defaultdict(list)
    target_level_errors = defaultdict(list)


    for item in results:
        item_id = item['id']
        prompt = item['prompt']
        model_output = item['model_output']
        gt_dict = parse_ground_truth_dict(item['ground_truth'])


        targets = extract_targets_from_prompt(prompt)


        pred_dict, is_valid, error_type = extract_sentiment_dict_from_output(model_output)


        pred_pairs = []
        true_pairs = []
        target_results = {}

        for target in targets:
            if target in gt_dict:
                true_sentiment = gt_dict[target]
                true_pairs.append((target, true_sentiment))

                if target in pred_dict:
                    pred_sentiment = pred_dict[target]
                    pred_pairs.append((target, pred_sentiment))
                    target_results[target] = {
                        'predicted': pred_sentiment,
                        'ground_truth': true_sentiment,
                        'correct': pred_sentiment == true_sentiment
                    }


                    if pred_sentiment != true_sentiment:
                        error_pattern = f"{true_sentiment}_to_{pred_sentiment}"
                        target_level_errors[error_pattern].append(f"{item_id}_{target}")
                else:
                    target_results[target] = {
                        'predicted': None,
                        'ground_truth': true_sentiment,
                        'correct': False
                    }


        detailed_item = {
            'id': item_id,
            'targets': targets,
            'model_output': model_output,
            'extracted_prediction': pred_dict,
            'ground_truth': gt_dict,
            'target_results': target_results,
            'all_targets_correct': all(result['correct'] for result in target_results.values()),
            'valid': is_valid
        }
        detailed_results.append(detailed_item)


        if not is_valid:
            extraction_errors[error_type].append(item_id)


        if len(pred_pairs) > 0:
            predictions.append(pred_pairs)
            ground_truths.append(true_pairs)


    if len(predictions) == 0:
        return {
            'error': 'No valid predictions found',
            'total_samples': len(results),
            'extraction_errors': dict(extraction_errors)
        }


    metrics = calculate_multimodal_absa_metrics(predictions, ground_truths)


    all_correct_samples = sum(1 for item in detailed_results if item['all_targets_correct'])
    sample_level_accuracy = all_correct_samples / len(detailed_results)


    all_true_sentiments = []
    all_pred_sentiments = []
    for item in detailed_results:
        for target, result in item['target_results'].items():
            if result['ground_truth']:
                all_true_sentiments.append(result['ground_truth'])
            if result['predicted']:
                all_pred_sentiments.append(result['predicted'])


    evaluation_result = {
        'task_info': {
            'task_name': 'multimodal.aspect.based.sentiment.analysis',
            'dataset': 'Twitter2015/2017',
            'evaluation_time': datetime.now().isoformat(),
            'total_samples': len(results),
            'valid_samples': len(predictions),
            'extraction_success_rate': round(len(predictions) / len(results), 4),
            'total_target_pairs': metrics['valid_pairs']
        },
        'metrics': {
            'Micro_F1': round(metrics['micro_f1'], 4),
            'Macro_F1': round(metrics['macro_f1'], 4),
            'Micro_Precision': round(metrics['micro_precision'], 4),
            'Micro_Recall': round(metrics['micro_recall'], 4),
            'Sample_Level_Accuracy': round(sample_level_accuracy, 4)
        },
        'per_class_metrics': {
            label: {
                'precision': round(metrics['per_class_metrics'][label]['precision'], 4),
                'recall': round(metrics['per_class_metrics'][label]['recall'], 4),
                'f1_score': round(metrics['per_class_metrics'][label]['f1_score'], 4),
                'support': metrics['per_class_metrics'][label]['support']
            } for label in metrics['per_class_metrics']
        },
        'error_analysis': {
            'extraction_errors': {
                error_type: {
                    'count': len(sample_ids),
                    'sample_ids': sample_ids
                } for error_type, sample_ids in extraction_errors.items()
            },
            'target_level_errors': {
                error_pattern: {
                    'count': len(target_ids),
                    'target_ids': target_ids
                } for error_pattern, target_ids in target_level_errors.items()
            }
        },
        'distribution': {
            'ground_truth_sentiments': dict(Counter(all_true_sentiments)),
            'predicted_sentiments': dict(Counter(all_pred_sentiments))
        }
    }


    base_name = result_file_path.replace('.json', '')


    eval_output_file = f"{base_name}_evaluation.json"
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)


    detailed_output_file = f"{base_name}_detailed_results.json"
    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)


    problem_samples = [item for item in detailed_results if not item['all_targets_correct']]
    if problem_samples:
        problem_report_file = f"{base_name}_problem_samples.json"
        with open(problem_report_file, 'w', encoding='utf-8') as f:
            json.dump(problem_samples, f, ensure_ascii=False, indent=2)


    print(f"Evaluation complete: {len(results)} samples")
    print(f"Key metrics: Micro F1={evaluation_result['metrics']['Micro_F1']}, Sample-level Accuracy={evaluation_result['metrics']['Sample_Level_Accuracy']}")
    print(f"Number of target pairs: {evaluation_result['task_info']['total_target_pairs']}")
    print(f"Extraction success rate: {evaluation_result['task_info']['extraction_success_rate']}")
    print(f"Results saved to: {eval_output_file}")
    if problem_samples:
        print(f"Problematic samples: {len(problem_samples)}; see {problem_report_file} for details")

    return evaluation_result


if __name__ == "__main__":
    result_file = "model_result.json"

    try:
        evaluation_result = evaluate_multimodal_aspect_based_sentiment_analysis(result_file)

    except FileNotFoundError:
        print(f"Error: file not found {result_file}")
    except json.JSONDecodeError:
        print(f"Error: invalid format for {result_file}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")

