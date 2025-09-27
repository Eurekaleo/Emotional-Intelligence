import json
import re
import ast
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

def parse_quintuple_list(output_text):

    if not output_text:
        return [], False, "empty_output"

    try:

        cleaned_output = output_text.strip()


        list_match = re.search(r'\[(.*)\]', cleaned_output, re.DOTALL)
        if not list_match:
            return [], False, "no_list_structure"

        list_content = list_match.group(1).strip()
        if not list_content:
            return [], True, None


        try:
            full_list = ast.literal_eval('[' + list_content + ']')
        except:


            fixed_content = list_content

            fixed_content = re.sub(r'"([^"]*)"', r"'\1'", fixed_content)

            try:
                full_list = ast.literal_eval('[' + fixed_content + ']')
            except:

                return parse_tuples_manually(list_content)


        quintuples = []
        for item in full_list:
            if isinstance(item, (tuple, list)) and len(item) >= 5:

                quintuple = tuple(str(element).strip() for element in item[:5])
                quintuples.append(quintuple)
            else:
                return [], False, "invalid_tuple_structure"

        return quintuples, True, None

    except Exception as e:
        return [], False, f"parsing_error_{str(e)}"

def parse_tuples_manually(list_content):

    try:

        tuple_pattern = r'\(\s*([^)]+)\s*\)'
        matches = re.findall(tuple_pattern, list_content)

        quintuples = []
        for match in matches:

            elements = []
            current_element = ""
            in_quotes = False
            quote_char = None

            i = 0
            while i < len(match):
                char = match[i]

                if char in ['"', "'"] and (i == 0 or match[i-1] != '\\'):
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                elif char == ',' and not in_quotes:
                    elements.append(current_element.strip().strip('"\''))
                    current_element = ""
                    i += 1
                    continue

                current_element += char
                i += 1


            if current_element:
                elements.append(current_element.strip().strip('"\''))


            if len(elements) >= 5:
                quintuple = tuple(elements[:5])
                quintuples.append(quintuple)

        return quintuples, len(quintuples) > 0, "manual_parsing" if len(quintuples) > 0 else "manual_parsing_failed"

    except Exception as e:
        return [], False, f"manual_parsing_error_{str(e)}"

def normalize_quintuple(quintuple):

    holder, target, aspect, opinion, sentiment = quintuple


    sentiment_lower = sentiment.lower().strip()
    if sentiment_lower in ['positive', 'pos']:
        sentiment = 'positive'
    elif sentiment_lower in ['negative', 'neg']:
        sentiment = 'negative'
    elif sentiment_lower in ['neutral', 'neu']:
        sentiment = 'neutral'
    else:
        sentiment = sentiment_lower


    holder = holder.strip()
    target = target.strip()
    aspect = aspect.strip()
    opinion = opinion.strip()

    return (holder, target, aspect, opinion, sentiment)

def calculate_quintuple_metrics(predictions, ground_truths):

    total_pred = 0
    total_true = 0
    total_correct = 0

    exact_matches = 0
    partial_matches = {'holder': 0, 'target': 0, 'aspect': 0, 'opinion': 0, 'sentiment': 0}

    for pred_list, true_list in zip(predictions, ground_truths):

        pred_normalized = [normalize_quintuple(q) for q in pred_list]
        true_normalized = [normalize_quintuple(q) for q in true_list]

        total_pred += len(pred_normalized)
        total_true += len(true_normalized)


        pred_set = set(pred_normalized)
        true_set = set(true_normalized)
        exact_matches += len(pred_set & true_set)
        total_correct += len(pred_set & true_set)


        for pred_q in pred_normalized:
            for true_q in true_normalized:
                if pred_q == true_q:
                    continue
                for i, (field_name, pred_field, true_field) in enumerate(
                    zip(['holder', 'target', 'aspect', 'opinion', 'sentiment'], pred_q, true_q)):
                    if pred_field.lower() == true_field.lower():
                        partial_matches[field_name] += 1


    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'micro_f1': f1,
        'micro_precision': precision,
        'micro_recall': recall,
        'exact_matches': exact_matches,
        'total_predicted': total_pred,
        'total_ground_truth': total_true,
        'partial_matches': partial_matches
    }

def evaluate_multimodal_quintuple_extraction(result_file_path):



    with open(result_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    predictions = []
    ground_truths = []
    detailed_results = []
    extraction_errors = defaultdict(list)


    for item in results:
        item_id = item['id']
        model_output = item['model_output']


        if isinstance(item['ground_truth'], str):
            gt_quintuples, gt_valid, gt_error = parse_quintuple_list(item['ground_truth'])
        elif isinstance(item['ground_truth'], list):
            gt_quintuples = []
            for gt_item in item['ground_truth']:
                if isinstance(gt_item, (tuple, list)) and len(gt_item) >= 5:
                    gt_quintuples.append(tuple(str(element).strip() for element in gt_item[:5]))
            gt_valid = True
            gt_error = None
        else:
            gt_quintuples = []
            gt_valid = False
            gt_error = "invalid_ground_truth_format"


        pred_quintuples, pred_valid, pred_error = parse_quintuple_list(model_output)


        detailed_item = {
            'id': item_id,
            'model_output': model_output,
            'extracted_prediction': pred_quintuples,
            'ground_truth': gt_quintuples,
            'prediction_count': len(pred_quintuples),
            'ground_truth_count': len(gt_quintuples),
            'exact_matches': len(set(pred_quintuples) & set(gt_quintuples)) if pred_valid and gt_valid else 0,
            'valid': pred_valid and gt_valid
        }
        detailed_results.append(detailed_item)


        if not pred_valid:
            extraction_errors[pred_error].append(item_id)
        elif not gt_valid:
            extraction_errors[f"gt_{gt_error}"].append(item_id)


        if pred_valid and gt_valid:
            predictions.append(pred_quintuples)
            ground_truths.append(gt_quintuples)


    if len(predictions) == 0:
        return {
            'error': 'No valid predictions found',
            'total_samples': len(results),
            'extraction_errors': dict(extraction_errors)
        }


    metrics = calculate_quintuple_metrics(predictions, ground_truths)


    sample_with_predictions = sum(1 for item in detailed_results if item['prediction_count'] > 0)
    sample_with_correct_predictions = sum(1 for item in detailed_results if item['exact_matches'] > 0)


    all_pred_holders = []
    all_pred_targets = []
    all_pred_sentiments = []
    all_true_holders = []
    all_true_targets = []
    all_true_sentiments = []

    for pred_list, true_list in zip(predictions, ground_truths):
        for quintuple in pred_list:
            all_pred_holders.append(quintuple[0])
            all_pred_targets.append(quintuple[1])
            all_pred_sentiments.append(quintuple[4])

        for quintuple in true_list:
            all_true_holders.append(quintuple[0])
            all_true_targets.append(quintuple[1])
            all_true_sentiments.append(quintuple[4])


    evaluation_result = {
        'task_info': {
            'task_name': 'multimodal.quintuple.extraction',
            'dataset': 'PanoSent',
            'evaluation_time': datetime.now().isoformat(),
            'total_samples': len(results),
            'valid_samples': len(predictions),
            'extraction_success_rate': round(len(predictions) / len(results), 4)
        },
        'metrics': {
            'Micro_F1': round(metrics['micro_f1'], 4),
            'Micro_Precision': round(metrics['micro_precision'], 4),
            'Micro_Recall': round(metrics['micro_recall'], 4),
            'Exact_Matches': metrics['exact_matches'],
            'Total_Predicted': metrics['total_predicted'],
            'Total_Ground_Truth': metrics['total_ground_truth']
        },
        'sample_level_stats': {
            'samples_with_predictions': sample_with_predictions,
            'samples_with_correct_predictions': sample_with_correct_predictions,
            'avg_predictions_per_sample': round(metrics['total_predicted'] / len(predictions), 2) if len(predictions) > 0 else 0,
            'avg_ground_truth_per_sample': round(metrics['total_ground_truth'] / len(predictions), 2) if len(predictions) > 0 else 0
        },
        'partial_match_analysis': {
            field: {
                'matches': metrics['partial_matches'][field],
                'rate': round(metrics['partial_matches'][field] / max(metrics['total_predicted'], 1), 4)
            } for field in ['holder', 'target', 'aspect', 'opinion', 'sentiment']
        },
        'error_analysis': {
            'extraction_errors': {
                error_type: {
                    'count': len(sample_ids),
                    'sample_ids': sample_ids
                } for error_type, sample_ids in extraction_errors.items()
            }
        },
        'distribution': {
            'holders': {
                'ground_truth': dict(Counter(all_true_holders)),
                'predictions': dict(Counter(all_pred_holders))
            },
            'targets': {
                'ground_truth': dict(Counter(all_true_targets)),
                'predictions': dict(Counter(all_pred_targets))
            },
            'sentiments': {
                'ground_truth': dict(Counter(all_true_sentiments)),
                'predictions': dict(Counter(all_pred_sentiments))
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


    problem_samples = [item for item in detailed_results if item['exact_matches'] == 0 and item['ground_truth_count'] > 0]
    if problem_samples:
        problem_report_file = f"{base_name}_problem_samples.json"
        with open(problem_report_file, 'w', encoding='utf-8') as f:
            json.dump(problem_samples, f, ensure_ascii=False, indent=2)


    print(f"Evaluation complete: {len(results)} samples")
    print(f"Key metric: Micro F1={evaluation_result['metrics']['Micro_F1']}")
    print(f"Exact matches: {evaluation_result['metrics']['Exact_Matches']}/{evaluation_result['metrics']['Total_Ground_Truth']}")
    print(f"Average predictions per sample: {evaluation_result['sample_level_stats']['avg_predictions_per_sample']}")
    print(f"Extraction success rate: {evaluation_result['task_info']['extraction_success_rate']}")
    print(f"Results saved to: {eval_output_file}")
    if problem_samples:
        print(f"Problematic samples: {len(problem_samples)}; see {problem_report_file} for details")

    return evaluation_result


if __name__ == "__main__":
    result_file = "model_result.json"

    try:
        evaluation_result = evaluate_multimodal_quintuple_extraction(result_file)

    except FileNotFoundError:
        print(f"Error: file not found {result_file}")
    except json.JSONDecodeError:
        print(f"Error: invalid format for {result_file}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
