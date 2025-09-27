import json
import re
import ast
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

def parse_sentiment_flip_output(output_text):

    if not output_text:
        return [], False, "empty_output"


    valid_sentiments = ['positive', 'negative', 'neutral']
    valid_trigger_types = [
        'Introduction of New Information',
        'Logical Argumentation',
        'Participant Feedback and Interaction',
        'Personal Experience and Self-reflection'
    ]

    try:

        cleaned_output = output_text.strip()


        list_match = re.search(r'\[.*\]', cleaned_output, re.DOTALL)
        if not list_match:
            return [], False, "no_list_structure"

        list_content = list_match.group()


        try:
            parsed = ast.literal_eval(list_content)
        except:

            list_content = re.sub(r"'", '"', list_content)
            parsed = json.loads(list_content)

        if not isinstance(parsed, list):
            return [], False, "not_list"


        validated_flips = []
        for item in parsed:
            if not isinstance(item, dict):
                continue


            required_fields = ['holder', 'initial_sentiment', 'flipped_sentiment', 'trigger_type']
            if not all(field in item for field in required_fields):
                continue

            holder = str(item['holder']).strip()
            initial_sentiment = str(item['initial_sentiment']).lower().strip()
            flipped_sentiment = str(item['flipped_sentiment']).lower().strip()
            trigger_type = str(item['trigger_type']).strip()


            if initial_sentiment not in valid_sentiments or flipped_sentiment not in valid_sentiments:
                continue


            trigger_valid = False
            for valid_trigger in valid_trigger_types:
                if trigger_type.lower() in valid_trigger.lower() or valid_trigger.lower() in trigger_type.lower():
                    trigger_type = valid_trigger
                    trigger_valid = True
                    break

            if not trigger_valid:
                continue


            if not holder:
                continue

            validated_flips.append({
                'holder': holder,
                'initial_sentiment': initial_sentiment,
                'flipped_sentiment': flipped_sentiment,
                'trigger_type': trigger_type
            })

        return validated_flips, len(validated_flips) >= 0, None

    except Exception as e:
        return [], False, f"parsing_error_{str(e)}"

def normalize_flip_for_comparison(flip):

    return {
        'holder': flip['holder'].lower().strip(),
        'initial_sentiment': flip['initial_sentiment'].lower().strip(),
        'flipped_sentiment': flip['flipped_sentiment'].lower().strip(),
        'trigger_type': flip['trigger_type'].strip()
    }

def calculate_exact_match_f1(predictions, ground_truths):

    total_pred_flips = 0
    total_true_flips = 0
    exact_matches = 0


    all_metrics = []

    for pred_flips, true_flips in zip(predictions, ground_truths):

        norm_pred_flips = [normalize_flip_for_comparison(flip) for flip in pred_flips]
        norm_true_flips = [normalize_flip_for_comparison(flip) for flip in true_flips]

        total_pred_flips += len(pred_flips)
        total_true_flips += len(true_flips)


        sample_matches = 0
        for true_flip in norm_true_flips:
            if true_flip in norm_pred_flips:
                sample_matches += 1
                exact_matches += 1


        sample_precision = sample_matches / len(pred_flips) if len(pred_flips) > 0 else 0
        sample_recall = sample_matches / len(true_flips) if len(true_flips) > 0 else (1 if len(pred_flips) == 0 else 0)
        sample_f1 = 2 * sample_precision * sample_recall / (sample_precision + sample_recall) if (sample_precision + sample_recall) > 0 else 0

        all_metrics.append({
            'precision': sample_precision,
            'recall': sample_recall,
            'f1': sample_f1,
            'matches': sample_matches,
            'pred_count': len(pred_flips),
            'true_count': len(true_flips)
        })


    micro_precision = exact_matches / total_pred_flips if total_pred_flips > 0 else 0
    micro_recall = exact_matches / total_true_flips if total_true_flips > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0


    macro_precision = np.mean([m['precision'] for m in all_metrics]) if all_metrics else 0
    macro_recall = np.mean([m['recall'] for m in all_metrics]) if all_metrics else 0
    macro_f1 = np.mean([m['f1'] for m in all_metrics]) if all_metrics else 0

    return {
        'exact_match_f1': micro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'total_predicted_flips': total_pred_flips,
        'total_ground_truth_flips': total_true_flips,
        'exact_matches': exact_matches,
        'sample_metrics': all_metrics
    }

def analyze_flip_patterns(predictions, ground_truths):

    all_pred_sentiments = []
    all_true_sentiments = []
    all_pred_triggers = []
    all_true_triggers = []
    all_pred_holders = []
    all_true_holders = []

    for pred_flips, true_flips in zip(predictions, ground_truths):
        for flip in pred_flips:
            all_pred_sentiments.append(f"{flip['initial_sentiment']}->{flip['flipped_sentiment']}")
            all_pred_triggers.append(flip['trigger_type'])
            all_pred_holders.append(flip['holder'])

        for flip in true_flips:
            all_true_sentiments.append(f"{flip['initial_sentiment']}->{flip['flipped_sentiment']}")
            all_true_triggers.append(flip['trigger_type'])
            all_true_holders.append(flip['holder'])

    return {
        'sentiment_flip_patterns': {
            'predicted': dict(Counter(all_pred_sentiments)),
            'ground_truth': dict(Counter(all_true_sentiments))
        },
        'trigger_type_distribution': {
            'predicted': dict(Counter(all_pred_triggers)),
            'ground_truth': dict(Counter(all_true_triggers))
        },
        'holder_distribution': {
            'predicted': dict(Counter(all_pred_holders)),
            'ground_truth': dict(Counter(all_true_holders))
        }
    }

def evaluate_sentiment_flip_analysis(result_file_path):



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


        pred_flips, pred_valid, pred_error = parse_sentiment_flip_output(model_output)


        if isinstance(ground_truth, str):
            gt_flips, gt_valid, gt_error = parse_sentiment_flip_output(ground_truth)
        elif isinstance(ground_truth, list):
            gt_flips = ground_truth
            gt_valid = True
            gt_error = None
        else:
            gt_flips = []
            gt_valid = False
            gt_error = "invalid_ground_truth_format"


        detailed_item = {
            'id': item_id,
            'model_output': model_output,
            'ground_truth': ground_truth,
            'extracted_prediction': pred_flips,
            'extracted_ground_truth': gt_flips,
            'prediction_valid': pred_valid,
            'ground_truth_valid': gt_valid,
            'predicted_flips_count': len(pred_flips),
            'ground_truth_flips_count': len(gt_flips)
        }
        detailed_results.append(detailed_item)


        if not pred_valid:
            parsing_errors[pred_error].append(item_id)
        elif not gt_valid:
            parsing_errors[f"gt_{gt_error}"].append(item_id)


        if pred_valid and gt_valid:
            predictions.append(pred_flips)
            ground_truths.append(gt_flips)


    if len(predictions) == 0:
        return {
            'error': 'No valid predictions found',
            'total_samples': len(results),
            'parsing_errors': dict(parsing_errors)
        }


    metrics = calculate_exact_match_f1(predictions, ground_truths)


    pattern_analysis = analyze_flip_patterns(predictions, ground_truths)


    pred_flip_counts = [len(flips) for flips in predictions]
    true_flip_counts = [len(flips) for flips in ground_truths]


    evaluation_result = {
        'task_info': {
            'task_name': 'sentiment.flip.analysis',
            'dataset': 'PanoSent',
            'task_type': 'GEN',
            'evaluation_time': datetime.now().isoformat(),
            'total_samples': len(results),
            'valid_samples': len(predictions),
            'parsing_success_rate': round(len(predictions) / len(results), 4)
        },
        'metrics': {
            'Exact_Match_F1': round(metrics['exact_match_f1'], 4),
            'Micro_Precision': round(metrics['micro_precision'], 4),
            'Micro_Recall': round(metrics['micro_recall'], 4),
            'Macro_Precision': round(metrics['macro_precision'], 4),
            'Macro_Recall': round(metrics['macro_recall'], 4),
            'Macro_F1': round(metrics['macro_f1'], 4),
            'Total_Predicted_Flips': metrics['total_predicted_flips'],
            'Total_Ground_Truth_Flips': metrics['total_ground_truth_flips'],
            'Exact_Matches': metrics['exact_matches']
        },
        'flip_count_analysis': {
            'avg_predicted_flips_per_sample': round(np.mean(pred_flip_counts), 2),
            'avg_ground_truth_flips_per_sample': round(np.mean(true_flip_counts), 2),
            'predicted_flip_count_distribution': dict(Counter(pred_flip_counts)),
            'ground_truth_flip_count_distribution': dict(Counter(true_flip_counts))
        },
        'pattern_analysis': pattern_analysis,
        'error_analysis': {
            'parsing_errors': {
                error_type: {
                    'count': len(sample_ids),
                    'sample_ids': sample_ids
                } for error_type, sample_ids in parsing_errors.items()
            }
        }
    }


    zero_pred_samples = sum(1 for count in pred_flip_counts if count == 0)
    zero_true_samples = sum(1 for count in true_flip_counts if count == 0)

    evaluation_result['zero_flip_analysis'] = {
        'predicted_zero_flips': zero_pred_samples,
        'ground_truth_zero_flips': zero_true_samples,
        'zero_flip_accuracy': sum(1 for p, t in zip(pred_flip_counts, true_flip_counts) if p == 0 and t == 0) / len(predictions)
    }


    base_name = result_file_path.replace('.json', '')


    eval_output_file = f"{base_name}_evaluation.json"
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)


    detailed_output_file = f"{base_name}_detailed_results.json"
    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)


    low_score_samples = []
    for i, sample_metric in enumerate(metrics['sample_metrics']):
        if sample_metric['f1'] < 0.5:
            low_score_samples.append({
                'id': detailed_results[i]['id'],
                'f1_score': sample_metric['f1'],
                'precision': sample_metric['precision'],
                'recall': sample_metric['recall'],
                'predicted_flips': detailed_results[i]['extracted_prediction'],
                'ground_truth_flips': detailed_results[i]['extracted_ground_truth']
            })

    if low_score_samples:
        low_score_report_file = f"{base_name}_low_score_samples.json"
        with open(low_score_report_file, 'w', encoding='utf-8') as f:
            json.dump(low_score_samples, f, ensure_ascii=False, indent=2)


    parsing_error_samples = [item for item in detailed_results if not item['prediction_valid']]
    if parsing_error_samples:
        parsing_error_report_file = f"{base_name}_parsing_error_samples.json"
        with open(parsing_error_report_file, 'w', encoding='utf-8') as f:
            json.dump(parsing_error_samples, f, ensure_ascii=False, indent=2)


    print(f"Evaluation complete: {len(results)} samples")
    print(f"Key metric: Exact Match F1={evaluation_result['metrics']['Exact_Match_F1']}")
    print(f"Micro-averaged metrics: Precision={evaluation_result['metrics']['Micro_Precision']}, Recall={evaluation_result['metrics']['Micro_Recall']}")
    print(f"Macro-averaged metrics: Precision={evaluation_result['metrics']['Macro_Precision']}, Recall={evaluation_result['metrics']['Macro_Recall']}, F1={evaluation_result['metrics']['Macro_F1']}")
    print(f"Parsing success rate: {evaluation_result['task_info']['parsing_success_rate']}")


    print(f"\nFlip statistics:")
    print(f"Average predicted flips per sample: {evaluation_result['flip_count_analysis']['avg_predicted_flips_per_sample']}")
    print(f"Average ground-truth flips per sample: {evaluation_result['flip_count_analysis']['avg_ground_truth_flips_per_sample']}")
    print(f"Zero-flip accuracy: {evaluation_result['zero_flip_analysis']['zero_flip_accuracy']:.4f}")


    print(f"\nSentiment flip patterns:")
    gt_patterns = evaluation_result['pattern_analysis']['sentiment_flip_patterns']['ground_truth']
    pred_patterns = evaluation_result['pattern_analysis']['sentiment_flip_patterns']['predicted']
    for pattern in sorted(set(list(gt_patterns.keys()) + list(pred_patterns.keys()))):
        gt_count = gt_patterns.get(pattern, 0)
        pred_count = pred_patterns.get(pattern, 0)
        print(f"  {pattern}: GT={gt_count}, Pred={pred_count}")


    print(f"\nTrigger type distribution:")
    gt_triggers = evaluation_result['pattern_analysis']['trigger_type_distribution']['ground_truth']
    pred_triggers = evaluation_result['pattern_analysis']['trigger_type_distribution']['predicted']
    for trigger in sorted(set(list(gt_triggers.keys()) + list(pred_triggers.keys()))):
        gt_count = gt_triggers.get(trigger, 0)
        pred_count = pred_triggers.get(trigger, 0)
        print(f"  {trigger}: GT={gt_count}, Pred={pred_count}")

    print(f"\nResults saved to: {eval_output_file}")
    if low_score_samples:
        print(f"Low-scoring samples: {len(low_score_samples)}; see {low_score_report_file} for details")
    if parsing_error_samples:
        print(f"Parsing-error samples: {len(parsing_error_samples)}; see {parsing_error_report_file} for details")

    return evaluation_result


if __name__ == "__main__":
    result_file = "model_result.json"

    try:
        evaluation_result = evaluate_sentiment_flip_analysis(result_file)

    except FileNotFoundError:
        print(f"Error: file not found {result_file}")
    except json.JSONDecodeError:
        print(f"Error: invalid format for {result_file}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
