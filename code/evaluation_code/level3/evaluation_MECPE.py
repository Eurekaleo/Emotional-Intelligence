import json
import re
import ast
from collections import Counter, defaultdict
from datetime import datetime


from sklearn.metrics import f1_score, precision_score, recall_score

def parse_emotion_cause_pairs(output_text):

    if not output_text:
        return {}, False, "empty_output"

    valid_emotions = ['joy', 'sadness', 'anger', 'disgust', 'fear', 'surprise', 'neutral']

    try:
        cleaned_output = output_text.strip()


        json_match = re.search(r'\{.*\}', cleaned_output, re.DOTALL)
        if not json_match:
            return {}, False, "no_json_structure"

        json_content = json_match.group()

        parsed = None


        try:

            parsed = ast.literal_eval(json_content)
        except (ValueError, SyntaxError) as e:


            if "EOF" in str(e) or "end of string" in str(e):


                last_comma_index = json_content.rfind(',')
                if last_comma_index != -1:

                    content_after_comma = json_content[last_comma_index+1:].strip()
                    if content_after_comma and content_after_comma != '}':
                        truncated_content = json_content[:last_comma_index] + '}'
                        try:

                            parsed = ast.literal_eval(truncated_content)

                        except (ValueError, SyntaxError):

                            pass


            if parsed is None:
                try:

                    json_content_fixed = json_content.replace("'", '"')
                    parsed = json.loads(json_content_fixed)
                except json.JSONDecodeError:

                    raise e


        if not isinstance(parsed, dict):
            return {}, False, "not_dictionary"

        validated_pairs = {}
        for utterance_id, pair_data in parsed.items():

            if not isinstance(pair_data, dict) or 'emotion' not in pair_data or 'cause_utterance_id' not in pair_data:
                continue

            emotion = str(pair_data['emotion']).lower().strip()
            cause_id = str(pair_data['cause_utterance_id']).strip()


            if emotion not in valid_emotions:
                continue


            try:
                int(utterance_id)
                int(cause_id)
            except (ValueError, TypeError):
                continue

            validated_pairs[str(utterance_id)] = {
                'emotion': emotion,
                'cause_utterance_id': cause_id
            }

        return validated_pairs, len(validated_pairs) > 0, None

    except Exception as e:

        return {}, False, f"parsing_error_{type(e).__name__}"

def calculate_pair_extraction_metrics(predictions, ground_truths):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    emotion_correct_on_common = 0
    cause_correct_on_common = 0
    total_common_utterances = 0

    all_true_emotions = []
    all_pred_emotions = []

    for pred_dict, true_dict in zip(predictions, ground_truths):

        pred_pairs = {(str(uid), data['emotion'], str(data['cause_utterance_id'])) for uid, data in pred_dict.items()}
        true_pairs = {(str(uid), data['emotion'], str(data['cause_utterance_id'])) for uid, data in true_dict.items()}


        all_pred_emotions.extend([data['emotion'] for data in pred_dict.values()])
        all_true_emotions.extend([data['emotion'] for data in true_dict.values()])


        total_tp += len(pred_pairs.intersection(true_pairs))
        total_fp += len(pred_pairs.difference(true_pairs))
        total_fn += len(true_pairs.difference(pred_pairs))


        common_ids = set(pred_dict.keys()) & set(true_dict.keys())
        total_common_utterances += len(common_ids)
        for uid in common_ids:
            if pred_dict[uid]['emotion'] == true_dict[uid]['emotion']:
                emotion_correct_on_common += 1
            if pred_dict[uid]['cause_utterance_id'] == true_dict[uid]['cause_utterance_id']:
                cause_correct_on_common += 1


    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    emotion_accuracy_common = emotion_correct_on_common / total_common_utterances if total_common_utterances > 0 else 0
    cause_accuracy_common = cause_correct_on_common / total_common_utterances if total_common_utterances > 0 else 0

    return {
        'micro_f1': micro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'emotion_accuracy_on_common_ids': emotion_accuracy_common,
        'cause_accuracy_on_common_ids': cause_accuracy_common,
        'total_predicted_pairs': total_tp + total_fp,
        'total_ground_truth_pairs': total_tp + total_fn,
        'exact_matches': total_tp,
        'emotion_distribution': dict(Counter(all_true_emotions)),
        'predicted_emotion_distribution': dict(Counter(all_pred_emotions))
    }

def evaluate_multimodal_emotion_cause_pair_extraction(result_file_path):


    with open(result_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    predictions = []
    ground_truths = []
    detailed_results = []
    extraction_errors = defaultdict(list)

    for item in results:
        item_id = item.get('id', 'unknown')
        model_output = item.get('model_output', '')
        ground_truth = item.get('ground_truth', {})


        pred_pairs, pred_valid, pred_error = parse_emotion_cause_pairs(model_output)


        if isinstance(ground_truth, str):
            gt_pairs, gt_valid, gt_error = parse_emotion_cause_pairs(ground_truth)
        else:
            gt_pairs, gt_valid, gt_error = parse_emotion_cause_pairs(json.dumps(ground_truth))

        detailed_item = {
            'id': item_id,
            'model_output': model_output,
            'ground_truth': ground_truth,
            'extracted_prediction': pred_pairs,
            'extracted_ground_truth': gt_pairs,
            'prediction_valid': pred_valid,
            'ground_truth_valid': gt_valid,
            'predicted_pairs_count': len(pred_pairs),
            'ground_truth_pairs_count': len(gt_pairs)
        }
        detailed_results.append(detailed_item)

        if not pred_valid:
            extraction_errors[pred_error].append(item_id)
        if not gt_valid:

            extraction_errors[f"gt_{gt_error}"].append(item_id)


        if pred_valid and gt_valid:
            predictions.append(pred_pairs)
            ground_truths.append(gt_pairs)

    if not predictions:
        print("Error: No valid prediction–ground truth pairs were found for evaluation.")
        print(f"Total samples: {len(results)}")
        print("Parsing error stats:", json.dumps(dict(extraction_errors), indent=2))
        return {
            'error': 'No valid prediction and ground_truth pairs found for evaluation',
            'total_samples': len(results),
            'extraction_errors': dict(extraction_errors)
    }


    metrics = calculate_pair_extraction_metrics(predictions, ground_truths)

    evaluation_result = {
        'task_info': {
            'task_name': 'multimodal.emotion.cause.pair.extraction',
            'dataset': 'ECF',
            'evaluation_time': datetime.now().isoformat(),
            'total_samples': len(results),
            'valid_samples_for_eval': len(predictions),
            'extraction_success_rate': round(len(predictions) / len(results) if results else 0, 4)
        },
        'metrics': {
            'Micro_F1': round(metrics['micro_f1'], 4),
            'Micro_Precision': round(metrics['micro_precision'], 4),
            'Micro_Recall': round(metrics['micro_recall'], 4),
            'Emotion_ACC_on_Common_IDs': round(metrics['emotion_accuracy_on_common_ids'], 4),
            'Cause_ACC_on_Common_IDs': round(metrics['cause_accuracy_on_common_ids'], 4),
            'Total_Predicted_Pairs': metrics['total_predicted_pairs'],
            'Total_Ground_Truth_Pairs': metrics['total_ground_truth_pairs'],
            'Exact_Matches': metrics['exact_matches']
        },
        'distribution_analysis': {
            'emotion_distribution': metrics['emotion_distribution'],
            'predicted_emotion_distribution': metrics['predicted_emotion_distribution']
        },
        'error_analysis': {
            'extraction_errors': {
                error_type: {
                    'count': len(sample_ids),
                    'sample_ids': sample_ids[:10]
                } for error_type, sample_ids in extraction_errors.items()
            }
        }
    }


    base_name = result_file_path.rsplit('.', 1)[0]
    eval_output_file = f"{base_name}_evaluation.json"
    detailed_output_file = f"{base_name}_detailed_results.json"

    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)

    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)


    print("\n" + "="*50)
    print("Evaluation Report")
    print("="*50)
    print(f"Evaluation complete: {len(results)} total samples")
    print(f"Valid evaluation samples (both prediction and label parsed successfully): {len(predictions)} / {len(results)} (extraction success rate: {evaluation_result['task_info']['extraction_success_rate']:.2%})")
    print("\n--- Primary Metrics ---")
    print(f"Micro F1: {evaluation_result['metrics']['Micro_F1']:.4f}")
    print(f"Micro Precision: {evaluation_result['metrics']['Micro_Precision']:.4f}")
    print(f"Micro Recall: {evaluation_result['metrics']['Micro_Recall']:.4f}")
    print("\n--- Auxiliary Metrics ---")
    print(f"Emotion accuracy on common IDs (Emotion ACC): {evaluation_result['metrics']['Emotion_ACC_on_Common_IDs']:.4f}")
    print(f"Cause accuracy on common IDs (Cause ACC): {evaluation_result['metrics']['Cause_ACC_on_Common_IDs']:.4f}")
    print(f"Total predicted pairs: {evaluation_result['metrics']['Total_Predicted_Pairs']}")
    print(f"Total ground-truth pairs: {evaluation_result['metrics']['Total_Ground_Truth_Pairs']}")
    print(f"Exact matches: {evaluation_result['metrics']['Exact_Matches']}")

    if extraction_errors:
        print("\n--- Parsing Error Analysis ---")
        for error_type, info in evaluation_result['error_analysis']['extraction_errors'].items():
            print(f"- Type: {error_type}, Count: {info['count']}")

    print(f"Detailed evaluation results saved to: {eval_output_file}")
    print(f"Per-sample parsing details saved to: {detailed_output_file}")


    return evaluation_result




if __name__ == "__main__":

    result_file = "model_result.json"

    try:
        evaluation_result = evaluate_multimodal_emotion_cause_pair_extraction(result_file)

    except FileNotFoundError:
        print(f"Error: File '{result_file}' not found. Please check that the path is correct.")
    except json.JSONDecodeError:
        print(f"Error: File '{result_file}' is not valid JSON. Ensure the file is a JSON list (starts with '[' and ends with ']').")
    except Exception as e:
        print(f"An unknown error occurred during evaluation: {e}")
