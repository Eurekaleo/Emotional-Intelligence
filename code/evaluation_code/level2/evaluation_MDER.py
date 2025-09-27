import json
import re
import ast
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

def extract_emotion_from_output(model_output):

    if not model_output:
        return None, False, "empty_output"


    valid_emotions = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']

    try:

        if "{'emotion':" in model_output or '{"emotion":' in model_output:

            cleaned_output = model_output.strip()


            json_match = re.search(r'\{[^}]*\}', cleaned_output)
            if json_match:
                cleaned_output = json_match.group()


            try:
                parsed = ast.literal_eval(cleaned_output)
            except:

                parsed = json.loads(cleaned_output)

            if 'emotion' in parsed and isinstance(parsed['emotion'], str):
                emotion = parsed['emotion'].lower().strip()
                if emotion in valid_emotions:
                    return emotion, True, None
                else:
                    return emotion, False, "invalid_emotion_label"


        cleaned_output = model_output.lower().strip()


        for emotion in valid_emotions:
            if cleaned_output == emotion:
                return emotion, True, None


        for emotion in valid_emotions:
            if emotion in cleaned_output:
                return emotion, False, "emotion_found_but_not_properly_formatted"

        return None, False, "no_valid_emotion_found"

    except Exception as e:
        return None, False, f"parsing_error_{str(e)}"

def evaluate_multiparty_dialogue_emotion_recognition(result_file_path):



    with open(result_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    predictions = []
    ground_truths = []
    detailed_results = []
    extraction_errors = defaultdict(list)
    prediction_errors = defaultdict(list)


    emotion_labels = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']


    for item in results:
        item_id = item['id']
        model_output = item['model_output']
        gt_emotion = item['ground_truth'].lower().strip()


        pred_emotion, is_valid, error_type = extract_emotion_from_output(model_output)


        detailed_item = {
            'id': item_id,
            'model_output': model_output,
            'extracted_prediction': pred_emotion,
            'ground_truth': gt_emotion,
            'correct': pred_emotion == gt_emotion if pred_emotion else False,
            'valid': is_valid
        }
        detailed_results.append(detailed_item)


        if not is_valid:
            extraction_errors[error_type].append(item_id)
        elif pred_emotion != gt_emotion:
            error_pattern = f"{gt_emotion}_to_{pred_emotion}"
            prediction_errors[error_pattern].append(item_id)


        if is_valid:
            predictions.append(pred_emotion)
            ground_truths.append(gt_emotion)


    if len(predictions) == 0:
        return {
            'error': 'No valid predictions found',
            'total_samples': len(results),
            'extraction_errors': dict(extraction_errors)
        }


    accuracy = accuracy_score(ground_truths, predictions)
    weighted_f1 = f1_score(ground_truths, predictions, average='weighted')
    macro_f1 = f1_score(ground_truths, predictions, average='macro')
    micro_f1 = f1_score(ground_truths, predictions, average='micro')


    cm = confusion_matrix(ground_truths, predictions, labels=emotion_labels)


    class_report = classification_report(ground_truths, predictions,
                                       target_names=emotion_labels,
                                       output_dict=True,
                                       zero_division=0)


    per_class_metrics = {}
    for i, label in enumerate(emotion_labels):
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


    evaluation_result = {
        'task_info': {
            'task_name': 'multiparty.dialogue.emotion.recognition',
            'dataset': 'MELD',
            'evaluation_time': datetime.now().isoformat(),
            'total_samples': len(results),
            'valid_predictions': len(predictions),
            'extraction_success_rate': round(len(predictions) / len(results), 4)
        },
        'metrics': {
            'ACC': round(accuracy, 4),
            'WAF': round(weighted_f1, 4),
            'Macro_F1': round(macro_f1, 4),
            'Micro_F1': round(micro_f1, 4)
        },
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': {
            'labels': emotion_labels,
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
        }
    }


    confusion_pairs = []
    for i, label1 in enumerate(emotion_labels):
        for j, label2 in enumerate(emotion_labels):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'true_emotion': label1,
                    'predicted_emotion': label2,
                    'count': int(cm[i, j]),
                    'percentage': round(cm[i, j] / np.sum(cm[i, :]) * 100, 2)
                })

    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
    evaluation_result['emotion_confusion_analysis'] = {
        'most_confused_pairs': confusion_pairs[:10]
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
    print(f"Extraction success rate: {evaluation_result['task_info']['extraction_success_rate']}")
    print(f"Results saved to: {eval_output_file}")
    if problem_samples:
        print(f"Problematic samples: {len(problem_samples)}; see {problem_report_file} for details")


    if confusion_pairs:
        print("\nMost confusable emotion pairs:")
        for pair in confusion_pairs[:5]:
            print(f"  {pair['true_emotion']} → {pair['predicted_emotion']}: {pair['count']} times ({pair['percentage']}%)")

    return evaluation_result


if __name__ == "__main__":
    result_file = "model_result.json"

    try:
        evaluation_result = evaluate_multiparty_dialogue_emotion_recognition(result_file)

    except FileNotFoundError:
        print(f"Error: file not found {result_file}")
    except json.JSONDecodeError:
        print(f"Error: invalid format for {result_file}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")

