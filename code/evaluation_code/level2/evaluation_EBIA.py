import json
import re
import ast
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

def extract_emotion_intent_from_output(model_output):

    if not model_output:
        return None, None, False, "empty_output"


    valid_emotions = ['happy', 'surprise', 'sad', 'disgust', 'anger', 'fear', 'neutral']
    valid_intents = ['questioning', 'agreeing', 'acknowledging', 'encouraging',
                    'consoling', 'suggesting', 'wishing', 'neutral']

    try:

        if "{'emotion':" in model_output or '{"emotion":' in model_output:

            cleaned_output = model_output.strip()


            cleaned_output = re.sub(r"'intent'(\s*,\s*)'", r"'intent':\1'", cleaned_output)
            cleaned_output = re.sub(r'"intent"(\s*,\s*)"', r'"intent":\1"', cleaned_output)

            if not cleaned_output.startswith('{'):

                json_match = re.search(r'\{[^}]*\}', cleaned_output)
                if json_match:
                    cleaned_output = json_match.group()


            try:
                parsed = ast.literal_eval(cleaned_output)
            except:

                parsed = json.loads(cleaned_output)

            if 'emotion' in parsed and 'intent' in parsed:
                emotion = parsed['emotion'].lower().strip()
                intent = parsed['intent'].lower().strip()

                emotion_valid = emotion in valid_emotions
                intent_valid = intent in valid_intents

                if emotion_valid and intent_valid:
                    return emotion, intent, True, None
                elif not emotion_valid and not intent_valid:
                    return emotion, intent, False, "invalid_emotion_and_intent"
                elif not emotion_valid:
                    return emotion, intent, False, "invalid_emotion"
                else:
                    return emotion, intent, False, "invalid_intent"


        emotion_found = None
        intent_found = None

        for emotion in valid_emotions:
            if emotion in model_output.lower():
                emotion_found = emotion
                break

        for intent in valid_intents:
            if intent in model_output.lower():
                intent_found = intent
                break

        if emotion_found and intent_found:
            return emotion_found, intent_found, False, "labels_found_but_not_properly_formatted"
        elif emotion_found:
            return emotion_found, None, False, "only_emotion_found"
        elif intent_found:
            return None, intent_found, False, "only_intent_found"

        return None, None, False, "no_labels_pattern"

    except Exception as e:
        return None, None, False, f"parsing_error_{str(e)}"

def parse_ground_truth(ground_truth):

    if isinstance(ground_truth, dict):
        return ground_truth.get('emotion', '').lower(), ground_truth.get('intent', '').lower()
    elif isinstance(ground_truth, str):

        cleaned_gt = ground_truth.strip()
        cleaned_gt = re.sub(r"'intent'(\s*,\s*)'", r"'intent':\1'", cleaned_gt)

        try:
            parsed = ast.literal_eval(cleaned_gt)
            return parsed.get('emotion', '').lower(), parsed.get('intent', '').lower()
        except:
            return '', ''
    else:
        return '', ''

def evaluate_emotion_based_intent_analysis(result_file_path):



    with open(result_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    emotion_predictions = []
    emotion_ground_truths = []
    intent_predictions = []
    intent_ground_truths = []
    detailed_results = []
    extraction_errors = defaultdict(list)
    prediction_errors = defaultdict(list)


    emotion_labels = ['happy', 'surprise', 'sad', 'disgust', 'anger', 'fear', 'neutral']
    intent_labels = ['questioning', 'agreeing', 'acknowledging', 'encouraging',
                    'consoling', 'suggesting', 'wishing', 'neutral']


    for item in results:
        item_id = item['id']
        model_output = item['model_output']
        gt_emotion, gt_intent = parse_ground_truth(item['ground_truth'])


        pred_emotion, pred_intent, is_valid, error_type = extract_emotion_intent_from_output(model_output)


        detailed_item = {
            'id': item_id,
            'model_output': model_output,
            'extracted_prediction': {
                'emotion': pred_emotion,
                'intent': pred_intent
            },
            'ground_truth': {
                'emotion': gt_emotion,
                'intent': gt_intent
            },
            'emotion_correct': pred_emotion == gt_emotion if pred_emotion else False,
            'intent_correct': pred_intent == gt_intent if pred_intent else False,
            'both_correct': (pred_emotion == gt_emotion and pred_intent == gt_intent) if (pred_emotion and pred_intent) else False,
            'valid': is_valid
        }
        detailed_results.append(detailed_item)


        if not is_valid:
            extraction_errors[error_type].append(item_id)
        else:
            if pred_emotion != gt_emotion:
                error_pattern = f"emotion_{gt_emotion}_to_{pred_emotion}"
                prediction_errors[error_pattern].append(item_id)
            if pred_intent != gt_intent:
                error_pattern = f"intent_{gt_intent}_to_{pred_intent}"
                prediction_errors[error_pattern].append(item_id)


        if is_valid:
            emotion_predictions.append(pred_emotion)
            emotion_ground_truths.append(gt_emotion)
            intent_predictions.append(pred_intent)
            intent_ground_truths.append(gt_intent)


    if len(emotion_predictions) == 0:
        return {
            'error': 'No valid predictions found',
            'total_samples': len(results),
            'extraction_errors': dict(extraction_errors)
        }


    emotion_accuracy = accuracy_score(emotion_ground_truths, emotion_predictions)
    emotion_weighted_f1 = f1_score(emotion_ground_truths, emotion_predictions, average='weighted')
    emotion_macro_f1 = f1_score(emotion_ground_truths, emotion_predictions, average='macro')


    intent_accuracy = accuracy_score(intent_ground_truths, intent_predictions)
    intent_weighted_f1 = f1_score(intent_ground_truths, intent_predictions, average='weighted')
    intent_macro_f1 = f1_score(intent_ground_truths, intent_predictions, average='macro')


    both_correct = sum(1 for item in detailed_results if item['both_correct'])
    joint_accuracy = both_correct / len([item for item in detailed_results if item['valid']])


    emotion_cm = confusion_matrix(emotion_ground_truths, emotion_predictions, labels=emotion_labels)
    intent_cm = confusion_matrix(intent_ground_truths, intent_predictions, labels=intent_labels)


    emotion_class_report = classification_report(emotion_ground_truths, emotion_predictions,
                                               target_names=emotion_labels,
                                               output_dict=True, zero_division=0)
    intent_class_report = classification_report(intent_ground_truths, intent_predictions,
                                              target_names=intent_labels,
                                              output_dict=True, zero_division=0)


    evaluation_result = {
        'task_info': {
            'task_name': 'emotion.based.intent.analysis',
            'dataset': 'MC-EIU',
            'evaluation_time': datetime.now().isoformat(),
            'total_samples': len(results),
            'valid_predictions': len(emotion_predictions),
            'extraction_success_rate': round(len(emotion_predictions) / len(results), 4)
        },
        'metrics': {
            'emotion_metrics': {
                'ACC': round(emotion_accuracy, 4),
                'WAF': round(emotion_weighted_f1, 4),
                'Macro_F1': round(emotion_macro_f1, 4)
            },
            'intent_metrics': {
                'ACC': round(intent_accuracy, 4),
                'WAF': round(intent_weighted_f1, 4),
                'Macro_F1': round(intent_macro_f1, 4)
            },
            'joint_metrics': {
                'Joint_ACC': round(joint_accuracy, 4)
            }
        },
        'per_class_metrics': {
            'emotion': {
                label: {
                    'precision': round(emotion_class_report[label]['precision'], 4),
                    'recall': round(emotion_class_report[label]['recall'], 4),
                    'f1_score': round(emotion_class_report[label]['f1-score'], 4),
                    'support': int(emotion_class_report[label]['support'])
                } for label in emotion_labels if label in emotion_class_report
            },
            'intent': {
                label: {
                    'precision': round(intent_class_report[label]['precision'], 4),
                    'recall': round(intent_class_report[label]['recall'], 4),
                    'f1_score': round(intent_class_report[label]['f1-score'], 4),
                    'support': int(intent_class_report[label]['support'])
                } for label in intent_labels if label in intent_class_report
            }
        },
        'confusion_matrices': {
            'emotion': {
                'labels': emotion_labels,
                'matrix': emotion_cm.tolist()
            },
            'intent': {
                'labels': intent_labels,
                'matrix': intent_cm.tolist()
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
            'emotion': {
                'ground_truth': dict(Counter(emotion_ground_truths)),
                'predictions': dict(Counter(emotion_predictions))
            },
            'intent': {
                'ground_truth': dict(Counter(intent_ground_truths)),
                'predictions': dict(Counter(intent_predictions))
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


    problem_samples = [item for item in detailed_results if not item['both_correct']]
    if problem_samples:
        problem_report_file = f"{base_name}_problem_samples.json"
        with open(problem_report_file, 'w', encoding='utf-8') as f:
            json.dump(problem_samples, f, ensure_ascii=False, indent=2)


    print(f"Evaluation complete: {len(results)} samples")
    print(f"Emotion metrics: ACC={evaluation_result['metrics']['emotion_metrics']['ACC']}, WAF={evaluation_result['metrics']['emotion_metrics']['WAF']}")
    print(f"Intent metrics: ACC={evaluation_result['metrics']['intent_metrics']['ACC']}, WAF={evaluation_result['metrics']['intent_metrics']['WAF']}")
    print(f"Joint accuracy: {evaluation_result['metrics']['joint_metrics']['Joint_ACC']}")
    print(f"Extraction success rate: {evaluation_result['task_info']['extraction_success_rate']}")
    print(f"Results saved to: {eval_output_file}")
    if problem_samples:
        print(f"Problematic samples: {len(problem_samples)}; see {problem_report_file} for details")

    return evaluation_result


if __name__ == "__main__":
    result_file = "model_result.json"

    try:
        evaluation_result = evaluate_emotion_based_intent_analysis(result_file)

    except FileNotFoundError:
        print(f"Error: file not found {result_file}")
    except json.JSONDecodeError:
        print(f"Error: invalid format for {result_file}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")

