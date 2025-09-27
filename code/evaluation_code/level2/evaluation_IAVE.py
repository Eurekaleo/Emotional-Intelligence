import json
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np


ONTOLOGY = {
    "Clothing": {
        "Sleeve Style": ["3/4 Sleeve", "Long Sleeve", "Short Sleeve", "Sleeveless", "Strappy"],
        "Neckline": ["Button Down", "Cowl Neck", "Crew Neck", "Halter", "Henley", "Polo", "Scoop Neck", "Square Neck", "Strapless", "Turtleneck", "V-Neck"],
        "Length": ["Capri", "Long Dress/Gown", "Midi", "Mini/Short"],
        "Shoulder Style": ["Cold Shoulder", "Off Shoulder", "One Shoulder"],
    },
    "Footwear": {
        "Shaft Height": ["Ankle Boot", "Bootie", "Knee High", "Mid Calf", "Over The Knee"],
        "Athletic Shoe Style": ["Basketball", "Climbing Shoe", "Cycling", "Golf", "Hiking Boot", "Running Shoe", "Skateboarding Shoe", "Soccer", "Tennis", "Training Shoe", "Volleyball", "Walking"],
        "Boot Style": ["Chelsea", "Combat", "Western/Cowboy", "Motorcycle", "Rain Boots", "Snow Boots"],
        "Heel Height": ["Flat", "High Heel", "Low Heel", "Mid Heel"],
        "Toe Style": ["Pointed Toe", "Round Toe"],
    }
}

def extract_attribute_type_from_prompt(prompt):


    attribute_patterns = [
        r"What is ([^?]+) of this product",
        r"What ([^?]+) does this product have",
        r"Identify the ([^?]+) of this product"
    ]

    for pattern in attribute_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            attribute_type = match.group(1).strip()


            for category in ONTOLOGY.values():
                if attribute_type in category:
                    return attribute_type, category[attribute_type]


            for category in ONTOLOGY.values():
                for attr_name, values in category.items():
                    if attr_name.lower() in attribute_type.lower() or attribute_type.lower() in attr_name.lower():
                        return attr_name, values


    choices_match = re.search(r"Answer with the option from the given choices directly:([^.]+)", prompt)
    if choices_match:
        choices_text = choices_match.group(1).strip()
        choices = [choice.strip() for choice in choices_text.split(',')]


        for category in ONTOLOGY.values():
            for attr_name, values in category.items():
                if set(choices) == set(values) or len(set(choices) & set(values)) > len(choices) * 0.8:
                    return attr_name, values

    return "Unknown", []

def extract_value_from_output(model_output, valid_values):
 
    if not model_output:
        return None, False, "empty_output"


    cleaned_output = model_output.strip()


    for value in valid_values:
        if cleaned_output == value:
            return value, True, None


    for value in valid_values:
        if cleaned_output.lower() == value.lower():
            return value, True, None


    for value in valid_values:
        if value in cleaned_output or cleaned_output in value:
            return value, True, None


    cleaned_normalized = re.sub(r'[^\w\s]', '', cleaned_output.lower())
    for value in valid_values:
        value_normalized = re.sub(r'[^\w\s]', '', value.lower())
        if cleaned_normalized == value_normalized:
            return value, True, None


    for value in valid_values:
        value_words = value.lower().split()
        if all(word in cleaned_output.lower() for word in value_words):
            return value, False, "keyword_match"


    if any(word in cleaned_output.lower() for value in valid_values for word in value.lower().split()):
        return None, False, "partial_match_failed"
    else:
        return None, False, "no_valid_value_found"

def evaluate_implicit_attribute_value_extraction(result_file_path):



    with open(result_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)


    attribute_results = defaultdict(lambda: {
        'predictions': [],
        'ground_truths': [],
        'detailed_results': [],
        'extraction_errors': defaultdict(list),
        'prediction_errors': defaultdict(list)
    })

    overall_predictions = []
    overall_ground_truths = []
    overall_detailed_results = []
    overall_extraction_errors = defaultdict(list)


    for item in results:
        item_id = item['id']
        prompt = item['prompt']
        model_output = item['model_output']
        gt_value = item['ground_truth'].strip()


        attribute_type, valid_values = extract_attribute_type_from_prompt(prompt)


        pred_value, is_valid, error_type = extract_value_from_output(model_output, valid_values)


        detailed_item = {
            'id': item_id,
            'attribute_type': attribute_type,
            'valid_values': valid_values,
            'model_output': model_output,
            'extracted_prediction': pred_value,
            'ground_truth': gt_value,
            'correct': pred_value == gt_value if pred_value else False,
            'valid': is_valid
        }


        attribute_results[attribute_type]['detailed_results'].append(detailed_item)
        overall_detailed_results.append(detailed_item)


        if not is_valid:
            attribute_results[attribute_type]['extraction_errors'][error_type].append(item_id)
            overall_extraction_errors[error_type].append(item_id)
        elif pred_value != gt_value:
            error_pattern = f"{gt_value}_to_{pred_value}"
            attribute_results[attribute_type]['prediction_errors'][error_pattern].append(item_id)


        if pred_value:
            attribute_results[attribute_type]['predictions'].append(pred_value)
            attribute_results[attribute_type]['ground_truths'].append(gt_value)
            overall_predictions.append(pred_value)
            overall_ground_truths.append(gt_value)


    if len(overall_predictions) == 0:
        return {
            'error': 'No valid predictions found',
            'total_samples': len(results),
            'extraction_errors': dict(overall_extraction_errors)
        }


    overall_accuracy = accuracy_score(overall_ground_truths, overall_predictions)
    overall_weighted_f1 = f1_score(overall_ground_truths, overall_predictions, average='weighted')
    overall_macro_f1 = f1_score(overall_ground_truths, overall_predictions, average='macro')


    attribute_metrics = {}
    for attr_type, attr_data in attribute_results.items():
        if len(attr_data['predictions']) > 0:
            attr_accuracy = accuracy_score(attr_data['ground_truths'], attr_data['predictions'])
            attr_weighted_f1 = f1_score(attr_data['ground_truths'], attr_data['predictions'], average='weighted')


            all_values = list(set(attr_data['ground_truths'] + attr_data['predictions']))


            attr_cm = confusion_matrix(attr_data['ground_truths'], attr_data['predictions'], labels=all_values)


            attr_class_report = classification_report(attr_data['ground_truths'], attr_data['predictions'],
                                                   target_names=all_values,
                                                   output_dict=True, zero_division=0)

            attribute_metrics[attr_type] = {
                'ACC': round(attr_accuracy, 4),
                'WAF': round(attr_weighted_f1, 4),
                'total_samples': len(attr_data['detailed_results']),
                'valid_predictions': len(attr_data['predictions']),
                'per_class_metrics': {
                    label: {
                        'precision': round(attr_class_report[label]['precision'], 4),
                        'recall': round(attr_class_report[label]['recall'], 4),
                        'f1_score': round(attr_class_report[label]['f1-score'], 4),
                        'support': int(attr_class_report[label]['support'])
                    } for label in all_values if label in attr_class_report
                },
                'confusion_matrix': {
                    'labels': all_values,
                    'matrix': attr_cm.tolist()
                },
                'distribution': {
                    'ground_truth': dict(Counter(attr_data['ground_truths'])),
                    'predictions': dict(Counter(attr_data['predictions']))
                }
            }


    evaluation_result = {
        'task_info': {
            'task_name': 'implicit.attribute.value.extraction',
            'dataset': 'ImplicitAVE',
            'evaluation_time': datetime.now().isoformat(),
            'total_samples': len(results),
            'valid_predictions': len(overall_predictions),
            'extraction_success_rate': round(len(overall_predictions) / len(results), 4),
            'attribute_types_count': len(attribute_results)
        },
        'overall_metrics': {
            'ACC': round(overall_accuracy, 4),
            'WAF': round(overall_weighted_f1, 4),
            'Macro_F1': round(overall_macro_f1, 4)
        },
        'attribute_metrics': attribute_metrics,
        'error_analysis': {
            'overall_extraction_errors': {
                error_type: {
                    'count': len(sample_ids),
                    'sample_ids': sample_ids
                } for error_type, sample_ids in overall_extraction_errors.items()
            }
        },
        'overall_distribution': {
            'ground_truth': dict(Counter(overall_ground_truths)),
            'predictions': dict(Counter(overall_predictions))
        }
    }


    base_name = result_file_path.replace('.json', '')


    eval_output_file = f"{base_name}_evaluation.json"
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)


    detailed_output_file = f"{base_name}_detailed_results.json"
    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(overall_detailed_results, f, ensure_ascii=False, indent=2)


    attribute_detailed_file = f"{base_name}_attribute_detailed_results.json"
    with open(attribute_detailed_file, 'w', encoding='utf-8') as f:
        formatted_attr_results = {}
        for attr_type, attr_data in attribute_results.items():
            formatted_attr_results[attr_type] = {
                'detailed_results': attr_data['detailed_results'],
                'extraction_errors': dict(attr_data['extraction_errors']),
                'prediction_errors': dict(attr_data['prediction_errors'])
            }
        json.dump(formatted_attr_results, f, ensure_ascii=False, indent=2)


    problem_samples = [item for item in overall_detailed_results if not item['correct']]
    if problem_samples:
        problem_report_file = f"{base_name}_problem_samples.json"
        with open(problem_report_file, 'w', encoding='utf-8') as f:
            json.dump(problem_samples, f, ensure_ascii=False, indent=2)


    print(f"Evaluation complete: {len(results)} samples")
    print(f"Overall metrics: ACC={evaluation_result['overall_metrics']['ACC']}, WAF={evaluation_result['overall_metrics']['WAF']}")
    print(f"Number of attribute types: {evaluation_result['task_info']['attribute_types_count']}")
    print(f"Extraction success rate: {evaluation_result['task_info']['extraction_success_rate']}")
    print(f"Results saved to: {eval_output_file}")
    if problem_samples:
        print(f"Problematic samples: {len(problem_samples)}; see {problem_report_file} for details")


    print("\nPerformance by attribute type:")
    for attr_type, metrics in attribute_metrics.items():
        print(f"  {attr_type}: ACC={metrics['ACC']}, WAF={metrics['WAF']}, Samples={metrics['total_samples']}")

    return evaluation_result


if __name__ == "__main__":
    result_file = "model_result.json"

    try:
        evaluation_result = evaluate_implicit_attribute_value_extraction(result_file)

    except FileNotFoundError:
        print(f"Error: file not found {result_file}")
    except json.JSONDecodeError:
        print(f"Error: invalid format for {result_file}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
