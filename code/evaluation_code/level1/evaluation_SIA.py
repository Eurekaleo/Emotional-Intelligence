import json
import re
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def extract_emotion_from_output(model_output):

    if not model_output:
        return None, False, "empty_output"

                 
    valid_emotions = [
        "strong negative",
        "moderate negative",
        "slight negative",
        "neutral",
        "slight positive",
        "moderate positive",
        "strong positive",
    ]

                     
    patterns = [
        r"['\"]emotion['\"]:\s*['\"]([^'\"]+)['\"]",                                         
        r"emotion['\"]?\s*:\s*['\"]?([^'\"]+)['\"]?",                                   
        r"\b(strong\s+negative|moderate\s+negative|slight\s+negative|neutral|slight\s+positive|moderate\s+positive|strong\s+positive)\b",           
    ]

    for pattern in patterns:
        match = re.search(pattern, model_output.lower(), re.IGNORECASE)
        if match:
            emotion = match.group(1).lower().strip()
            if emotion in valid_emotions:
                return emotion, True, None

               
    if re.search(r"['\"]emotion['\"]", model_output.lower()):
        return None, False, "invalid_emotion_label"
    elif any(
        word in model_output.lower() for word in ["negative", "positive", "neutral"]
    ):
        return None, False, "emotion_found_but_not_extracted"
    else:
        return None, False, "no_emotion_pattern"


def evaluate_sentiment_intensity_analysis(result_file_path):


            
    with open(result_file_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    predictions = []
    ground_truths = []
    detailed_results = []
    extraction_errors = defaultdict(list)
    prediction_errors = defaultdict(list)

                      
    intensity_labels = [
        "strong negative",
        "moderate negative",
        "slight negative",
        "neutral",
        "slight positive",
        "moderate positive",
        "strong positive",
    ]

            
    for item in results:
        item_id = item["id"]
        model_output = item["model_output"]
        gt_label = item["ground_truth"].lower()

                
        pred_label, is_valid, error_type = extract_emotion_from_output(model_output)

                
        detailed_item = {
            "id": item_id,
            "model_output": model_output,
            "extracted_prediction": pred_label,
            "ground_truth": gt_label,
            "correct": pred_label == gt_label if pred_label else False,
            "valid": is_valid,
        }
        detailed_results.append(detailed_item)

                
        if not is_valid:
            extraction_errors[error_type].append(item_id)
        elif pred_label != gt_label:
            error_pattern = f"{gt_label}_to_{pred_label}"
            prediction_errors[error_pattern].append(item_id)

                      
        if is_valid:
            predictions.append(pred_label)
            ground_truths.append(gt_label)

               
    if len(predictions) == 0:
        return {
            "error": "No valid predictions found",
            "total_samples": len(results),
            "extraction_errors": dict(extraction_errors),
        }

            
    accuracy = accuracy_score(ground_truths, predictions)
    weighted_f1 = f1_score(ground_truths, predictions, average="weighted")
    macro_f1 = f1_score(ground_truths, predictions, average="macro")

          
    cm = confusion_matrix(ground_truths, predictions, labels=intensity_labels)

            
    class_report = classification_report(
        ground_truths,
        predictions,
        target_names=intensity_labels,
        output_dict=True,
        zero_division=0,
    )

            
    evaluation_result = {
        "task_info": {
            "task_name": "sentiment.intensity.analysis",
            "dataset": "CMU-MOSEI",
            "evaluation_time": datetime.now().isoformat(),
            "total_samples": len(results),
            "valid_predictions": len(predictions),
            "extraction_success_rate": round(len(predictions) / len(results), 4),
        },
        "metrics": {
            "ACC": round(accuracy, 4),
            "WAF": round(weighted_f1, 4),
            "Macro_F1": round(macro_f1, 4),
        },
        "per_class_metrics": {
            label: {
                "precision": round(class_report[label]["precision"], 4),
                "recall": round(class_report[label]["recall"], 4),
                "f1_score": round(class_report[label]["f1-score"], 4),
                "support": int(class_report[label]["support"]),
            }
            for label in intensity_labels
            if label in class_report
        },
        "confusion_matrix": {"labels": intensity_labels, "matrix": cm.tolist()},
        "error_analysis": {
            "extraction_errors": {
                error_type: {"count": len(sample_ids), "sample_ids": sample_ids}
                for error_type, sample_ids in extraction_errors.items()
            },
            "prediction_errors": {
                error_pattern: {"count": len(sample_ids), "sample_ids": sample_ids}
                for error_pattern, sample_ids in prediction_errors.items()
            },
        },
        "distribution": {
            "ground_truth": dict(Counter(ground_truths)),
            "predictions": dict(Counter(predictions)),
        },
    }

             
    base_name = result_file_path.replace(".json", "")

                 
    eval_output_file = f"{base_name}_evaluation.json"
    with open(eval_output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)

                    
    detailed_output_file = f"{base_name}_detailed_results.json"
    with open(detailed_output_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

                 
    problem_samples = [item for item in detailed_results if not item["correct"]]
    if problem_samples:
        problem_report_file = f"{base_name}_problem_samples.json"
        with open(problem_report_file, "w", encoding="utf-8") as f:
            json.dump(problem_samples, f, ensure_ascii=False, indent=2)


    if problem_samples:
        print(f"Problematic samples: {len(problem_samples)},see {problem_report_file}")

    return evaluation_result


      
if __name__ == "__main__":
    result_file = "model_result.json"             

    try:
        evaluation_result = evaluate_sentiment_intensity_analysis(result_file)

    except FileNotFoundError:
        print(f"Error: File not found {result_file}")
    except json.JSONDecodeError:
        print(f"Error: {result_file} Invalid format")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
