import json
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    DataCollatorForTokenClassification,
    Trainer
)
from transformers import TrainingArguments

import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from pathlib import Path

# --- 1. CONFIGURATION ---
MODEL_PATH = "./bert_recipe_model"
INPUT_FILE = "data/training_data.json"
OUTPUT_REPORT = "evaluation_report.json"

# Define the exact Label Map (BIO Scheme) - must match train_BERT.py
label_list = ["O", "B-AMT", "I-AMT", "B-UNIT", "I-UNIT", "B-NAME", "I-NAME", "B-DESC", "I-DESC"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Entity types (for grouping BIO tags)
entity_types = {
    "AMT": ["B-AMT", "I-AMT"],
    "UNIT": ["B-UNIT", "I-UNIT"],
    "NAME": ["B-NAME", "I-NAME"],
    "DESC": ["B-DESC", "I-DESC"]
}

def load_data(input_file):
    """Load and prepare data from JSON file."""
    print(f"Loading data from {input_file}...")
    try:
        with open(input_file, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {input_file} not found. Please run preprocess.py first.")
        exit(1)
    
    print("Formatting data for BERT...")
    data_dict = {"tokens": [], "ner_tags": []}
    
    for sentence in raw_data:
        tokens = [item[0] for item in sentence]
        tags = [item[1] for item in sentence]
        
        try:
            tag_ids = [label2id[t] for t in tags]
            data_dict["tokens"].append(tokens)
            data_dict["ner_tags"].append(tag_ids)
        except KeyError as e:
            continue
    
    return Dataset.from_dict(data_dict)

def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize and align labels with subword tokens."""
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        padding=True
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_detailed_metrics(predictions, labels, label_list):
    """Compute detailed metrics including per-entity and overall metrics."""
    # Flatten predictions and labels
    flat_predictions = []
    flat_labels = []
    
    for pred, label in zip(predictions, labels):
        for p, l in zip(pred, label):
            if l != -100:  # Ignore special tokens
                flat_predictions.append(p)
                flat_labels.append(l)
    
    # Convert to label strings
    true_predictions = [label_list[p] for p in flat_predictions]
    true_labels = [label_list[l] for l in flat_labels]
    
    # Overall metrics using seqeval
    seqeval = evaluate.load("seqeval")
    
    # Group predictions and labels by sentence for seqeval
    sentence_predictions = []
    sentence_labels = []
    current_pred = []
    current_label = []
    
    for pred, label in zip(predictions, labels):
        current_pred = []
        current_label = []
        for p, l in zip(pred, label):
            if l != -100:
                current_pred.append(label_list[p])
                current_label.append(label_list[l])
        sentence_predictions.append(current_pred)
        sentence_labels.append(current_label)
    
    seqeval_results = seqeval.compute(
        predictions=sentence_predictions, 
        references=sentence_labels
    )
    
    # Per-entity metrics
    entity_metrics = {}
    for entity_type, tags in entity_types.items():
        entity_predictions = []
        entity_labels = []
        
        for pred, label in zip(true_predictions, true_labels):
            pred_is_entity = any(pred.startswith(tag) for tag in tags)
            label_is_entity = any(label.startswith(tag) for tag in tags)
            entity_predictions.append(1 if pred_is_entity else 0)
            entity_labels.append(1 if label_is_entity else 0)
        
        # Calculate precision, recall, F1 for this entity
        tp = sum(1 for p, l in zip(entity_predictions, entity_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(entity_predictions, entity_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(entity_predictions, entity_labels) if p == 0 and l == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        entity_metrics[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(entity_labels)
        }
    
    # Classification report (per-label metrics)
    classification_rep = classification_report(
        true_labels, 
        true_predictions, 
        labels=label_list,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, true_predictions, labels=label_list)
    
    return {
        "overall": {
            "precision": seqeval_results.get("overall_precision", 0.0),
            "recall": seqeval_results.get("overall_recall", 0.0),
            "f1": seqeval_results.get("overall_f1", 0.0),
            "accuracy": seqeval_results.get("overall_accuracy", 0.0)
        },
        "per_entity": entity_metrics,
        "per_label": classification_rep,
        "confusion_matrix": cm.tolist(),
        "label_list": label_list
    }

def evaluate_model(model_path, test_dataset, tokenizer, batch_size=16):
    """Evaluate the trained model on test dataset."""
    print(f"\n📊 Loading model from {model_path}...")
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Make sure the model has been trained first (run train_BERT.py)")
        exit(1)
    
    # Tokenize test dataset
    print("🔤 Tokenizing test dataset...")
    tokenized_test = test_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )
    
    # Setup trainer for evaluation
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./eval_temp",
        per_device_eval_batch_size=batch_size,
        fp16=True,
        dataloader_num_workers=0,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
    )
    
    # Run evaluation
    print("🚀 Running evaluation...")
    eval_results = trainer.evaluate(tokenized_test)
    
    # Get predictions
    print("🔮 Generating predictions...")
    predictions = trainer.predict(tokenized_test)
    
    # Compute detailed metrics
    print("📈 Computing detailed metrics...")
    pred_ids = np.argmax(predictions.predictions, axis=2)
    labels = predictions.label_ids
    
    detailed_metrics = compute_detailed_metrics(pred_ids, labels, label_list)
    
    return detailed_metrics, eval_results

def print_metrics_report(metrics, eval_results):
    """Print a formatted metrics report."""
    print("\n" + "="*70)
    print(" 📊 BERT MODEL EVALUATION REPORT")
    print("="*70)
    
    # Overall Metrics
    print("\n🎯 OVERALL METRICS:")
    print("-" * 70)
    overall = metrics["overall"]
    print(f"  Precision:  {overall['precision']:.4f}")
    print(f"  Recall:     {overall['recall']:.4f}")
    print(f"  F1-Score:   {overall['f1']:.4f}")
    print(f"  Accuracy:   {overall['accuracy']:.4f}")
    
    # Per-Entity Metrics
    print("\n📋 PER-ENTITY METRICS:")
    print("-" * 70)
    print(f"{'Entity':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    for entity_type, entity_metrics in metrics["per_entity"].items():
        print(f"{entity_type:<10} {entity_metrics['precision']:<12.4f} "
              f"{entity_metrics['recall']:<12.4f} {entity_metrics['f1']:<12.4f} "
              f"{entity_metrics['support']:<10}")
    
    # Per-Label Metrics (Top labels)
    print("\n🏷️  PER-LABEL METRICS (Top Labels):")
    print("-" * 70)
    per_label = metrics["per_label"]
    if isinstance(per_label, dict):
        # Show metrics for each label
        print(f"{'Label':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        for label in label_list:
            if label in per_label:
                label_metrics = per_label[label]
                print(f"{label:<15} {label_metrics.get('precision', 0):<12.4f} "
                      f"{label_metrics.get('recall', 0):<12.4f} "
                      f"{label_metrics.get('f1-score', 0):<12.4f} "
                      f"{label_metrics.get('support', 0):<10}")
    
    # Loss
    if "eval_loss" in eval_results:
        print(f"\n📉 Evaluation Loss: {eval_results['eval_loss']:.4f}")
    
    print("\n" + "="*70)

def save_report(metrics, eval_results, output_file):
    """Save evaluation report to JSON file."""
    report = {
        "overall_metrics": metrics["overall"],
        "per_entity_metrics": metrics["per_entity"],
        "per_label_metrics": metrics["per_label"],
        "eval_loss": eval_results.get("eval_loss", None),
        "label_list": metrics["label_list"],
        "confusion_matrix": metrics["confusion_matrix"]
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Evaluation report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained BERT model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=MODEL_PATH,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--data_file", 
        type=str, 
        default=INPUT_FILE,
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=OUTPUT_REPORT,
        help="Output file for evaluation report"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--test_split", 
        type=float, 
        default=0.2,
        help="Test split ratio (if evaluating on full dataset)"
    )
    parser.add_argument(
        "--use_test_set",
        action="store_true",
        help="Use test set from train/test split (same seed as training)"
    )
    
    args = parser.parse_args()
    
    # Load tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load data
    dataset = load_data(args.data_file)
    
    # Split dataset (using same seed as training for consistency)
    if args.use_test_set:
        dataset = dataset.train_test_split(test_size=args.test_split, seed=42)
        test_dataset = dataset["test"]
        print(f"📊 Using test set: {len(test_dataset)} samples")
    else:
        # Evaluate on full dataset
        test_dataset = dataset
        print(f"📊 Evaluating on full dataset: {len(test_dataset)} samples")
    
    # Evaluate model
    metrics, eval_results = evaluate_model(
        args.model_path, 
        test_dataset, 
        tokenizer, 
        batch_size=args.batch_size
    )
    
    # Print report
    print_metrics_report(metrics, eval_results)
    
    # Save report
    save_report(metrics, eval_results, args.output)
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()

