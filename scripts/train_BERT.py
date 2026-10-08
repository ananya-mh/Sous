import json
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
import evaluate

# --- 1. CONFIGURATION ---
MODEL_CHECKPOINT = "bert-base-uncased"
OUTPUT_DIR = "./bert_recipe_model"
INPUT_FILE = "data/training_data.json"

# Hyperparameters for RTX 4060 Ti (8GB)
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5

# --- 2. LOAD & PREPARE DATA ---
print(f"Loading data from {INPUT_FILE}...")
try:
    with open(INPUT_FILE, 'r') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: {INPUT_FILE} not found. Please run preprocess.py first.")
    exit()

# Define the exact Label Map (BIO Scheme)
# This must match what your preprocess.py generates
label_list = ["O", "B-AMT", "I-AMT", "B-UNIT", "I-UNIT", "B-NAME", "I-NAME", "B-DESC", "I-DESC"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

print("Formatting data for BERT...")
data_dict = {"tokens": [], "ner_tags": []}

for sentence in raw_data:
    # sentence is a list of [token, label]
    tokens = [item[0] for item in sentence]
    tags = [item[1] for item in sentence]
    
    try:
        # Convert string tags (e.g., 'B-UNIT') to IDs (e.g., 3)
        tag_ids = [label2id[t] for t in tags]
        data_dict["tokens"].append(tokens)
        data_dict["ner_tags"].append(tag_ids)
    except KeyError as e:
        # This catches tags not in our list (e.g. typos in preprocess.py)
        continue

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict(data_dict)

# Split: 80% Train, 20% Test
dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(f"Training on {len(dataset['train'])} samples, Testing on {len(dataset['test'])} samples.")

# --- 3. TOKENIZATION & ALIGNMENT ---
print("Initializing Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens (CLS, SEP) get -100 so loss ignores them
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Start of a new word -> use the label
                label_ids.append(label[word_idx])
            else:
                # Sub-word part (e.g., ##ed) -> ignore (-100)
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# --- 4. METRICS (Using seqeval) ---
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],  # pyright: ignore[reportOptionalSubscript]
        "recall": results["overall_recall"],  # pyright: ignore[reportOptionalSubscript]
        "f1": results["overall_f1"],  # pyright: ignore[reportOptionalSubscript]
        "accuracy": results["overall_accuracy"],  # pyright: ignore[reportOptionalSubscript]
    }

# --- 5. MODEL SETUP ---
print("Loading BERT Model...")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# --- 6. TRAINING ARGUMENTS (RTX 4060 Ti OPTIMIZED) ---
args = TrainingArguments(
    output_dir=OUTPUT_DIR,  # pyright: ignore[reportCallIssue]
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    
    # --- GPU Optimizations ---
    fp16=True,                # Mixed Precision (Crucial for 4060 Ti)
    dataloader_num_workers=0, # Set to 0 for Windows (multiprocessing issues)
    logging_dir='./logs',
    logging_steps=100,
    report_to="none"          # Disable WandB/MLFlow prompts
)

if __name__ == "__main__":
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 7. START TRAINING ---
    print("\n🚀 Starting Training...")
    trainer.train()

    # --- 8. SAVE ---
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR) # Important: Save tokenizer too!

    print("✅ DONE! You can now run 'inference.py' (make sure to update it to use BERT pipeline).")