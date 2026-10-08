import pandas as pd
import nltk
import ast
import re
import json

# List of common units to force-tag
KNOWN_UNITS = {
    "cup", "cups", "c.", "c", 
    "tsp", "teaspoon", "teaspoons", "t.", "t", 
    "tbsp", "tablespoon", "tablespoons", "T.", "T",
    "oz", "ounce", "ounces", 
    "lb", "pound", "pounds", 
    "g", "gram", "grams", 
    "kg", "kilogram", "kilograms",
    "ml", "milliliter", "milliliters",
    "l", "liter", "liters",
    "package", "packages", "pkg",
    "can", "cans",
    "slice", "slices",
    "clove", "cloves",
    "pinch", "pinches",
    "dash", "dashes"
}
STOP_WORDS = {",", ".", "and", "or", "with", "into", "to"}

def generate_bio_tags(row):
    input_text = str(row['input'])
    tokens = nltk.word_tokenize(input_text)
    labels = ['O'] * len(tokens)

    # Clean CSV values for matching
    qty = str(row['qty']).lower() if pd.notna(row['qty']) else ""
    unit = str(row['unit']).lower() if pd.notna(row['unit']) else ""
    name = str(row['name']).lower() if pd.notna(row['name']) else ""
    comment = str(row['comment']).lower() if pd.notna(row['comment']) else ""

    for i, token in enumerate(tokens):
        token_lower = token.lower()
        
        # --- IMPROVED LOGIC START ---
        
        # 1. FORCE TAG NUMBERS & FRACTIONS
        # Matches "1", "1/2", "0.5"
        if re.match(r'^\d+([/\.]\d+)?$', token_lower):
            labels[i] = 'B-AMT'
            continue # Skip other checks
            
        # 2. FORCE TAG KNOWN UNITS
        if token_lower in KNOWN_UNITS:
            labels[i] = 'B-UNIT'
            continue

        if token_lower in STOP_WORDS:
            labels[i] = 'O'
            continue

        # --- END IMPROVED LOGIC ---

        # 3. Standard CSV Matching (Fallback)
        if token_lower in name:
            if i > 0 and ('NAME' in labels[i-1]):
                labels[i] = 'I-NAME'
            else:
                labels[i] = 'B-NAME'
        
        elif token_lower in comment:
            labels[i] = 'B-DESC'
            
    return list(zip(tokens, labels))

# --- Main Execution ---
print("Loading dataset...")
df = pd.read_csv("data/nyt-ingredients-snapshot-2015.csv")

# Drop extremely complex lines for simplicity
df = df.dropna(subset=['input', 'name'])

print("Generating BIO tags... (this might take a moment)")
# Apply the function to create training data
training_data = df.apply(generate_bio_tags, axis=1).tolist()

output_file = 'data/training_data.json'
with open(output_file, 'w') as f:
    json.dump(training_data, f)
print(f"✅ Preprocessed data saved to: {output_file}")