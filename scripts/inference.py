from transformers import pipeline
from groq import Groq
import re
import os
import json
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "bert_recipe_model" # Your local BERT model folder

# Groq-hosted open-weight model; override with GROQ_MODEL in .env
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# --- 2. LOAD BERT MODEL (Local Parser) ---
print("⏳ Loading BERT Model...")
try:
    nlp = pipeline("token-classification", model=MODEL_PATH, aggregation_strategy="first")
except Exception as e:
    print(f"❌ Error loading BERT: {e}")
    exit()

def clean_text(text):
    """
    Fixes BERT tokenizer artifacts (## and punctuation spacing).
    """
    text = text.replace("##", "")
    text = re.sub(r"\s+'\s+", "'", text) # Fix: hershey ' s -> hershey's
    text = re.sub(r"\s+([,.;])", r"\1", text) # Fix: minced , -> minced,
    return text.strip()

def parse_recipe_bert(text):
    """
    Uses local BERT model to extract Amount, Unit, Item, Descriptor.
    """
    results = nlp(text)
    parsed = {"amount": [], "unit": [], "item": [], "descriptor": []}
    
    for entity in results:
        label = entity['entity_group']
        word = clean_text(entity['word'])
        
        if label == 'AMT': parsed['amount'].append(word)
        elif label == 'UNIT': parsed['unit'].append(word)
        elif label == 'NAME': parsed['item'].append(word)
        elif label == 'DESC': parsed['descriptor'].append(word)
            
    return {k: " ".join(v) for k, v in parsed.items()}

def get_substitute(item, amount, unit, constraint):
    """
    Uses the LLM (via Groq) to find a substitute and calculate new math.
    """
    # Prompt Engineering: Force JSON output
    prompt = f"""
    Act as a professional food scientist.
    
    Task: Suggest a substitute for the ingredient below based on the constraint.
    
    Input: {amount} {unit} of {item}
    Constraint: {constraint}
    
    Rules:
    1. If a valid substitute exists, calculate the new quantity based on potency/density.
    2. If NO valid substitute exists for this constraint, return "found": false.
    3. Return ONLY valid JSON.
    
    Output JSON Format:
    {{
        "found": true,
        "substitute_item": "Name of new ingredient",
        "new_amount": "Number (decimal or fraction string)",
        "new_unit": "Unit (usually same, but changes for eggs/etc)",
        "reason": "Brief explanation of why this works"
    }}
    """
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0)
        
        # Clean response (remove ```json wrappers if present)
        clean_json = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
        
    except Exception as e:
        return {"found": False, "reason": f"API Error: {str(e)}"}

# --- 4. MAIN INTERFACE ---
print("\n" + "="*50)
print(" 🥗 HYBRID RECIPE AI (BERT + LLM)")
print("="*50)

while True:
    user_input = input("\n📝 Enter Recipe Line (q to quit): ")
    if user_input.lower() == 'q': break
    
    # Step 1: Parse locally (Fast & Free)
    try:
        parsed_result = parse_recipe_bert(user_input)
        print(f"🔹 BERT PARSED: {parsed_result}")
    except Exception as e:
        print(f"⚠️ Parsing Error: {e}")
        continue
    
    # Step 2: Substitute via Cloud (Smart)
    if parsed_result['item']:
        constraint = input("   ❓ Constraint (vegan/keto/gluten-free/none): ").strip().lower()
        
        if constraint and constraint != 'none':
            print("   ✨ Asking the LLM for advice...")
            
            llm_result = get_substitute(
                item=parsed_result['item'],
                amount=parsed_result['amount'],
                unit=parsed_result['unit'],
                constraint=constraint
            )
            
            if llm_result.get('found'):
                print(f"\n   ✅ LLM SUGGESTION:")
                print(f"   Swap with: {llm_result['substitute_item']}")
                print(f"   Quantity:  {llm_result['new_amount']} {llm_result.get('new_unit', '')}")
                print(f"   Reason:    {llm_result['reason']}")
            else:
                print(f"   ⚠️ LLM Response: {llm_result.get('reason', 'No good substitute found.')}")