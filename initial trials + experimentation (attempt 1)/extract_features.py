import os
import glob
import json
import time
import pandas as pd
from openai import OpenAI

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "Clean Transcripts")
OUTPUT_FILE = "symptom_features.csv"
MODEL = "gpt-4o"

# Using the same Category Map and Symptoms as experiments.ipynb
CATEGORY_MAP = {
    'RES': ['Fever', 'Cough', 'Dyspnea', 'Wheezing', 'Rhinorrhea', 'Pharyngitis', 'Congestion'],
    'CAR': ['Chest Pain', 'Palpitations', 'Syncope', 'Edema'],
    'MSK': ['Myalgia', 'Arthralgia', 'Back Pain'],
    'GAS': ['Nausea/Emesis', 'Diarrhea', 'Abdominal Pain'],
    'DER': ['Pruritus', 'Rash'],
    'OTH': ['Headache', 'Exposure To Sick', 'Anosmia', 'Ageusia', 'Fatigue']
}

ALL_SYMPTOMS = []
for cat_list in CATEGORY_MAP.values():
    ALL_SYMPTOMS.extend(cat_list)

# Initialize OpenAI Client
api_key = os.environ.get("OPENAI_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=api_key)

def get_zeroshot_json_prompts(transcript):
    sys = "You are a medical assistant expert in symptom extraction. You output strictly in JSON."
    symptom_list = json.dumps(ALL_SYMPTOMS)
    # Using the exact same prompt from experiments.ipynb
    user = f'''
Analyze the transcript and identify the presence of these symptoms: {symptom_list}.

Return a JSON object with two keys:
1. "symptoms": detailed object where keys are symptom names and values are 0 (Not Mentioned), 1 (Negated), 2 (Present).
2. "category": one of ["RES", "CAR", "MSK", "GAS", "DER"]

Transcript:
"""{transcript}"""
'''
    return sys, user

def call_llm(transcript):
    sys_p, user_p = get_zeroshot_json_prompts(transcript)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p}
            ],
            response_format={ "type": "json_object" }, 
            temperature=0.0 # Zero-shot should be deterministic for feature extraction
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def parse_response(response_text):
    try:
        data = json.loads(response_text)
        symptoms = data.get("symptoms", {})
        category = data.get("category", "Unknown")
        
        # Ensure all symptoms are present
        row = {s: symptoms.get(s, 0) for s in ALL_SYMPTOMS}
        row['Predicted_Category'] = category
        return row
    except json.JSONDecodeError:
        print("Failed to parse JSON response")
        return None

def main():
    # 1. Get list of files
    files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    print(f"Found {len(files)} transcripts in {DATA_DIR}")
    
    # 2. Check for existing progress
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        processed_files = set(df_existing['Filename'].tolist())
        print(f"Resuming... {len(processed_files)} files already processed.")
    else:
        df_existing = pd.DataFrame()
        processed_files = set()
        
    # Open CSV in append mode (or create)
    # We will write row by row to save progress
    
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        
        if filename in processed_files:
            continue
            
        print(f"Processing {i+1}/{len(files)}: {filename}...", flush=True)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            continue
            
        # Call LLM
        response_text = call_llm(transcript)
        if not response_text:
            print(f"Skipping {filename} due to LLM error.")
            continue
            
        # Parse
        row = parse_response(response_text)
        if not row:
            print(f"Skipping {filename} due to parse error.")
            continue
            
        row['Filename'] = filename
        
        # Append to CSV
        df_row = pd.DataFrame([row])
        
        # Reorder columns to have Filename first, then Predicted_Category, then symptoms
        cols = ['Filename', 'Predicted_Category'] + ALL_SYMPTOMS
        df_row = df_row[cols]
        
        if not os.path.exists(OUTPUT_FILE):
            df_row.to_csv(OUTPUT_FILE, index=False)
        else:
            df_row.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            
        # Rate limit
        time.sleep(0.2)
        
    print("Done! Results saved to", OUTPUT_FILE)

if __name__ == "__main__":
    main()
