import pandas as pd
import os
import random
import glob

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "Clean Transcripts")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "test_gold_standard.csv")

# Target counts for sampling
TARGETS = {
    "RES": 4,  # Respiratory
    "CAR": 4,  # Cardiac
    "MSK": 4,  # Musculoskeletal
    "GAS": 4,  # Gastrointestinal
    "DER": 4   # Dermatological
}

# List of symptoms to annotate
SYMPTOMS = [
    "Fever", "Cough", "Shortness of Breath", "Sore Throat", "Chest Pain", 
    "Fatigue", "Headache", "Nausea", "Vomiting", "Diarrhea", 
    "Abdominal Pain", "Rash", "Joint Pain", "Muscle Pain", "Dizziness"
]

def get_sampled_cases(data_dir, targets, output_file):
    print("Performing random sampling...")
    sampled_files = []
    
    all_files = glob.glob(os.path.join(data_dir, "*.txt"))
    print(f"Found {len(all_files)} files in {data_dir}")
    file_map = {}
    
    # Group files by prefix
    for f in all_files:
        basename = os.path.basename(f)
        prefix = basename[:3]
        if prefix not in file_map:
            file_map[prefix] = []
        file_map[prefix].append(f)
        
    # Sample according to targets
    records = []
    for prefix, count in targets.items():
        available = file_map.get(prefix, [])
        print(f"Prefix {prefix}: Found {len(available)} files.")
        if len(available) < count:
            print(f"Warning: Requested {count} for {prefix}, but only found {len(available)}. Taking all.")
            selection = available
        else:
            selection = random.sample(available, count)
            
        for path in selection:
            filename = os.path.basename(path)
            records.append({
                "filename": filename,
                "category": prefix,
                "full_path": path
            })
            
    # Create DataFrame
    df = pd.DataFrame(records)
    # Initialize symptoms columns with 0
    for s in SYMPTOMS:
        df[s] = 0
        
    # Save initial version
    df.to_csv(output_file, index=False)
    print(f"Sampled {len(df)} cases. Saved to {output_file}.")
    return df

if __name__ == "__main__":
    df = get_sampled_cases(DATA_DIR, TARGETS, OUTPUT_FILE)
    print("Columns:", df.columns)
    print(df.head())
    print("Category counts:")
    print(df['category'].value_counts())
