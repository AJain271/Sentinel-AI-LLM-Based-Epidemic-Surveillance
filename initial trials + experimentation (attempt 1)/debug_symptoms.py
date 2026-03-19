
import pandas as pd
import os
import sys

# Redirect stdout to a file
sys.stdout = open('debug_output.txt', 'w')

# Load the CSV
csv_file = "gold_standard_trinary.csv"
if not os.path.exists(csv_file):
    print("CSV not found!")
    exit()

df = pd.read_csv(csv_file)

print("Columns in CSV:", df.columns.tolist())

# Simulate the widget update behavior
SYMPTOM_GROUPS = {
    "Respiratory (The 'Outbreak' Core)": {
        "Fever": 'Includes: chills, night sweats, "burning up", "hot"',
        "Cough": 'Includes: hacking, productive/wet, dry, "barking"',
        "Dyspnea": 'Includes: shortness of breath, \"winded\", difficulty breathing',
        "Wheezing": 'Includes: whistling sound when breathing',
        "Rhinorrhea": 'Includes: runny nose, nasal drip',
        "Pharyngitis": 'Includes: sore throat, \"scratchy throat\"'
    },
    "Cardiac": {
        "Chest Pain": 'Includes: tightness, pressure, "elephant on chest"',
        "Palpitations": 'Includes: racing heart, skipped beats, "thumping"',
        "Syncope": 'Includes: fainting, blacking out',
        "Edema": 'Includes: swollen ankles, leg swelling'
    },
    "Musculoskeletal": {
        "Myalgia": 'Includes: muscle aches, body aches',
        "Arthralgia": 'Includes: joint pain, "stiff joints"',
        "Back Pain": 'Includes: lower back pain, lumbar strain'
    },
    "Gastrointestinal": {
        "Nausea/Emesis": 'Includes: feeling sick to stomach, vomiting',
        "Diarrhea": 'Includes: loose stools',
        "Abdominal Pain": 'Includes: stomach cramps, \"belly ache\"'
    },
    "Dermatology": {
        "Pruritus": 'Includes: itching, \"itchy skin\"',
        "Rash": 'Includes: hives, redness, skin lesion, \"breakout\"'
    }
}
 
widget_options = []
for group, symptoms in SYMPTOM_GROUPS.items():
    for s in symptoms:
        widget_options.append(s)

print("\nWidget Options:", widget_options)

# Check for mismatches
in_csv_not_widget = [c for c in df.columns if c not in widget_options and c not in ['filename', 'category', 'full_path']]
in_widget = [c for c in widget_options if c in df.columns]

print("\nColumns in CSV but not in Widget:")
print(in_csv_not_widget)

print("\nColumns in Widget and CSV:")
print(in_widget)

try:
    index = 0
    symptom = "Dyspnea"
    new_value = 2
    
    # Verify column exists
    if symptom not in df.columns:
        print(f"Adding missing column {symptom}")
        df[symptom] = 0
    else:
        print(f"Column {symptom} exists in CSV.")
        
    print(f"Old value for {symptom} at index {index}: {df.at[index, symptom]}")
    df.at[index, symptom] = new_value
    print(f"New value for {symptom} at index {index}: {df.at[index, symptom]}")
    
    # Check 'Shortness of Breath' if exists
    if 'Shortness of Breath' in df.columns:
        print(f"Value for 'Shortness of Breath' at index {index}: {df.at[index, 'Shortness of Breath']}")
        
    # Save to a temp file
    df.to_csv("test_update_output.csv", index=False)
    print("Saved test output.")
    
except Exception as e:
    print(f"Error during update: {e}")
    
sys.stdout.close()
