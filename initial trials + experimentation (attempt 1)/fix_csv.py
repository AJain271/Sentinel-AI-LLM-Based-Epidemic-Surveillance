
import pandas as pd
import shutil

# Config
input_file = 'gold_standard_trinary.csv'
backup_file = 'gold_standard_trinary_backup.csv'

# Create backup
shutil.copyfile(input_file, backup_file)
print(f"Created backup at {backup_file}")

df = pd.read_csv(input_file)
print("Original Columns:", df.columns.tolist())

# Mappings: Target (New) <- Sources (Old)
mappings = {
    'Dyspnea': ['Shortness of Breath'],
    'Pharyngitis': ['Sore Throat'],
    'Nausea/Emesis': ['Nausea', 'Vomiting'],
    'Arthralgia': ['Joint Pain'],
    'Myalgia': ['Muscle Pain'],
    'Abdominal Pain': ['Abdominal Pain'], # Same name, just ensuring it acts as valid target
}

# List of columns to drop (Obsolete)
# We will identify these dynamically: any column not in the KEEP list and not in metadata
keep_columns = [
    'filename', 'category', 'full_path',
    'Fever', 'Cough', 'Dyspnea', 'Wheezing', 'Rhinorrhea', 'Pharyngitis',
    'Chest Pain', 'Palpitations', 'Syncope', 'Edema',
    'Myalgia', 'Arthralgia', 'Back Pain',
    'Nausea/Emesis', 'Diarrhea', 'Abdominal Pain',
    'Pruritus', 'Rash'
]

# 1. Migrate Data
for target, sources in mappings.items():
    if target not in df.columns:
        df[target] = 0
    
    for source in sources:
        if source in df.columns:
            # Copy data from source to target if target is 0
            # We use a mask where target is 0 and source is not 0
            mask = (df[target] == 0) & (df[source] != 0)
            count = mask.sum()
            if count > 0:
                print(f"Migrating {count} values from '{source}' to '{target}'")
                df.loc[mask, target] = df.loc[mask, source]

# 2. Add missing columns with 0
for col in keep_columns:
    if col not in df.columns:
        print(f"Adding missing expected column: {col}")
        df[col] = 0

# 3. Drop obsolete columns
to_drop = [c for c in df.columns if c not in keep_columns]
if to_drop:
    print(f"Dropping obsolete columns: {to_drop}")
    df.drop(columns=to_drop, inplace=True)

# 4. Reorder columns
df = df[keep_columns]

# Save
df.to_csv(input_file, index=False)
print("Transformation complete. Saved cleaned CSV.")
print("Final Columns:", df.columns.tolist())
