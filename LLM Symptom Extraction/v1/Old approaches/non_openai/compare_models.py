"""Quick comparison: GPT-4o vs Llama 3.1 8B on RES0001.txt"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "non_openai"))

import extract_symptoms as es

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Clean Transcripts", "RES0001.txt"), "r", encoding="utf-8") as f:
    transcript = f.read()

# GPT-4o
es.BACKEND = "openai"
es._client = None
gpt = es.extract(transcript)

# Llama 3.1
es.BACKEND = "ollama"
es.OLLAMA_MODEL = "llama3.1:8b"
es._client = None
llama = es.extract(transcript)

labels = {0: "---", 1: "NEG", 2: "YES"}

print(f"{'SYMPTOM':20s}  GPT-4o  Llama3.1")
print("-" * 45)
diffs = 0
for key in es.SYMPTOM_KEYS:
    g = gpt["scores"][key]
    l = llama["scores"][key]
    marker = "  <-- DIFF" if g != l else ""
    if g > 0 or l > 0:
        print(f"{key:20s}  {labels[g]:>5s}   {labels[l]:>5s}{marker}")
        if g != l:
            diffs += 1

print(f"\nDifferences: {diffs}")
print(f"\nGPT-4o reasoning: {gpt['reasoning']}")
print(f"Llama reasoning:  {llama['reasoning']}")

gn = gpt.get("novel_symptoms", [])
ln = llama.get("novel_symptoms", [])
if gn:
    print(f"\nGPT-4o novel: {[ns['symptom'] for ns in gn]}")
if ln:
    print(f"Llama novel:  {[ns['symptom'] for ns in ln]}")
