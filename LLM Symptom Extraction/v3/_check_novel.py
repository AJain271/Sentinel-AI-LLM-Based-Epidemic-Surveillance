"""Find novel virus cases where the LLM missed some hallmark symptoms."""
import csv

rows = list(csv.DictReader(open('results/master_results.csv', encoding='utf-8')))

print("=== Novel Virus Samples with Imperfect Recall ===")
print("(any trial, ZS or FS only — these are the ones to manually review)")
print()

issues = []
for r in rows:
    if r['method'] == 'regex':
        continue
    nc = int(r['novel_count'])
    if nc > 0:
        nm = int(r['novel_matched'])
        nr = float(r['novel_recall']) if r['novel_recall'] else 0
        if nr < 1.0:
            issues.append(r)

# Group by filename
from collections import defaultdict
grouped = defaultdict(list)
for r in issues:
    grouped[r['filename']].append(r)

print(f"{'Filename':<58} {'Method':<10} {'Trial':<6} {'Match':<7} {'Recall':<7}")
print("-" * 95)
for fname in sorted(grouped):
    for r in sorted(grouped[fname], key=lambda x: (x['method'], x['trial'])):
        nc = int(r['novel_count'])
        nm = int(r['novel_matched'])
        nr = float(r['novel_recall']) if r['novel_recall'] else 0
        print(f"{fname[:57]:<58} {r['method']:<10} {r['trial']:<6} {nm}/{nc:<5} {nr:.3f}")

total_novel_runs = sum(1 for r in rows if int(r.get('novel_count', 0)) > 0 and r['method'] != 'regex')
print(f"\nImperfect recalls: {len(issues)} out of {total_novel_runs} novel LLM runs")
print(f"Unique transcripts affected: {len(grouped)}")

# Also show novel false positives (non-novel cases where LLM hallucinated novel symptoms)
print("\n=== Novel False Positives (non-novel cases flagged with novel symptoms) ===")
fps = []
for r in rows:
    if r['method'] == 'regex':
        continue
    nfp = int(r['novel_false_positives'])
    if nfp > 0:
        fps.append(r)

if fps:
    print(f"{'Filename':<58} {'Method':<10} {'Trial':<6} {'FP Count':<8}")
    print("-" * 85)
    for r in fps:
        print(f"{r['filename'][:57]:<58} {r['method']:<10} {r['trial']:<6} {r['novel_false_positives']:<8}")
else:
    print("  None found.")
