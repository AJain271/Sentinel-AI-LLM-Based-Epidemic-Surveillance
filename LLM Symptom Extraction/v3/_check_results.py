"""Quick check of benchmark results."""
import csv, statistics
from collections import defaultdict

rows = list(csv.DictReader(open('results/master_results.csv', encoding='utf-8')))
print(f"Total rows: {len(rows)}")

# Per trial breakdown
print("\n=== Per-Trial Accuracy ===")
print(f"{'Trial':>5}  {'ZeroShot':>8}  {'FewShot':>8}  {'Regex':>8}")
for t in ['1','2','3']:
    parts = []
    for m in ['zeroshot','fewshot','regex']:
        vals = [float(r['accuracy']) for r in rows if r['trial']==t and r['method']==m]
        parts.append(f"{sum(vals)/len(vals):8.1%}")
    print(f"  {t:>3}  {'  '.join(parts)}")

# Per case type
print("\n=== Accuracy by Case Type (avg all trials) ===")
print(f"{'CaseType':>18}  {'ZeroShot':>8}  {'FewShot':>8}  {'Regex':>8}")
ct_data = defaultdict(lambda: defaultdict(list))
for r in rows:
    ct_data[r['case_type']][r['method']].append(float(r['accuracy']))
for ct in sorted(ct_data):
    parts = []
    for m in ['zeroshot','fewshot','regex']:
        vals = ct_data[ct][m]
        parts.append(f"{sum(vals)/len(vals):8.1%}")
    print(f"  {ct:>16}  {'  '.join(parts)}")

# Novel recall
print("\n=== Novel Recall by Case Type ===")
for ct in sorted(ct_data):
    for m in ['zeroshot','fewshot']:
        vals = [float(r['novel_recall']) for r in rows if r['case_type']==ct and r['method']==m and r['novel_recall']]
        if vals:
            avg_r = sum(vals)/len(vals)
            print(f"  {ct:>16}  {m:>10}  recall={avg_r:.3f}  (n={len(vals)})")

# Novel false positives
print("\n=== Novel False Positives (avg per sample) ===")
for m in ['zeroshot','fewshot','regex']:
    vals = [int(r['novel_false_positives']) for r in rows]
    # filter to just this method
    vals = [int(r['novel_false_positives']) for r in rows if r['method']==m]
    print(f"  {m:>12}  avg_FP={sum(vals)/len(vals):.2f}")

# Trial consistency
print("\n=== Trial Consistency (stdev of mean accuracy) ===")
for m in ['zeroshot','fewshot']:
    trial_means = []
    for t in ['1','2','3']:
        vals = [float(r['accuracy']) for r in rows if r['trial']==t and r['method']==m]
        trial_means.append(sum(vals)/len(vals))
    sd = statistics.stdev(trial_means)
    print(f"  {m:>12}  means={[round(x,4) for x in trial_means]}  stdev={sd:.4f}")

# Token cost comparison
print("\n=== Token Usage (avg per sample) ===")
for m in ['zeroshot','fewshot','regex']:
    prompt = [int(r['prompt_tokens']) for r in rows if r['method']==m]
    compl = [int(r['completion_tokens']) for r in rows if r['method']==m]
    print(f"  {m:>12}  prompt={sum(prompt)//len(prompt):>5}  completion={sum(compl)//len(compl):>5}  total={sum(prompt)//len(prompt)+sum(compl)//len(compl):>5}")

# Worst performing transcripts
print("\n=== Bottom 10 Transcripts (ZeroShot, avg across trials) ===")
zs_rows = [r for r in rows if r['method']=='zeroshot']
file_accs = defaultdict(list)
for r in zs_rows:
    file_accs[r['filename']].append(float(r['accuracy']))
bottom = sorted(file_accs.items(), key=lambda x: sum(x[1])/len(x[1]))[:10]
for fname, accs in bottom:
    ct = next(r['case_type'] for r in rows if r['filename']==fname)
    print(f"  {sum(accs)/len(accs):.1%}  {ct:>14}  {fname}")
