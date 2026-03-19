import csv
import numpy as np
from collections import defaultdict

def parse(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            r['novel_count'] = int(r['novel_count'])
            r['novel_matched'] = int(r['novel_matched'])
            r['novel_recall'] = float(r['novel_recall']) if r['novel_recall'] else None
            r['novel_false_positives'] = int(r['novel_false_positives'])
            r['trial'] = int(r['trial'])
            rows.append(r)
    return rows

main = [r for r in parse('results/master_results.csv')
        if r.get('case_type') == 'novel_virus' and r['method'] in ('zeroshot', 'fewshot')]
hold = parse('results/novel_holdout/holdout_results.csv')
combined = main + hold

for src_label, src_rows in [('BENCHMARK 25', main), ('HOLDOUT 25', hold), ('ALL 50', combined)]:
    print(f'\n=== {src_label} ===')
    for m in ['zeroshot', 'fewshot']:
        mr = [r for r in src_rows if r['method'] == m]
        recalls = [r['novel_recall'] for r in mr if r['novel_recall'] is not None]
        perfect = sum(1 for r in recalls if r == 1.0)
        fps = sum(r['novel_false_positives'] for r in mr)
        matched = sum(r['novel_matched'] for r in mr)
        total = sum(r['novel_count'] for r in mr)
        print(f'  {m:>10}: recall={np.mean(recalls):.4f} std={np.std(recalls):.4f} '
              f'perfect={perfect}/{len(recalls)} matched={matched}/{total} FPs={fps}')

print('\n=== PER-TRIAL (ALL 50) ===')
for m in ['zeroshot', 'fewshot']:
    for t in [1, 2, 3]:
        tr = [r for r in combined if r['method'] == m and r['trial'] == t]
        recalls = [r['novel_recall'] for r in tr if r['novel_recall'] is not None]
        perf = sum(1 for r in recalls if r == 1.0)
        print(f'  {m:>10} T{t}: recall={np.mean(recalls):.4f} perfect={perf}/{len(recalls)}')

print('\n=== IMPERFECT PATIENTS (ALL 50) ===')
imperfect = defaultdict(lambda: defaultdict(list))
for r in combined:
    if r['novel_recall'] is not None and r['novel_recall'] < 1.0:
        short = r['filename'].replace('NOVEL_SYNTHETIC_v3_', '').replace('.txt', '')
        imperfect[short][r['method']].append((r['trial'], r['novel_recall']))

for name in sorted(imperfect):
    parts = []
    for m in ['zeroshot', 'fewshot']:
        if m in imperfect[name]:
            trials = imperfect[name][m]
            detail = ', '.join(f'T{t}={rc:.2f}' for t, rc in sorted(trials))
            parts.append(f'{m}: {detail}')
    print(f'  {name}: {" | ".join(parts)}')
