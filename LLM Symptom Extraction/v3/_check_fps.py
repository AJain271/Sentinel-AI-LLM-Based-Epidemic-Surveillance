import csv
from collections import Counter

rows = list(csv.DictReader(open('results/master_results.csv')))
print("Case types:", Counter(r['case_type'] for r in rows))
print()

zs = [r for r in rows if r['method'] == 'zeroshot']
fps_all = [r for r in zs if int(r['novel_false_positives']) > 0]
print(f"Total ZS rows: {len(zs)}, with FP>0: {len(fps_all)}")
for r in fps_all:
    fn = r['filename'][:55]
    ct = r['case_type']
    print(f"  {fn}  trial={r['trial']}  FPs={r['novel_false_positives']}  case={ct}")

# Check ALL methods for FPs
print("\nAll methods FP>0:")
for r in rows:
    if int(r['novel_false_positives']) > 0:
        fn = r['filename'][:55]
        print(f"  {r['method']}  {fn}  trial={r['trial']}  FPs={r['novel_false_positives']}  case={r['case_type']}")

# Also check holdout
try:
    h_rows = list(csv.DictReader(open('results/novel_holdout/holdout_results.csv')))
    zs_h = [r for r in h_rows if r['method'] == 'zeroshot']
    fps_h = [r for r in zs_h if int(r['novel_false_positives']) > 0]
    print(f"\nHoldout ZS rows: {len(zs_h)}, with FP>0: {len(fps_h)}")
    for r in fps_h:
        fn = r['filename'][:55]
        print(f"  {fn}  trial={r['trial']}  FPs={r['novel_false_positives']}")
except FileNotFoundError:
    print("\nNo holdout CSV found")
