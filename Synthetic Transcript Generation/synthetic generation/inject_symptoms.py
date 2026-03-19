"""
Inject novel / emerging symptoms into an existing ground-truth JSON file.

Usage:
    # Add a single symptom
    python inject_symptoms.py --symptom "Purple discoloration of toes" --category conditions

    # Add with a custom code
    python inject_symptoms.py --symptom "Yellow discoloration of fingers" --category conditions --code NOVEL-001

    # Add multiple at once
    python inject_symptoms.py --symptom "Purple discoloration of toes" --symptom "Yellow discoloration of fingers"

    # Specify input/output paths
    python inject_symptoms.py --input ground_truth.json --output modified_ground_truth.json --symptom "Purple toes"

    # Reset: remove all injected symptoms
    python inject_symptoms.py --reset

    # List currently injected symptoms
    python inject_symptoms.py --list
"""

import json
import os
import sys
import argparse
from datetime import datetime

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
_modified = os.path.join(DEFAULT_DIR, "modified_ground_truth.json")
_base = os.path.join(DEFAULT_DIR, "ground_truth.json")
DEFAULT_INPUT = _modified if os.path.exists(_modified) else _base
DEFAULT_OUTPUT = os.path.join(DEFAULT_DIR, "modified_ground_truth.json")


def load_ground_truth(path: str) -> dict:
    """Load the ground-truth JSON, or exit on error."""
    if not os.path.exists(path):
        print(f"ERROR: Ground-truth file not found: {path}")
        print("  Run ccda_to_ground_truth.py first to generate it.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ground_truth(gt: dict, path: str):
    """Save modified ground truth to disk."""
    gt["last_modified"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {path}")


def inject_symptom(gt: dict, description: str, category: str = "conditions",
                   code: str = "", severity: str = ""):
    """Add a novel symptom to the ground truth."""
    entry = {
        "description": description,
        "code_system": "novel/emerging",
        "code": code or f"NOVEL-{len(gt.get('injected_symptoms', [])) + 1:03d}",
        "source": "injected",
        "injected_at": datetime.now().isoformat(),
    }
    if severity:
        entry["severity"] = severity

    # Add to the injected_symptoms registry
    if "injected_symptoms" not in gt:
        gt["injected_symptoms"] = []
    gt["injected_symptoms"].append({
        "description": description,
        "category": category,
        "code": entry["code"],
        "injected_at": entry["injected_at"],
    })

    # Also add to the relevant clinical section so it feeds into transcript generation
    if category not in gt:
        gt[category] = []
    gt[category].append(entry)

    print(f"  ✓ Injected: \"{description}\" → {category} (code: {entry['code']})")


def reset_injected(gt: dict):
    """Remove all injected symptoms from every section."""
    removed = 0
    for key in list(gt.keys()):
        if isinstance(gt[key], list):
            original_len = len(gt[key])
            gt[key] = [item for item in gt[key]
                       if not (isinstance(item, dict) and item.get("source") == "injected")]
            removed += original_len - len(gt[key])
    gt["injected_symptoms"] = []
    print(f"  ✓ Removed {removed} injected entries across all sections.")


def list_injected(gt: dict):
    """Print all currently injected symptoms."""
    injected = gt.get("injected_symptoms", [])
    if not injected:
        print("  No injected symptoms found.")
        return
    print(f"  {len(injected)} injected symptom(s):")
    for i, item in enumerate(injected, 1):
        print(f"    {i}. [{item.get('category', '?')}] {item['description']} "
              f"(code: {item.get('code', '?')}, at: {item.get('injected_at', '?')})")


def main():
    parser = argparse.ArgumentParser(
        description="Inject novel symptoms into ground-truth JSON for LLM testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inject_symptoms.py --symptom "Purple discoloration of toes"
  python inject_symptoms.py --symptom "Yellow fingers" --code NOVEL-COVID-TOES
  python inject_symptoms.py --list
  python inject_symptoms.py --reset
        """,
    )
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT,
                        help="Path to ground_truth.json (default: %(default)s)")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                        help="Path for output JSON (default: %(default)s)")
    parser.add_argument("--symptom", "-s", action="append", default=[],
                        help="Symptom description to inject (can repeat)")
    parser.add_argument("--category", "-c", default="conditions",
                        help="Clinical section to inject into (default: conditions)")
    parser.add_argument("--code", default="",
                        help="Custom code for the symptom")
    parser.add_argument("--severity", default="",
                        help="Optional severity level")
    parser.add_argument("--reset", action="store_true",
                        help="Remove all injected symptoms")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List currently injected symptoms")

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Symptom Injection Tool")
    print(f"{'='*60}")

    gt = load_ground_truth(args.input)
    print(f"  Loaded: {args.input}")
    print(f"  Patient: {gt.get('demographics', {}).get('name', '?')}")

    if args.list:
        list_injected(gt)
        return

    if args.reset:
        reset_injected(gt)
        save_ground_truth(gt, args.output)
        return

    if not args.symptom:
        print("\n  No symptoms specified. Use --symptom \"description\" to inject.")
        print("  Use --help for full usage.")
        list_injected(gt)
        return

    print()
    for symptom in args.symptom:
        inject_symptom(gt, symptom, args.category, args.code, args.severity)

    print()
    save_ground_truth(gt, args.output)

    # Summary
    print(f"\n  Total injected symptoms: {len(gt.get('injected_symptoms', []))}")
    print(f"  ✓ Done!")


if __name__ == "__main__":
    main()
