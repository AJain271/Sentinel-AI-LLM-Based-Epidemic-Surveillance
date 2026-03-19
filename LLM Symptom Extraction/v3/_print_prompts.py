"""Print the full Zero-Shot and Few-Shot prompts (with a dummy transcript)."""
from extract_zeroshot import build_prompt as zs_prompt
from extract_fewshot import build_prompt as fs_prompt

DUMMY = "(transcript would go here)"

print("=" * 80)
print("ZERO-SHOT PROMPT")
print("=" * 80)
sys_msg, user_msg = zs_prompt(DUMMY)
print(f"\n--- SYSTEM MESSAGE ---\n{sys_msg}")
print(f"\n--- USER MESSAGE ---\n{user_msg}")

print("\n\n")
print("=" * 80)
print("FEW-SHOT PROMPT")
print("=" * 80)
sys_msg2, user_msg2 = fs_prompt(DUMMY)
print(f"\n--- SYSTEM MESSAGE ---\n{sys_msg2}")
print(f"\n--- USER MESSAGE ---\n{user_msg2}")
