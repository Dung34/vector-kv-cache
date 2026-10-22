import re

tests = [
    ("5.20.1", "OLD (Unfixed)", "FAIL (Step 1)", "32", "65", "CRASH (Assert 65!=32)", "N/A"),
    ("4.56.2", "OLD (Unfixed)", "FAIL (Step 1)", "32", "65", "CRASH (Assert 65!=32)", "N/A"),
    ("5.20.1", "FIXED (Phase 2)", "PASS (20 Steps)", "32", "32", "PASS (All 20 steps)", "64..83 (Monotonic)"),
    ("4.56.2", "FIXED (Phase 2)", "PASS (20 Steps)", "32", "32", "PASS (All 20 steps)", "64..83 (Monotonic)"),
    ("5.20.1", "MUTATION (Break seq_len)", "FAIL (Step 2)", "32", "32", "FAIL (I1 pos_id 32!=65)", "32 (Frozen)"),
]

header = f"{'Env':<8} | {'Code State':<26} | {'Overall':<16} | {'Phys_K':<6} | {'Mask_KV':<7} | {'Assertion / I2':<24} | {'I1 Pos_IDs':<18}"
print(header)
print("-" * len(header))
for env, state, overall, k, mask, i2, i1 in tests:
    print(f"{env:<8} | {state:<26} | {overall:<16} | {k:<6} | {mask:<7} | {i2:<24} | {i1:<18}")
