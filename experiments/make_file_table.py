import hashlib

files = [
    ("phase0b_deep_v5", "experiments/phase0b_deep.py", "logs/phase0b_deep_v5.log"),
    ("phase0b_v5_patch_off", "experiments/phase1_reproduce.py", "logs/phase0b_v5_patch_off.log"),
    ("phase0b_deep_v4", "experiments/phase0b_deep.py", "logs/phase0b_deep_v4.log"),
    ("phase0b_v4_patch_off", "experiments/phase1_reproduce.py", "logs/phase0b_v4_patch_off.log"),
]

header = f"{'Tag':<22} | {'Script':<32} | {'Log':<30} | {'SHA256 (head8...tail8)':<22}"
print(header)
print("-" * len(header))
for tag, script, log in files:
    hasher = hashlib.sha256()
    with open(log, "rb") as f:
        text = f.read()
    hasher.update(text)
    h = hasher.hexdigest()
    h_short = f"{h[:8]}...{h[-8:]}"
    print(f"{tag:<22} | {script:<32} | {log:<30} | {h_short:<22}")
