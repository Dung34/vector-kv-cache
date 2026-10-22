import hashlib

files = [
    ("phase2_v5_old", "experiments/phase2_test.py", "logs/phase2_v5_old.log"),
    ("phase2_v4_old", "experiments/phase2_test.py", "logs/phase2_v4_old.log"),
    ("phase2_v5_fixed", "experiments/phase2_test.py", "logs/phase2_v5_fixed.log"),
    ("phase2_v4_fixed", "experiments/phase2_test.py", "logs/phase2_v4_fixed.log"),
    ("phase2_v5_mutation", "experiments/phase2_test.py", "logs/phase2_v5_mutation.log"),
]

header = f"{'Tag':<22} | {'Script':<28} | {'Log':<28} | {'SHA256 (head8...tail8)':<22}"
print(header)
print("-" * len(header))
for tag, script, log in files:
    hasher = hashlib.sha256()
    with open(log, "rb") as f:
        hasher.update(f.read())
    h = hasher.hexdigest()
    h_short = f"{h[:8]}...{h[-8:]}"
    print(f"{tag:<22} | {script:<28} | {log:<28} | {h_short:<22}")
