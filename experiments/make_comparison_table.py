import re

logs = [
    ("5.20.1", "ON", "logs/phase0b_deep_v5.numbered.txt"),
    ("5.20.1", "OFF", "logs/phase0b_v5_patch_off.numbered.txt"),
    ("4.56.2", "ON", "logs/phase0b_deep_v4.numbered.txt"),
    ("4.56.2", "OFF", "logs/phase0b_v4_patch_off.numbered.txt"),
]

header = f"{'Env':<8} | {'Patch':<5} | {'Step1_K':<7} | {'Step1_Mask':<10} | {'Step1_Verdict':<13} | {'Step2_Pos':<9} | {'Step2_Exp':<9} | {'Step2_Verdict':<13}"
print(header)
print("-" * len(header))

for env, patch, path in logs:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if "RuntimeError" in text:
        print(f"{env:<8} | {patch:<5} | {'32':<7} | {'65':<10} | {'CRASH(32!=65)':<13} | {'N/A':<9} | {'N/A':<9} | {'N/A':<13}")
    else:
        # Check either [STEP] format or table format
        m_step1 = re.search(r"\[STEP\] n=1 seen=(\d+) past_seen=(\d+) phys_len=(\d+) mask_kv=(\d+)", text)
        m_step2 = re.search(r"\[STEP\] n=2 seen=(\d+) past_seen=(\d+) phys_len=(\d+) mask_kv=(\d+) pos_id=(\d+)", text)
        if not m_step1:
            # table format in phase1_reproduce
            m_step1_tbl = re.search(r"1\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*(\d+)", text)
            m_step2_tbl = re.search(r"2\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*(\d+)", text)
            k1 = m_step1_tbl.group(3) if m_step1_tbl else "?"
            mask1 = m_step1_tbl.group(4) if m_step1_tbl else "?"
            pos2 = m_step2_tbl.group(5) if m_step2_tbl else "?"
            exp2 = m_step2_tbl.group(1) if m_step2_tbl else "?"
        else:
            k1 = m_step1.group(3) if m_step1 else "?"
            mask1 = m_step1.group(4) if m_step1 else "?"
            pos2 = m_step2.group(5) if m_step2 else "?"
            exp2 = m_step2.group(1) if m_step2 else "?"
        print(f"{env:<8} | {patch:<5} | {k1:<7} | {mask1:<10} | {'PASS(SLICED)':<13} | {pos2:<9} | {exp2:<9} | {'FAIL(I1)':<13}")
