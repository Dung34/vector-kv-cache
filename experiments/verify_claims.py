import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Verify claims against numbered log files")
    parser.add_argument("claims_file", nargs="?", default="reports/phase0b_claims.tsv", help="Path to claims TSV file")
    args = parser.parse_args()

    if not os.path.exists(args.claims_file):
        print(f"[ERR] type=FileNotFoundError msg='Claims file not found: {args.claims_file}'")
        sys.exit(1)

    with open(args.claims_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\r\n") for line in f if line.strip() and not line.startswith("#")]

    claims_status = {}
    total = 0
    ok_count = 0
    mismatch_count = 0

    for line in lines:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        total += 1
        cid = parts[0].strip()
        label = parts[1].strip()
        target = parts[2].strip()

        if label == "QUAN_SAT":
            if len(parts) < 6:
                print(f"{cid} MISMATCH expected='6 columns' found='{len(parts)} columns'")
                mismatch_count += 1
                claims_status[cid] = False
                continue
            start_line = int(parts[3].strip())
            end_line = int(parts[4].strip())
            expected_str = parts[5].strip()

            log_file = target
            # Tự động hỗ trợ cả đường dẫn tới .numbered.txt và .log
            if not os.path.exists(log_file):
                # Thử đổi đuôi giữa .numbered.txt và .log
                if log_file.endswith(".numbered.txt") and os.path.exists(log_file.replace(".numbered.txt", ".log")):
                    pass
                elif log_file.endswith(".log") and os.path.exists(log_file.replace(".log", ".numbered.txt")):
                    log_file = log_file.replace(".log", ".numbered.txt")
                else:
                    print(f"{cid} MISMATCH file_not_found='{target}'")
                    mismatch_count += 1
                    claims_status[cid] = False
                    continue

            with open(log_file, "r", encoding="utf-8", errors="replace") as lf:
                file_lines = lf.readlines()

            if start_line < 1 or end_line > len(file_lines) or start_line > end_line:
                print(f"{cid} MISMATCH invalid_range={start_line}-{end_line} file_len={len(file_lines)}")
                mismatch_count += 1
                claims_status[cid] = False
                continue

            # Kiểm tra xem chuỗi có xuất hiện trong khoảng dòng không
            range_content = "".join(file_lines[start_line - 1 : end_line])
            if expected_str in range_content:
                print(f"{cid} OK")
                ok_count += 1
                claims_status[cid] = True
            else:
                print(f"{cid} MISMATCH expected=\"{expected_str}\" found_in_range=False")
                mismatch_count += 1
                claims_status[cid] = False

        elif label == "SUY_LUAN":
            # Target là danh sách các premise IDs, ví dụ: C1,C2
            premises = [p.strip() for p in target.split(",") if p.strip()]
            all_premises_ok = True
            for p in premises:
                if not claims_status.get(p, False):
                    all_premises_ok = False
                    break
            if all_premises_ok and len(premises) > 0:
                print(f"{cid} OK")
                ok_count += 1
                claims_status[cid] = True
            else:
                print(f"{cid} MISMATCH failed_premises='{target}'")
                mismatch_count += 1
                claims_status[cid] = False

        elif label == "CHUA_KIEM_CHUNG":
            if len(target) > 0:
                print(f"{cid} OK")
                ok_count += 1
                claims_status[cid] = True
            else:
                print(f"{cid} MISMATCH missing_verification_command")
                mismatch_count += 1
                claims_status[cid] = False
        else:
            print(f"{cid} MISMATCH unknown_label='{label}'")
            mismatch_count += 1
            claims_status[cid] = False

    print(f"[CLAIMS] total={total} ok={ok_count} mismatch={mismatch_count}")
    if mismatch_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
