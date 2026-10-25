#!/usr/bin/env bash

if [[ $# -lt 3 ]]; then
    printf 'usage: bash experiments/run_exp.sh <SCRIPT> <NONCE> <TAG> [PY]\n' >&2
    exit 2
fi

script=$1
nonce=$2
tag=$3
py=${4:-python}
mkdir -p logs
log_path="logs/${tag}.log"
numbered_path="logs/${tag}.numbered.txt"

PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "$py" -u "$script" --nonce "$nonce" 2>&1 | tee "$log_path"
exit_code=${PIPESTATUS[0]}
printf '[EXIT] code=%s\n' "$exit_code" >> "$log_path"
sed -i 's/\r$//' "$log_path"

line_number=1
while IFS= read -r line || [[ -n "$line" ]]; do
    printf '%4d| %s\n' "$line_number" "$line"
    line_number=$((line_number + 1))
done < "$log_path" > "$numbered_path"

log_hash=$(sha256sum "$log_path" | awk '{print $1}')
printf '[HASH] %s\n' "$log_hash"
exit "$exit_code"
