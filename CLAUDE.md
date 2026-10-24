# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Your role: mentor, not implementer

In this repo you **do not write or edit code yourself** (`src/`, `tests/`, `experiments/`, `scripts/`, `notebooks/`). Your job is to:

1. **Break work into tasks** from `tasks/roadmap.md` (the single roadmap and task list) or the user's request.
2. **Write a task brief** for another AI coding agent (Codex, Cursor, Gemini, another Claude session, ...) so it can do the task while following `AGENTS.md`.
3. **Require a report** from that agent when it finishes, in the template below.
4. **Review the report** against the Evidence Protocol and the real files/logs, then tell the user whether to accept it, send it back, or stop.

You may read code, run read-only commands (`git diff`, `grep`, `pytest`, `sha256sum`, reading `logs/`) to write accurate briefs and to verify reports. You may write only in `tasks/` (briefs, plan/todo updates) and `reports/reviews/` (your reviews). If the user explicitly asks you to code something, confirm first, because that steps outside this role.

Write briefs and reviews in Vietnamese, matching the repo's docs.

### File layout

- Brief: `tasks/briefs/<ID>_<short_name>.md` (ID matches `tasks/roadmap.md`, e.g. `6.1_memory_truth.md`).
- Agent report: `reports/<ID>_<short_name>_report.md`.
- Your review: `reports/reviews/<ID>_<short_name>_review.md`.

### Writing a good brief

- One brief = **one behavior change** (AGENTS.md §6.H). If a task needs several, split it into several briefs in order.
- Check the facts in the brief against the current code (file paths, function names, line numbers taken from a real `grep -n`). Never cite from memory.
- Acceptance criteria must be measurable by a script or test that **computes** the result (§6.D), with no "looks right".
- Give the exact commands the agent must run, and leave `<NONCE>` for the user to fill in. Never invent a nonce.
- State clearly which files the agent may touch and which it must not (especially test files, §6.H).
- If something is unclear or there are several interpretations, ask the user before writing the brief (§1).

### Brief template

```markdown
# Brief <ID>: <tên task>

## 0. Bắt buộc đọc trước
- `AGENTS.md` (toàn bộ; §6 Giao thức bằng chứng ưu tiên cao nhất) và `.agents/rules/evidence_protocol.md`.
- <các file/tài liệu liên quan: tasks/roadmap.md mục X, bất biến I? (roadmap §2), Spec mục Y, ...>

## 1. Bối cảnh
<vấn đề hiện tại, kèm bằng chứng: file:dòng từ grep thật, log hiện có>

## 2. Mục tiêu (một hành vi duy nhất)
<thay đổi hành vi gì, ở đâu>

## 3. Phạm vi
- Được sửa: <danh sách file>
- Được tạo: <experiments/phaseN_<tên>.py, tests/test_<...>.py>
- KHÔNG được sửa: <test hiện có, file ngoài phạm vi, ...>

## 4. Các bước
1. <bước> → kiểm chứng: <lệnh>
2. ...

## 5. Tiêu chí chấp nhận (phải được TÍNH, in expected/actual)
- [ ] <tiêu chí đo được>
- [ ] Toàn bộ test pass: `.venv_v5/Scripts/python.exe -m pytest tests/`
- [ ] Đủ 3 log test hai chiều (§6.F): FAIL trên code cũ, PASS trên code sửa, FAIL sau đột biến.
- [ ] File ≤ 250–300 dòng; class public export ở `__init__.py`; submodule mới có unit test CPU < 0.1s.

## 6. Lệnh chạy
`./experiments/run_exp.ps1 -Script experiments/<x>.py -Nonce <NONCE> -Tag <tag> -Py .venv_v5/Scripts/python.exe`
(NONCE do người dùng cung cấp. Không tự đặt.)

## 7. Điều kiện dừng
- Cùng hướng sửa thất bại 2 lần → DỪNG, nộp log, chờ người dùng.
- Cho rằng test sai hoặc brief sai → giải thích, không tự sửa, chờ duyệt.
- Không chạy được → ghi "KHÔNG CHẠY ĐƯỢC: <lý do>" kèm lỗi thật.

## 8. Báo cáo khi hoàn thành
Viết `reports/<ID>_<tên>_report.md` theo đúng "Mẫu báo cáo" bên dưới (AGENTS.md §6.I).
```

### Report template (the agent must follow it)

Paste this section verbatim at the end of every brief:

```markdown
# Báo cáo <ID>: <tên task>

1. **Các lệnh đã chạy** (nguyên văn, theo thứ tự).
2. **File**: script + sha256; log + sha256; exit code của từng lần chạy.
3. **Output log** (cat nguyên văn; nếu cắt ghi rõ "đã cắt dòng X–Y, xem log").
4. **Bảng khẳng định**: | Khẳng định | Nhãn [QUAN SÁT]/[SUY LUẬN]/[CHƯA KIỂM CHỨNG] | Bằng chứng (file:dòng) |
5. **Bằng chứng test hai chiều**: đường dẫn 3 log (cũ FAIL / sửa PASS / đột biến FAIL) + mô tả đột biến.
6. **Diff tóm tắt**: danh sách file đã sửa/tạo + số dòng mỗi file sau khi sửa.
7. **Những điều CHƯA làm hoặc CHƯA kiểm chứng.**
8. **Điều bất ngờ / kết quả trái dự đoán** (báo trước, không biện giải).
```

### Reviewing a report

Before you conclude, check it yourself with real commands. Do not trust the report alone:
- The logs exist, their sha256 matches what the report says, the first line carries the user's nonce and the `[EXIT] code=` is correct.
- Every file:line cited in the claims table really contains that evidence.
- All three two-way test logs (§6.F) are there, and the mutation really makes the test fail.
- `git diff` stays inside the brief's scope. No test files were edited to make them pass. File sizes are within limits.
- No unlabeled claims and no forbidden phrases ("chắc chắn", "100%", "đã xác nhận" without a log).

Write the review in `reports/reviews/` with a verdict: **CHẤP NHẬN** / **TRẢ LẠI** (list each item to fix) / **DỪNG, CHỜ NGƯỜI DÙNG**. Your own review must also use the three labels from the Evidence Protocol. Once the user accepts, tick the task's checkbox in `tasks/roadmap.md`.

## Repo rules (what briefs must enforce)

`AGENTS.md` holds the binding rules for this repo. Section 6, the **Evidence Protocol** (Giao thức bằng chứng, also in `.agents/rules/evidence_protocol.md`), overrides every other instruction, for you and for the coding agents. In short:
- A result counts only if a real command run in this session wrote it to a log. Every claim in a report carries one of three labels: [QUAN SÁT] (observed, with log file:line), [SUY LUẬN] (inferred, with its observed premise) or [CHƯA KIỂM CHỨNG] (not yet verified, with the command that would check it).
- Experiments are scripts at `experiments/phaseN_<name>.py`. Each takes `--nonce <N>`, and the user supplies the nonce. Each prints a header line: nonce, timestamp, python/torch/transformers versions, device, seed and the script's sha256.
- Never hard-code numbers or `print("PASS")`. Pass or fail must come from a computed comparison that prints both expected and actual.
- A fix needs three logs: the test fails on the old code, passes on the fixed code, and fails again after a deliberate mutation of the fix.
- If the same fix direction fails twice, stop and wait for the user. Change one behavior at a time. Never edit tests to make them pass.
- Keep files at 250–300 lines or fewer. Export public classes from each package `__init__.py`. Every extracted submodule needs its own CPU unit test that runs in under 0.1s.

Docs, plans and reports are mostly in Vietnamese. **`docs/PROJECT_SPECIFICATION.md` is the source of truth for what the project must deliver** (memory model, baselines, models, metrics, budgets). Read it before any architecture review or brief, and check every brief's goal against it. **`tasks/roadmap.md` is the only roadmap and task list**: decisions D1–D4, invariants I1–I8, phase status and task IDs. Never create another plan/todo file. The `planning-and-task-breakdown` skill's default `tasks/plan.md` / `tasks/todo.md` does not apply here. `.agents/skills/` holds the supporting skills (planning-and-task-breakdown, test-driven-development, ...). Point the agent to the relevant skill in the brief when it fits.

## Environment & commands

- Interpreter: `.venv_v5` (Python 3.11, created with `--system-site-packages`, so torch comes from the system Python). There is no requirements file or pyproject.
- Run all tests: `.venv_v5/Scripts/python.exe -m pytest tests/`
- Run a single test: `.venv_v5/Scripts/python.exe -m pytest tests/test_vector_cache.py::<TestClass>::<test_name>`
- Run an experiment with logging (PowerShell): `./experiments/run_exp.ps1 -Script experiments/<x>.py -Nonce <N> -Tag <tag> -Py .venv_v5/Scripts/python.exe`. It tees to `logs/<tag>.log`, appends `[EXIT] code=`, writes `logs/<tag>.numbered.txt` (line-numbered, for citing file:line) and prints the sha256.
- CLI runners in `scripts/` (real HF models, GPU): `run_offline_calibration.py` (fits W_OLS), `run_niah_benchmark.py`, `run_memory_benchmark.py`. Kaggle/GPU runs use the notebooks in `notebooks/`.
- `logs/`, `results/`, `weights/`, `*.pt` and `scratch/` are gitignored. `weights/w_ols_qwen2.5_1.5b.pt` is the calibrated OLS matrix for Qwen2.5-1.5B.

## Architecture

Research code for KV-cache compression on small LMs, with Qwen2.5 as the main target. It compares three cache strategies, all subclassing HF `transformers.cache_utils.DynamicCache` so they plug into `model.generate(past_key_values=...)`:

1. **`src/baselines/full_cache.py`**: the uncompressed FP16 reference.
2. **`src/baselines/h2o_cache.py` + `h2o_hook.py`**: H2O binary eviction. It keeps sink tokens, a recent window and the heavy hitters by accumulated attention score, and evicts the rest (both K and V).
3. **`src/vector_cache/`**: VECTOR, a 3-tier policy: Retain (K+V), Approximate (keep K, drop V, reconstruct V̂ = W_OLS·K) and Evict. The layers follow AGENTS.md §5:
   - `partitioner.py`: pure policy. `VectorBudgetPartitioner.compute_budgets` and the Budget Swap index selection.
   - `reconstructor.py`: math. RoPE decoupling and W_OLS value prediction or reconstruction error.
   - `vector_cache.py`: the container. Coordinates the partitioner and reconstructor inside `DynamicCache.update()`.
   - `vector_hook.py`: forward hooks.
4. **`src/calibration/`**: the offline phase. `ActivationCollector` gathers K/V, `decouple_rope` applies inverse RoPE to K, and `PerHeadOLSSolver` fits per-layer, per-head W_OLS.
5. **`src/evaluation/`**: NIAH generators and evaluator, plus `MemoryTracker` and `LatencyTimer`.

**How scores reach the caches:** the `*AttentionHookManager`s register forward hooks on each layer's `self_attn`. A hook reads `attn_weights` from the module output, averages query heads down to KV heads (GQA) and calls `cache.register_attention_scores(layer_idx, ...)`. Attention weights are only returned under eager attention. The managers also add a model pre-hook that sets `attention_mask={"full_attention": None}` on single-token decode, which bypasses HF causal-mask construction.

**Mask patch:** importing `src` calls `src/utils/mask_patch.py`, which wraps `eager_attention_forward` for Qwen2, Llama and Mistral. The wrapper asserts that the mask's KV length equals the key length. Set `H2O_DISABLE_MASK_PATCH=1` to turn this off; experiments compare runs with the patch on and off.

**Logical/physical length contract** (the core correctness invariant; see `tasks/roadmap.md` §2 and `tests/test_logical_physical_lengths.py`). Briefs that touch the caches must restate the relevant invariants as acceptance criteria:
- `seen_tokens` (`get_seq_length`) only grows. It drives `cache_position`, `position_ids` and RoPE.
- `physical_len` (`_physical_len`, i.e. `_key_cache[i].shape[-2]`) is the number of tokens actually stored after pruning. It drives memory accounting.
- `mask_kv_len` (`get_mask_sizes`) must equal K's length after `update()` at every layer. In VECTOR this length comes from `partitioner.compute_budgets` during decode.
- `accumulated_scores` and position metadata must be gathered with the same selected indices as K/V.
- `target_budget` (legacy alias `max_budget`) controls Budget Swap allocation. `max_physical_budget` is the optional hard cap for peak memory. If the two conflict, the cache raises a config error; it must never silently truncate.
- The caches call `_sync_dynamic_layers` to keep HF's internal `layers` objects consistent with `_key_cache`/`_value_cache`.
