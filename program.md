# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `run_manual.py` — the wrapper you use to launch hand-edited runs. Do not call `train.py` directly.
   - `propose_and_run.py` — the optional Bayesian optimization (BO) tool. Read it, but do not modify it.
   - `fingerprint.py` — how architecture fingerprinting works (informational only — you don't call this directly).
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run. (`runs.csv` is a separate, append-only log maintained automatically by `run_manual.py`/`propose_and_run.py` — don't create or edit it yourself, and don't worry if it already has rows from a previous run tag; that history is shared on purpose.)
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation).

**Always launch training through a wrapper, never `uv run train.py` directly** — the wrappers log every run to `runs.csv`, which is what lets the BO tool warm-start. Two ways to launch a run:

- `uv run run_manual.py [--lr X] [--weight-decay Y]` — runs `train.py` as-is (or with an explicit override) and logs the result as an `agent` run. Use this for every hand-edited experiment, including the baseline.
- `uv run propose_and_run.py` — hands the current architecture's `lr`/`weight_decay` choice to a Bayesian optimization tool (Sobol until 4 same-architecture observations exist, then a GP-UCB surrogate), runs it, and logs the result as a `tool` run. See "The Bayesian optimization tool" below.

**What you CAN do:**

- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Call `propose_and_run.py` on any iteration instead of hand-tuning `lr`/`weight_decay` yourself. It's always available and entirely optional — use your judgment on when it's worth it.

**What you CANNOT do:**

- Modify `prepare.py`. Read-only — fixed evaluation, data loading, tokenizer, training constants.
- Modify `propose_and_run.py`, `fingerprint.py`, or `run_manual.py`. These implement the BO tool and its protected CLI contract; changing them defeats the point.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.
- Modify the evaluation harness. `evaluate_bpb` in `prepare.py` is the ground truth metric.
- Remove or repurpose the `--lr`/`--weight-decay` CLI flags in `train.py`. You can change everything else about `train.py`, but these two flags must keep overriding `MATRIX_LR`/`WEIGHT_DECAY` exactly as they do now — the BO tool checks that a run actually used what it requested and fails loudly if not.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline — `uv run run_manual.py` with `train.py` as-is.

### The Bayesian optimization tool

`propose_and_run.py` is a standing option, not a step you have to take. It only ever touches `lr` and `weight_decay` — it never edits `train.py` or makes architecture decisions. A few scenarios where it's a good fit:

- You've just landed an architecture change and want a reasonable `lr`/`weight_decay` for it before judging whether the change itself was worth it, but don't have a strong hand-tuned guess.
- You've been hand-tuning hyperparameters on the current architecture for a while and want a more systematic search than another manual guess.
- You're between architectural ideas and want to spend an iteration productively.

It's a weaker fit right after a structural edit you're not confident in yet — a bad architecture won't be saved by good hyperparameters, so it's usually cheaper to sanity-check the architecture with current defaults first.

The tool fingerprints the current architecture (`DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, `WINDOW_PATTERN`) and only warm-starts from prior runs — agent or tool, from this session or an earlier one — that share the exact same fingerprint. Change any of those four fields and the tool restarts cold for that architecture; that's expected, not a bug.

**If a tool run improves val_bpb:** treat it exactly like a hand-edited win. Update `MATRIX_LR` and `WEIGHT_DECAY` in `train.py` to the values `propose_and_run.py` used, then `git commit` — this is your new "keep" state, and future hand-edited runs will build on it. Skipping this means hand-edited runs quietly ignore what the tool found and revert to the old defaults.

**If it doesn't improve:** nothing to commit — `train.py` wasn't touched, so there's nothing to revert either.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
AUTORESEARCH_RESULT={"lr": 0.04, "val_bpb": 0.9979, "weight_decay": 0.2}
```

Both wrapper scripts mirror this output live and parse the `AUTORESEARCH_RESULT` line themselves to populate `runs.csv` — you don't need to parse it, but it's a useful sanity check on what values a run actually used. You can still extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit   val_bpb  memory_gb   status   description
```

1. git commit hash (short, 7 chars) — for a BO-tool win, this is the commit you just made updating the defaults.
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes.
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes.
4. status: `keep`, `discard`, or `crash`.
5. short text description of what this experiment tried. For BO-tool runs, note that it came from the tool, e.g. `"BO: lr=0.0412, weight_decay=0.183"`.

Example:

```
commit   val_bpb  memory_gb   status   description
a1b2c3d  0.997900 44.0  keep  baseline
b2c3d4e  0.993200 44.2  keep  increase LR to 0.04
c3d4e5f  1.005000 44.0  discard  switch to GeLU activation
d4e5f6g  0.991800 44.1  keep  BO: lr=0.0521, weight_decay=0.142
```

This is separate from `runs.csv`, which the wrapper scripts populate automatically on every run (agent or tool) with the raw numbers used for BO — you never write to `runs.csv` directly.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Decide your move for this iteration:
   - **Hand-edit**: tune `train.py` with an experimental idea by directly hacking the code, then `git commit`.
   - **BO tool**: skip editing and committing for now — `propose_and_run.py` runs `train.py` as it currently stands, just with different `lr`/`weight_decay`.
3. Run the experiment (redirect everything — do NOT use tee or let output flood your context):
   - Hand-edit: `uv run run_manual.py > run.log 2>&1` (add `--lr`/`--weight-decay` if you want an explicit override without editing the file).
   - BO tool: `uv run propose_and_run.py > run.log 2>&1`
4. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
5. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
6. Record the results in `results.tsv` (do not commit this file — leave it untracked by git).
7. Decide keep/discard:
   - **Hand-edit path**: if val_bpb improved, keep the commit and advance the branch. If equal or worse, `git reset` back to where you started.
   - **BO tool path**: if val_bpb improved, update `MATRIX_LR`/`WEIGHT_DECAY` in `train.py` to match what the tool used, then `git commit` — this becomes the new state to advance from. If equal or worse, there's nothing to commit or reset; just move on.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: if it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in `results.tsv`, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer, and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes, or use the BO tool for a systematic hyperparameter pass. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!