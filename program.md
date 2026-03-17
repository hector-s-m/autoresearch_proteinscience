# autoresearch — antibody/nanobody design

This is an experiment to have the LLM do its own research on antibody sequence modeling.

## Domain

We train an **autoregressive GPT** over antibody amino acid sequences with:
- **Region annotations**: FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4 (IMGT-style)
- **Chain type conditioning**: `<HEAVY>`, `<LIGHT>`, `<NANOBODY>`
- **Species conditioning**: `<HUMAN>`, `<MOUSE>`, `<CAMEL>`, etc.

The model learns to generate antibody variable region sequences token-by-token, conditioned on species, chain type, and region markers. This follows the IgLM paradigm (autoregressive, GPT-style) but with the MuonAdamW optimizer and modern transformer architecture from autoresearch.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar17`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch_ab/processed/` contains `train_sequences.pt` and `val_sequences.pt`. If not, tell the human to run `uv run prepare.py` (or `uv run prepare.py --synthetic-only` for testing).
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_ppl` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_ppl.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_ppl gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_ppl:          3.456789
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

You can extract the key metric from the log file:

```
grep "^val_ppl:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_ppl	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_ppl achieved (e.g. 3.456789) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_ppl	memory_gb	status	description
a1b2c3d	3.456789	44.0	keep	baseline
b2c3d4e	3.234567	44.2	keep	increase LR to 0.06
c3d4e5f	3.567890	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar17`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_ppl:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_ppl improved (lower), you "advance" the branch, keeping the git commit
9. If val_ppl is equal or worse, you git reset back to where you started

## Domain-specific experiment ideas

Here are ideas specifically relevant to antibody sequence modeling:

**Architecture**:
- CDR-focused attention: weight attention more heavily within CDR regions
- Region-aware positional encoding: different positional embeddings per region type
- Separate CDR generation heads for CDR1/2/3 vs framework
- Larger model (deeper/wider) — antibody sequences are much shorter than text, so you can afford more parameters per token

**Training**:
- Focal loss for non-germline positions (AbLang2 insight): antibodies are biased toward germline sequences; upweight rare non-germline residues that drive antigen binding
- CDR3-weighted loss: CDR3 is the most diverse and therapeutically important region
- Curriculum learning: train on frameworks first, then CDRs
- Higher learning rates (sequences are short, vocab is small)

**Data representation**:
- Structure-conditioned generation: load dihedral angles from SAbDab structures, add a structure encoder (MLP over sin/cos of phi/psi angles) that adds to token embeddings
- Paired VH-VL training: concatenate heavy and light chains with `<SEP>` token
- Remove region markers and see if the model can implicitly learn region boundaries

**Regularization**:
- Dropout on CDR embeddings to improve generalization
- Masking augmentation (randomly mask some tokens, predict them — auxiliary MLM loss)

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
