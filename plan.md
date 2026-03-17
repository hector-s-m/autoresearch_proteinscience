# Plan: Antibody/Nanobody Autoresearch — Sequence + Structure-Guided

## Overview

Pivot the autoresearch framework from general text LLM pretraining to **antibody/nanobody design** with two complementary approaches:

1. **Sequence-guided**: Autoregressive language model over antibody amino acid sequences (inspired by IgLM's GPT approach + AbLang2's germline-aware training)
2. **Structure-guided**: Inverse folding — condition sequence generation on backbone dihedral angles (inspired by IGLOO's VQ tokenization of phi/psi angles)

Both share the same transformer backbone. Structure conditioning is additive, so the model can do unconditional generation (sequence-only) or structure-conditioned generation.

---

## Design Inspirations

### From AbLang/AbLang2 (OPIG, Charlotte Deane)
- **Training data**: OAS (Observed Antibody Space) — 35.6M unpaired + 1.26M paired sequences
- **Focal loss** for non-germline residues: standard cross-entropy over-predicts germline residues. AbLang2 uses focal loss to upweight rare non-germline positions, which are the therapeutically interesting mutations
- **Paired VH|VL support**: handle both unpaired and paired antibody sequences
- AbLang2 architecture: ESM-2 style, 12 layers, 480-dim embeddings (encoder/MLM)

### From IGLOO (Ada Fang, Prescient Design)
- **Structure as dihedral tokens**: encode backbone (phi, psi) angles via VQ-VAE into discrete tokens from a learned codebook (8192 tokens, 128-dim embeddings)
- **Two-phase training**: pretrain on SAbDab + predicted structures, fine-tune on SAbDab only
- **IgLooLM/IgLooALM**: integrate loop-level structure tokens with residue-level sequence tokens in a single language model (fine-tuned from IgBert)
- CDR H3 redesigns achieve <1Å RMSD with sequence identity of 0.27 — diverse sequences, conserved structure

### From IgLM (Graylab, Johns Hopkins)
- **Autoregressive/GPT-style** (closest to our decoder-only setup)
- **Chain type conditioning**: `[HEAVY]`, `[LIGHT]`, `[CAMEL]` tokens
- **Species conditioning**: `[HUMAN]`, `[MOUSE]`, etc.
- **Infilling**: redesign specific CDR regions while keeping framework intact
- This is the most architecturally aligned reference for our approach

---

## Architecture

### Tokenization (inspired by IgLM + AbLang2)
- **20 amino acid tokens** (standard AAs: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
- **1 unknown token**: X
- **Region tokens** (from IMGT/AHO numbering): `<FR1>`, `<CDR1>`, `<FR2>`, `<CDR2>`, `<FR3>`, `<CDR3>`, `<FR4>`
- **Chain type tokens** (IgLM-style): `<HEAVY>`, `<LIGHT>`, `<NANOBODY>`
- **Species tokens** (IgLM-style): `<HUMAN>`, `<MOUSE>`, `<CAMEL>`, `<RABBIT>`, `<RHESUS>`, `<RAT>`
- **Special tokens**: `<BOS>`, `<EOS>`, `<PAD>`, `<SEP>` (for paired VH-VL), `<MASK>` (for optional MLM pretraining)
- **Structure tokens**: `<STRUCT>` (marks structure-conditioned mode), `<NO_STRUCT>` (sequence-only mode)
- Total vocab: ~42 tokens

### Data Sources
- **Sequences**: OAS (Observed Antibody Space) — millions of antibody sequences with region annotations and species labels
- **Structures**: SAbDab — ~7000+ antibody crystal structures from PDB
- **Nanobodies**: Filter OAS for VHH/camelid sequences + nanobody entries in SAbDab

### Sequence Model (autoregressive, IgLM-inspired)
- GPT/decoder-only transformer (same as current `train.py` architecture)
- Input format: `<HUMAN> <HEAVY> <FR1> E V Q L ... <CDR1> G Y T F ... <CDR3> A R D ... <FR4> W G Q ... <EOS>`
- Paired: `<HUMAN> <HEAVY> ... <SEP> <LIGHT> ... <EOS>`
- Nanobody: `<CAMEL> <NANOBODY> <FR1> Q V Q L ... <EOS>`
- Autoregressive next-token prediction (cross-entropy loss)
- **Optional focal loss** (AbLang2-inspired): upweight non-germline positions during training

### Structure Encoder (IGLOO-inspired dihedral approach)
Two options for the autoresearch agent to explore:

**Option A — Dihedral angle binning (simple)**:
- Discretize backbone (phi, psi) into bins (e.g., 36 bins of 10° each per angle → 36×36 = 1296 structure tokens)
- Each residue gets a (phi_bin, psi_bin) pair encoded as a single discrete token
- Structure tokens are embedded and added to sequence embeddings
- Pro: simple, no additional encoder needed. Con: coarse, loses fine-grained geometry

**Option B — Continuous structure encoder (IGLOO-style)**:
- Input: backbone Cα coordinates per residue → compute (phi, psi, omega) dihedral angles
- Small MLP/1D-conv encoder produces per-residue structural embeddings
- Add structural embeddings to sequence token embeddings before transformer
- Pro: retains full geometric information. Con: more parameters, harder to optimize in 5min

**Conditioning toggle**:
- Prefix `<STRUCT>` token when structure is provided, `<NO_STRUCT>` when not
- During training, randomly mask out structure info (dropout) so model learns both modes
- At inference: structure-conditioned (inverse folding) or unconditional (sequence generation)

### Evaluation Metrics
- **val_ppl**: next-token perplexity on held-out antibody sequences (primary metric, replaces val_bpb)
- **aa_recovery**: for structure-conditioned mode — % of native residues recovered when given backbone
- **cdr_recovery**: aa_recovery restricted to CDR regions (harder, more therapeutically relevant)

---

## Files to Modify/Create

### 1. `prepare.py` — Complete rewrite
- Download OAS data (bulk download via OAS API — paired and unpaired human, mouse, camel sequences)
- Download SAbDab structures (PDB files via bulk download)
- Parse antibody sequences with region annotations (OAS provides IMGT-numbered sequences)
- Extract backbone dihedral angles from PDB files (using BioPython)
- Build amino acid tokenizer (simple lookup table — ~42 tokens, no BPE)
- Create train/val splits (split by clonotype hash to avoid leakage)
- Dataloader: yield (sequence_tokens, structure_dihedrals_or_None)
  - Mixed batches: ~90% sequence-only (from OAS), ~10% structure-conditioned (from SAbDab)
- Evaluation functions:
  - `evaluate_perplexity()`: standard autoregressive perplexity on held-out sequences
  - `evaluate_recovery()`: aa_recovery and cdr_recovery on held-out structures

### 2. `train.py` — Major rewrite
- Replace text vocab/embedding for amino acid tokens (~42 vocab)
- Shorter sequence length (antibodies are ~120-150 residues per chain, max ~350 for paired)
- Add optional structure encoder (simple dihedral embedding or MLP)
- Training: mixed sequence-only + structure-conditioned batches
- Keep the same optimizer (MuonAdamW) and 5-minute training budget framework
- Keep Flash Attention, rotary embeddings, etc. — just adapt for smaller sequences

### 3. `program.md` — Update
- Antibody-specific experimentation guidelines
- New metrics: val_ppl (primary), aa_recovery, cdr_recovery
- Updated results.tsv format
- Domain-specific experiment ideas for the agent to explore

### 4. `pyproject.toml` — Add dependencies
- `biopython` — PDB file parsing and dihedral angle computation

---

## Data Pipeline Detail

### OAS Download
- OAS provides a bulk download API for antibody sequences
- Each entry includes: full sequence, species, chain type, IMGT-numbered residues by region
- We download: human (VH, VL, VHH), mouse (VH, VL), camel (VHH)
- Expected: ~10-50M sequences depending on species/chain filters
- Stored as parquet files in `~/.cache/autoresearch_ab/data/`

### SAbDab Download
- SAbDab provides all antibody PDB structures with metadata
- Download summary file + PDB files for all entries
- Parse backbone atoms (N, CA, C) → compute (phi, psi) dihedral angles per residue
- ~7000 structures, aligned to IMGT numbering
- Stored in `~/.cache/autoresearch_ab/structures/`

### Sequence Format
Each training example is a flat token sequence:
```
<HUMAN> <HEAVY> <FR1> E V Q L V E S G G <CDR1> G Y T F T S Y <FR2> W V R Q A P <CDR2> I N P S G G <FR3> R F T I S ... <CDR3> A R D Y Y G ... <FR4> W G Q G T L V T V S S <EOS>
```

### Structure Format
For structure-conditioned examples:
- Dihedral angles: (N_residues, 2) tensor of (phi, psi) in radians
- Aligned 1:1 with amino acid tokens (not region/chain markers)
- Encoded as continuous features or discretized into bins
- When no structure: zeros / special no-structure embedding

---

## Experiment Loop Changes

The autoresearch loop stays the same conceptually:
1. Modify `train.py`
2. Run training for 5 minutes
3. Evaluate on held-out antibodies
4. Keep if val_ppl improves, discard if not

Key metrics in results.tsv:
```
commit	val_ppl	aa_recovery	cdr_recovery	memory_gb	status	description
```

Interesting experiment directions for the agent:
- Focal loss weighting for non-germline positions
- Different structure encoding strategies (bins vs continuous vs VQ)
- CDR-focused attention masking
- Separate CDR generation heads
- Nanobody-specific fine-tuning
- Chain-type embedding strategies
- Paired vs unpaired training curriculum

---

## Implementation Order

1. Update `pyproject.toml` with biopython dependency
2. Rewrite `prepare.py`: OAS/SAbDab data download, amino acid tokenizer, dataloader, evaluation
3. Rewrite `train.py`: antibody GPT with structure encoder
4. Update `program.md` for antibody-specific experimentation
5. Test end-to-end: download data → train → evaluate
6. Commit and push
