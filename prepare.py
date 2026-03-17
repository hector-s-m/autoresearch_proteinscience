"""
One-time data preparation for antibody/nanobody autoresearch.
Downloads antibody sequences from OAS and structures from SAbDab.
Builds amino acid tokenizer and creates train/val splits.

Usage:
    python prepare.py                     # full prep (OAS + SAbDab + fallback)
    python prepare.py --num-oas-units 10  # limit OAS download
    python prepare.py --synthetic-only    # use synthetic data only (for testing)

Data and processed sequences are stored in ~/.cache/autoresearch_ab/.
"""

import os
import sys
import time
import math
import json
import gzip
import re
import hashlib
import random
import argparse
from multiprocessing import Pool
from urllib.parse import urljoin

import requests
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 512        # max tokens per packed row (antibodies ~120-300 AAs + markers)
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
EVAL_TOKENS = 10 * 65536  # ~655K tokens for validation

# ---------------------------------------------------------------------------
# Vocabulary (fixed, do not modify)
# ---------------------------------------------------------------------------

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids
REGION_TOKENS = ["<FR1>", "<CDR1>", "<FR2>", "<CDR2>", "<FR3>", "<CDR3>", "<FR4>"]
CHAIN_TOKENS = ["<HEAVY>", "<LIGHT>", "<NANOBODY>"]
SPECIES_TOKENS = ["<HUMAN>", "<MOUSE>", "<CAMEL>", "<RABBIT>", "<RHESUS>", "<RAT>"]
CONTROL_TOKENS = [
    "<BOS>", "<EOS>", "<PAD>", "<SEP>", "<MASK>",
    "<STRUCT>", "<NO_STRUCT>", "<UNK>",
]

ALL_TOKENS = AMINO_ACIDS + REGION_TOKENS + CHAIN_TOKENS + SPECIES_TOKENS + CONTROL_TOKENS
VOCAB_SIZE = len(ALL_TOKENS)  # 44

TOKEN_TO_ID = {tok: i for i, tok in enumerate(ALL_TOKENS)}
ID_TO_TOKEN = {i: tok for tok, i in TOKEN_TO_ID.items()}

# Convenience IDs
BOS_ID = TOKEN_TO_ID["<BOS>"]
EOS_ID = TOKEN_TO_ID["<EOS>"]
PAD_ID = TOKEN_TO_ID["<PAD>"]
SEP_ID = TOKEN_TO_ID["<SEP>"]
MASK_ID = TOKEN_TO_ID["<MASK>"]
STRUCT_ID = TOKEN_TO_ID["<STRUCT>"]
NO_STRUCT_ID = TOKEN_TO_ID["<NO_STRUCT>"]
UNK_ID = TOKEN_TO_ID["<UNK>"]

# Amino acid token IDs (for masking in evaluation)
AA_TOKEN_IDS = set(TOKEN_TO_ID[aa] for aa in AMINO_ACIDS)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_ab")
DATA_DIR = os.path.join(CACHE_DIR, "data")
STRUCTURES_DIR = os.path.join(CACHE_DIR, "structures")
PROCESSED_DIR = os.path.join(CACHE_DIR, "processed")

# OAS configuration
OAS_UNPAIRED_URL = "http://opig.stats.ox.ac.uk/webapps/oas/oas_unpaired/"
OAS_PAIRED_URL = "http://opig.stats.ox.ac.uk/webapps/oas/oas_paired/"

# SAbDab configuration
SABDAB_SUMMARY_URL = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"

# Validation fraction
VAL_FRACTION = 0.05

# OAS region column names
OAS_REGION_COLUMNS = {
    "FR1": "fwr1_aa",
    "CDR1": "cdr1_aa",
    "FR2": "fwr2_aa",
    "CDR2": "cdr2_aa",
    "FR3": "fwr3_aa",
    "CDR3": "cdr3_aa",
    "FR4": "fwr4_aa",
}

# Species mapping (lowercase key -> token)
SPECIES_MAP = {
    "human": "<HUMAN>", "homo sapiens": "<HUMAN>",
    "mouse": "<MOUSE>", "mus musculus": "<MOUSE>",
    "camel": "<CAMEL>", "camelus dromedarius": "<CAMEL>", "vicugna pacos": "<CAMEL>",
    "rabbit": "<RABBIT>", "oryctolagus cuniculus": "<RABBIT>",
    "rhesus": "<RHESUS>", "macaca mulatta": "<RHESUS>",
    "rat": "<RAT>", "rattus norvegicus": "<RAT>",
}

# Chain type mapping (lowercase key -> token)
CHAIN_MAP = {
    "heavy": "<HEAVY>", "igh": "<HEAVY>",
    "kappa": "<LIGHT>", "igk": "<LIGHT>",
    "lambda": "<LIGHT>", "igl": "<LIGHT>",
    "light": "<LIGHT>",
    "vhh": "<NANOBODY>", "nanobody": "<NANOBODY>",
}

# ---------------------------------------------------------------------------
# Data download: OAS
# ---------------------------------------------------------------------------

def _get_oas_data_urls(search_url, max_units=20):
    """Search OAS and return a list of data unit CSV.gz URLs."""
    session = requests.Session()

    try:
        # Step 1: GET the search page for CSRF token and cookies
        resp = session.get(search_url, timeout=30)
        resp.raise_for_status()

        # Extract CSRF token from the form
        csrf_match = re.search(
            r'name=["\']csrfmiddlewaretoken["\'].*?value=["\']([^"\']+)', resp.text
        )
        if not csrf_match:
            # Try cookie-based CSRF
            csrf_token = session.cookies.get("csrftoken", "")
            if not csrf_token:
                print("  Warning: Could not find CSRF token on OAS page")
                return []
        else:
            csrf_token = csrf_match.group(1)

        # Step 2: POST search with no filters (returns all data units)
        data = {"csrfmiddlewaretoken": csrf_token}
        headers = {"Referer": search_url}
        resp = session.post(
            search_url, data=data, headers=headers, timeout=120, allow_redirects=True
        )
        resp.raise_for_status()

        # Step 3: Look for bulk_download.sh link
        download_links = re.findall(
            r'href=["\']([^"\']*(?:bulk_download|download)[^"\']*)', resp.text
        )
        for link in download_links:
            if "bulk_download" in link.lower():
                full_url = link if link.startswith("http") else urljoin(search_url, link)
                try:
                    script_resp = session.get(full_url, timeout=60)
                    script_resp.raise_for_status()
                    # Parse wget URLs from the shell script
                    urls = re.findall(
                        r'(?:wget\s+["\']?)(https?://[^\s"\']+\.csv\.gz)',
                        script_resp.text,
                    )
                    if urls:
                        return urls[:max_units]
                except Exception:
                    pass

        # Fallback: look for CSV.gz URLs directly in the response page
        urls = re.findall(r'(https?://[^"\'>\s]*\.csv\.gz)', resp.text)
        return urls[:max_units]

    except Exception as e:
        print(f"  OAS search failed: {e}")
        return []


def _download_file(url, filepath, timeout=120):
    """Download a file with retry logic. Returns True on success."""
    if os.path.exists(filepath):
        return True

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            return True
        except (requests.RequestException, IOError) as e:
            print(f"    Attempt {attempt}/{max_attempts} failed: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def _parse_oas_csv(filepath, max_sequences=50000):
    """Parse a single OAS CSV.gz file, return list of antibody sequence dicts.

    OAS CSV format:
    - Row 0: JSON metadata encoded in column names (species, chain, etc.)
    - Row 1: actual column headers
    - Row 2+: data
    """
    sequences = []

    try:
        # Read metadata from first row
        meta_df = pd.read_csv(filepath, nrows=0, compression="gzip")
        try:
            metadata = json.loads(",".join(meta_df.columns))
            species = metadata.get("Species", "human").lower()
            chain = metadata.get("Chain", "Heavy").lower()
        except (json.JSONDecodeError, TypeError):
            species = "human"
            chain = "heavy"

        # Read actual data (header=1 skips the metadata row)
        df = pd.read_csv(
            filepath, header=1, compression="gzip",
            nrows=max_sequences, low_memory=False,
        )

        for _, row in df.iterrows():
            regions = {}
            for region_name, col_name in OAS_REGION_COLUMNS.items():
                if col_name in df.columns:
                    val = row.get(col_name)
                    if pd.notna(val):
                        val = str(val).strip().upper()
                        # Clean: keep only standard AAs and X
                        cleaned = "".join(
                            c for c in val if c in TOKEN_TO_ID or c == "X"
                        )
                        if cleaned and cleaned != "X":
                            regions[region_name] = cleaned

            # Need at least CDR3 and a few framework regions for a valid sequence
            if "CDR3" in regions and len(regions) >= 4:
                sequences.append({
                    "species": species,
                    "chain_type": chain,
                    "regions": regions,
                })

    except Exception as e:
        print(f"    Error parsing {filepath}: {e}")

    return sequences


def download_oas_data(num_units=20, max_sequences_per_unit=50000):
    """Download and parse OAS antibody sequences."""
    os.makedirs(DATA_DIR, exist_ok=True)

    all_sequences = []

    # Try unpaired first (more data)
    print("  OAS: Searching for unpaired data units...")
    urls = _get_oas_data_urls(OAS_UNPAIRED_URL, max_units=num_units)

    if not urls:
        # Try paired
        print("  OAS: Searching for paired data units...")
        urls = _get_oas_data_urls(OAS_PAIRED_URL, max_units=num_units)

    if not urls:
        print("  OAS: Could not find data URLs via search API")
        return []

    print(f"  OAS: Found {len(urls)} data unit URLs, downloading up to {num_units}...")

    for i, url in enumerate(urls[:num_units]):
        filename = os.path.basename(url)
        filepath = os.path.join(DATA_DIR, filename)
        print(f"  [{i+1}/{min(len(urls), num_units)}] {filename}...")

        if _download_file(url, filepath):
            seqs = _parse_oas_csv(filepath, max_sequences=max_sequences_per_unit)
            all_sequences.extend(seqs)
            print(f"    -> {len(seqs)} sequences (total: {len(all_sequences)})")

    return all_sequences


# ---------------------------------------------------------------------------
# Data download: SAbDab (supplementary / fallback)
# ---------------------------------------------------------------------------

def download_sabdab_data():
    """Download SAbDab summary TSV and extract antibody CDR sequences."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, "sabdab_summary.tsv")

    print("  SAbDab: Downloading summary...")
    try:
        if not os.path.exists(filepath):
            resp = requests.get(SABDAB_SUMMARY_URL, timeout=120)
            resp.raise_for_status()
            with open(filepath, "w") as f:
                f.write(resp.text)
            print(f"  SAbDab: Summary saved to {filepath}")

        df = pd.read_csv(filepath, sep="\t", low_memory=False)
        print(f"  SAbDab: {len(df)} entries loaded")

        sequences = []
        available_cols = {c.lower(): c for c in df.columns}

        # Discover CDR column names (various naming conventions)
        cdr_col_map = {}
        for col in df.columns:
            cu = col.upper().replace(" ", "").replace("_", "").replace("-", "")
            if "CDRH1" in cu:
                cdr_col_map["H_CDR1"] = col
            elif "CDRH2" in cu:
                cdr_col_map["H_CDR2"] = col
            elif "CDRH3" in cu:
                cdr_col_map["H_CDR3"] = col
            elif "CDRL1" in cu:
                cdr_col_map["L_CDR1"] = col
            elif "CDRL2" in cu:
                cdr_col_map["L_CDR2"] = col
            elif "CDRL3" in cu:
                cdr_col_map["L_CDR3"] = col

        print(f"  SAbDab: Found CDR columns: {list(cdr_col_map.keys())}")

        # Extract heavy chain CDR sequences
        for _, row in df.iterrows():
            # Heavy chain
            h_regions = {}
            for region_key in ["CDR1", "CDR2", "CDR3"]:
                col = cdr_col_map.get(f"H_{region_key}")
                if col and pd.notna(row.get(col)):
                    val = str(row[col]).strip().upper()
                    cleaned = "".join(c for c in val if c in TOKEN_TO_ID or c == "X")
                    if cleaned and cleaned != "X" and len(cleaned) > 1:
                        h_regions[region_key] = cleaned

            if len(h_regions) >= 2:
                # Determine species
                species = "human"
                for sp_col in ["organism", "species"]:
                    if sp_col in available_cols and pd.notna(row.get(available_cols[sp_col])):
                        org = str(row[available_cols[sp_col]]).lower()
                        for sp_key in SPECIES_MAP:
                            if sp_key in org:
                                species = sp_key
                                break
                        break

                sequences.append({
                    "species": species,
                    "chain_type": "heavy",
                    "regions": h_regions,
                })

            # Light chain
            l_regions = {}
            for region_key in ["CDR1", "CDR2", "CDR3"]:
                col = cdr_col_map.get(f"L_{region_key}")
                if col and pd.notna(row.get(col)):
                    val = str(row[col]).strip().upper()
                    cleaned = "".join(c for c in val if c in TOKEN_TO_ID or c == "X")
                    if cleaned and cleaned != "X" and len(cleaned) > 1:
                        l_regions[region_key] = cleaned

            if len(l_regions) >= 2:
                species = "human"
                for sp_col in ["organism", "species"]:
                    if sp_col in available_cols and pd.notna(row.get(available_cols[sp_col])):
                        org = str(row[available_cols[sp_col]]).lower()
                        for sp_key in SPECIES_MAP:
                            if sp_key in org:
                                species = sp_key
                                break
                        break

                sequences.append({
                    "species": species,
                    "chain_type": "light",
                    "regions": l_regions,
                })

        print(f"  SAbDab: Extracted {len(sequences)} sequences")
        return sequences

    except Exception as e:
        print(f"  SAbDab download failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Synthetic data fallback
# ---------------------------------------------------------------------------

def generate_synthetic_sequences(num_sequences=200000, seed=42):
    """Generate synthetic antibody sequences with realistic properties.

    Uses amino acid frequencies from real antibody repertoires and
    typical IMGT region length distributions. Last resort fallback
    that always works without any network access.
    """
    print(f"  Synthetic: Generating {num_sequences} sequences...")

    rng = random.Random(seed)

    # Approximate amino acid frequencies in antibody variable regions
    aa_weights = {
        "A": 6.8, "C": 2.2, "D": 4.7, "E": 5.3, "F": 3.4,
        "G": 8.2, "H": 2.1, "I": 4.0, "K": 4.6, "L": 8.2,
        "M": 1.8, "N": 3.6, "P": 5.2, "Q": 3.9, "R": 4.8,
        "S": 8.5, "T": 6.5, "V": 6.5, "W": 1.6, "Y": 3.8,
    }
    aas = list(aa_weights.keys())
    weights = [aa_weights[aa] for aa in aas]

    # Typical IMGT region lengths: (min, max)
    heavy_lengths = {
        "FR1": (25, 26), "CDR1": (6, 12), "FR2": (17, 17),
        "CDR2": (8, 10), "FR3": (32, 39), "CDR3": (3, 25), "FR4": (11, 11),
    }
    light_lengths = {
        "FR1": (23, 26), "CDR1": (6, 12), "FR2": (15, 17),
        "CDR2": (3, 3), "FR3": (32, 36), "CDR3": (7, 11), "FR4": (10, 11),
    }

    species_pool = ["human"] * 70 + ["mouse"] * 20 + ["camel"] * 10
    chain_pool = ["heavy"] * 50 + ["kappa"] * 25 + ["lambda"] * 20 + ["vhh"] * 5

    sequences = []
    for _ in range(num_sequences):
        chain = rng.choice(chain_pool)
        lengths = heavy_lengths if chain in ("heavy", "vhh") else light_lengths

        regions = {}
        for region, (min_len, max_len) in lengths.items():
            length = rng.randint(min_len, max_len)
            regions[region] = "".join(rng.choices(aas, weights=weights, k=length))

        species = rng.choice(species_pool)
        if chain == "vhh":
            species = "camel"

        sequences.append({
            "species": species,
            "chain_type": chain,
            "regions": regions,
        })

    return sequences


# ---------------------------------------------------------------------------
# Sequence tokenization
# ---------------------------------------------------------------------------

def tokenize_antibody(seq_dict):
    """Convert an antibody sequence dict to a list of token IDs.

    Output format:
    <BOS> <SPECIES> <CHAIN> <FR1> aa... <CDR1> aa... ... <FR4> aa... <EOS>
    """
    tokens = [BOS_ID]

    # Species token
    species = seq_dict["species"].lower()
    species_tok = SPECIES_MAP.get(species, "<HUMAN>")
    tokens.append(TOKEN_TO_ID[species_tok])

    # Chain type token
    chain = seq_dict["chain_type"].lower()
    chain_tok = CHAIN_MAP.get(chain, "<HEAVY>")
    tokens.append(TOKEN_TO_ID[chain_tok])

    # Region-annotated sequence
    for region_name in ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]:
        if region_name in seq_dict["regions"]:
            tokens.append(TOKEN_TO_ID[f"<{region_name}>"])
            for aa in seq_dict["regions"][region_name]:
                aa_upper = aa.upper()
                if aa_upper in TOKEN_TO_ID:
                    tokens.append(TOKEN_TO_ID[aa_upper])
                else:
                    tokens.append(UNK_ID)

    tokens.append(EOS_ID)
    return tokens


# ---------------------------------------------------------------------------
# Data processing and storage
# ---------------------------------------------------------------------------

def process_and_save(all_sequences, val_fraction=VAL_FRACTION):
    """Tokenize sequences, split into train/val, save to disk."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"Processing {len(all_sequences)} sequences...")

    # Tokenize all sequences
    tokenized = []
    for seq_dict in all_sequences:
        tokens = tokenize_antibody(seq_dict)
        if 10 <= len(tokens) <= MAX_SEQ_LEN:
            tokenized.append(tokens)

    print(f"  {len(tokenized)} sequences after filtering (10-{MAX_SEQ_LEN} tokens)")

    # Split by hash (deterministic, prevents CDR3 leakage)
    train_seqs = []
    val_seqs = []

    for tokens in tokenized:
        # Hash the full token sequence for deterministic splitting
        seq_hash = hashlib.md5(str(tokens).encode()).hexdigest()
        hash_val = int(seq_hash[:8], 16) / 0xFFFFFFFF
        if hash_val < val_fraction:
            val_seqs.append(tokens)
        else:
            train_seqs.append(tokens)

    print(f"  Train: {len(train_seqs)} sequences")
    print(f"  Val:   {len(val_seqs)} sequences")

    # Save
    torch.save(train_seqs, os.path.join(PROCESSED_DIR, "train_sequences.pt"))
    torch.save(val_seqs, os.path.join(PROCESSED_DIR, "val_sequences.pt"))

    # Stats
    train_tokens = sum(len(s) for s in train_seqs)
    val_tokens = sum(len(s) for s in val_seqs)
    total_tokens = train_tokens + val_tokens
    avg_len = total_tokens / len(tokenized) if tokenized else 0
    print(f"  Total tokens: {total_tokens:,} (avg {avg_len:.1f} per sequence)")
    print(f"  Saved to {PROCESSED_DIR}")

    return len(train_seqs), len(val_seqs)


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Simple amino acid tokenizer with special tokens.

    Unlike the BPE tokenizer in the original, this is a fixed character-level
    tokenizer over 20 amino acids plus region/chain/species/control tokens.
    No training needed -- the vocabulary is predefined.
    """

    def __init__(self):
        self.token_to_id = TOKEN_TO_ID
        self.id_to_token = ID_TO_TOKEN
        self._vocab_size = VOCAB_SIZE
        self._bos_token_id = BOS_ID

    @classmethod
    def from_directory(cls, *args, **kwargs):
        """Compatibility with original API. No directory needed."""
        return cls()

    def get_vocab_size(self):
        return self._vocab_size

    def get_bos_token_id(self):
        return self._bos_token_id

    def encode(self, text, **kwargs):
        """Encode amino acid string to token IDs."""
        if isinstance(text, str):
            return [self.token_to_id.get(c, UNK_ID) for c in text]
        elif isinstance(text, list):
            return [[self.token_to_id.get(c, UNK_ID) for c in t] for t in text]
        raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        """Decode token IDs to string."""
        return "".join(self.id_to_token.get(i, "?") for i in ids)


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def _load_sequences(split):
    """Load processed token sequences from disk."""
    filepath = os.path.join(PROCESSED_DIR, f"{split}_sequences.pt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Processed data not found at {filepath}. Run prepare.py first."
        )
    return torch.load(filepath, weights_only=False)


def _sequence_batches(split):
    """Infinite iterator yielding individual antibody token sequences."""
    sequences = _load_sequences(split)
    assert len(sequences) > 0, f"No {split} sequences found"
    epoch = 1
    while True:
        indices = list(range(len(sequences)))
        if split == "train":
            rng = random.Random(epoch * 31337)
            rng.shuffle(indices)
        for idx in indices:
            yield sequences[idx], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=500):
    """
    BOS-aligned dataloader with best-fit packing for antibody sequences.
    Every row packs multiple antibody sequences end-to-end (each starts with
    BOS and ends with EOS). Same interface as original: yields (inputs, targets, epoch).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    seq_iter = _sequence_batches(split)
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        seq, epoch = next(seq_iter)
        doc_buffer.append(seq)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest sequence that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(
                        doc, dtype=torch.long
                    )
                    pos += len(doc)
                else:
                    # No sequence fits -- crop shortest to fill remaining
                    shortest_idx = min(
                        range(len(doc_buffer)), key=lambda i: len(doc_buffer[i])
                    )
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(
                        doc[:remaining], dtype=torch.long
                    )
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE -- this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, tokenizer, batch_size):
    """
    Perplexity on validation antibody sequences.
    Computes exp(average cross-entropy per token) over all tokens.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = max(1, EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))
    total_loss = 0.0
    total_tokens = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        total_loss += loss_flat.sum().item()
        total_tokens += loss_flat.numel()
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# ---------------------------------------------------------------------------
# Structure utilities (for future use by the autoresearch agent)
# ---------------------------------------------------------------------------

def load_structure_data(split):
    """Load pre-computed dihedral angles. Returns None if not available.

    Future: The autoresearch agent can add structure-conditioned training
    by loading dihedral angles from SAbDab PDB files processed here.
    """
    filepath = os.path.join(PROCESSED_DIR, f"{split}_structures.pt")
    if os.path.exists(filepath):
        return torch.load(filepath, weights_only=False)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare antibody data for autoresearch"
    )
    parser.add_argument(
        "--num-oas-units", type=int, default=20,
        help="Number of OAS data units to download (default: 20)",
    )
    parser.add_argument(
        "--max-seq-per-unit", type=int, default=50000,
        help="Max sequences per OAS data unit (default: 50000)",
    )
    parser.add_argument(
        "--synthetic-only", action="store_true",
        help="Use synthetic data only (for testing, no network needed)",
    )
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Tokens: {ALL_TOKENS}")
    print()

    # Check if already processed
    train_path = os.path.join(PROCESSED_DIR, "train_sequences.pt")
    val_path = os.path.join(PROCESSED_DIR, "val_sequences.pt")
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_seqs = torch.load(train_path, weights_only=False)
        val_seqs = torch.load(val_path, weights_only=False)
        print(f"Data already processed: {len(train_seqs)} train, {len(val_seqs)} val")
        print(f"To re-process, delete: rm -rf {PROCESSED_DIR}")
        sys.exit(0)

    all_sequences = []

    if args.synthetic_only:
        print("Using synthetic data only (--synthetic-only flag)")
        all_sequences = generate_synthetic_sequences(500000)
    else:
        # Step 1: Try OAS
        print("Step 1: Downloading OAS antibody sequences...")
        oas_sequences = download_oas_data(
            num_units=args.num_oas_units,
            max_sequences_per_unit=args.max_seq_per_unit,
        )
        all_sequences.extend(oas_sequences)
        print(f"  Total from OAS: {len(oas_sequences)}")
        print()

        # Step 2: Try SAbDab
        print("Step 2: Downloading SAbDab sequences...")
        sabdab_sequences = download_sabdab_data()
        all_sequences.extend(sabdab_sequences)
        print(f"  Total from SAbDab: {len(sabdab_sequences)}")
        print()

        # Step 3: Fallback to synthetic if insufficient real data
        if len(all_sequences) < 10000:
            print("Step 3: Insufficient real data, generating synthetic fallback...")
            needed = max(200000, 100000 - len(all_sequences))
            synthetic = generate_synthetic_sequences(needed)
            all_sequences.extend(synthetic)
            print(f"  Generated {len(synthetic)} synthetic sequences")
            print()

    print(f"Total sequences: {len(all_sequences)}")
    print()

    # Process and save
    n_train, n_val = process_and_save(all_sequences)

    print()
    print("Done! Ready to train with: uv run train.py")
    print(f"  Train: {n_train} sequences")
    print(f"  Val:   {n_val} sequences")
