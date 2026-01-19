#!/usr/bin/env python3
"""
Pre-compute text embeddings from CSV file using LLM.

This script reads text data from a CSV file and extracts LLM hidden states,
saving them as HDF5 files for training.

Features:
- Checkpointing every N batches (default 100) to allow resuming
- Resume from checkpoint with --resume flag
- Incremental saving to avoid memory issues
- Fast LZF compression for quick saves (full float32 precision)

Usage:
    python scripts/precompute_text_embeddings_csv.py --csv ../data/169k.csv --output-dir ../data/embeddings
    python scripts/precompute_text_embeddings_csv.py --csv ../data/169k.csv --batch-size 4 --checkpoint-every 50
    python scripts/precompute_text_embeddings_csv.py --csv ../data/169k.csv --resume  # Resume from checkpoint

Output:
    Creates HDF5 file with structure:
    - text_embeddings.h5
        - desc_embeddings: [N, max_desc_len, d_llm]
        - desc_masks: [N, max_desc_len]
        - uids: [N] (string dataset)
"""

import argparse
import json
import sys
import gc
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_csv_data(csv_path: Path):
    """Load text data from CSV file."""
    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ["uid", "title", "description"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Clean data
    df = df.dropna(subset=["uid"])
    df["uid"] = df["uid"].astype(str)
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

    samples = []
    for _, row in df.iterrows():
        samples.append({
            "uid": str(row["uid"]),
            "title": str(row["title"]),
            "description": str(row["description"]),
        })

    return samples


def create_batches(samples, batch_size):
    """Create batches from samples."""
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]


def save_checkpoint(checkpoint_path: Path, batch_idx: int, processed_samples: int):
    """Save checkpoint information."""
    checkpoint = {
        "batch_idx": batch_idx,
        "processed_samples": processed_samples,
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f)


def load_checkpoint(checkpoint_path: Path):
    """Load checkpoint information."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None


def append_to_hdf5(
    output_path: Path,
    desc_embeddings: np.ndarray,
    desc_masks: np.ndarray,
    uids: list,
    model_name: str,
    max_desc_len: int,
    d_llm: int,
    total_samples: int,
    is_first_write: bool = False,
):
    """Append embeddings to HDF5 file (or create if first write)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_first_write:
        # Create new file with resizable datasets
        # Use LZF compression - much faster than gzip with decent compression
        with h5py.File(output_path, "w") as f:
            # Create resizable datasets with small chunks for faster appends
            f.create_dataset(
                "desc_embeddings",
                shape=(0, max_desc_len, d_llm),
                maxshape=(total_samples, max_desc_len, d_llm),
                dtype=np.float32,
                chunks=(10, max_desc_len, d_llm),  # Small chunks for fast appends
                compression="lzf",  # Fast compression
            )
            f.create_dataset(
                "desc_masks",
                shape=(0, max_desc_len),
                maxshape=(total_samples, max_desc_len),
                dtype=np.int64,
                chunks=(100, max_desc_len),
                compression="lzf",
            )
            # UIDs as variable-length strings
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset(
                "uids",
                shape=(0,),
                maxshape=(total_samples,),
                dtype=dt,
            )
            # Metadata
            f.attrs["model_name"] = model_name
            f.attrs["d_llm"] = d_llm
            f.attrs["max_desc_len"] = max_desc_len
            f.attrs["total_samples"] = total_samples
            f.attrs["n_samples"] = 0

    # Append data
    with h5py.File(output_path, "a") as f:
        current_size = f["desc_embeddings"].shape[0]
        new_size = current_size + desc_embeddings.shape[0]

        # Resize datasets
        f["desc_embeddings"].resize(new_size, axis=0)
        f["desc_masks"].resize(new_size, axis=0)
        f["uids"].resize(new_size, axis=0)

        # Write new data
        f["desc_embeddings"][current_size:new_size] = desc_embeddings
        f["desc_masks"][current_size:new_size] = desc_masks
        for i, uid in enumerate(uids):
            f["uids"][current_size + i] = uid

        # Update count
        f.attrs["n_samples"] = new_size

        # Flush to disk
        f.flush()


@torch.no_grad()
def extract_embeddings_with_checkpoints(
    model,
    tokenizer,
    samples,
    batch_size: int,
    max_desc_len: int,
    device: torch.device,
    output_path: Path,
    checkpoint_path: Path,
    checkpoint_every: int = 100,
    start_batch: int = 0,
    use_title_prefix: bool = True,
):
    """
    Extract LLM embeddings with periodic checkpointing.
    """
    model.eval()
    d_llm = model.config.hidden_size
    total_samples = len(samples)

    batches = list(create_batches(samples, batch_size))
    total_batches = len(batches)

    # Buffers for accumulating before checkpoint
    buffer_embeddings = []
    buffer_masks = []
    buffer_uids = []

    is_first_write = (start_batch == 0)

    pbar = tqdm(
        enumerate(batches),
        total=total_batches,
        desc="Extracting embeddings",
        initial=start_batch
    )

    for batch_idx, batch in pbar:
        # Skip already processed batches
        if batch_idx < start_batch:
            continue

        # Combine title and description for richer context
        if use_title_prefix:
            texts = [
                f"{s['title']}. {s['description']}" if s['title'] else s['description']
                for s in batch
            ]
        else:
            texts = [s["description"] or "A CAD model" for s in batch]

        texts = [t if t.strip() else "A CAD model" for t in texts]
        uids = [s["uid"] for s in batch]

        # Tokenize descriptions
        desc_tokens = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_desc_len,
            return_tensors="pt",
        )

        # Move to device
        desc_input_ids = desc_tokens["input_ids"].to(device)
        desc_attention_mask = desc_tokens["attention_mask"].to(device)

        # Extract description embeddings (all token hidden states)
        desc_outputs = model(
            input_ids=desc_input_ids,
            attention_mask=desc_attention_mask,
        )
        desc_hidden = desc_outputs.last_hidden_state  # [B, L_desc, d_llm]

        # Add to buffer (move to CPU immediately to free GPU memory)
        buffer_embeddings.append(desc_hidden.cpu().float().numpy())
        buffer_masks.append(desc_attention_mask.cpu().numpy())
        buffer_uids.extend(uids)

        # Clean up GPU tensors
        del desc_input_ids, desc_attention_mask, desc_outputs, desc_hidden

        # Checkpoint every N batches
        if (batch_idx + 1) % checkpoint_every == 0 or (batch_idx + 1) == total_batches:
            # Concatenate buffer
            if buffer_embeddings:
                pbar.set_postfix({"status": "saving..."})

                emb_array = np.concatenate(buffer_embeddings, axis=0)
                mask_array = np.concatenate(buffer_masks, axis=0)

                # Append to HDF5
                append_to_hdf5(
                    output_path=output_path,
                    desc_embeddings=emb_array,
                    desc_masks=mask_array,
                    uids=buffer_uids,
                    model_name=str(model.config._name_or_path),
                    max_desc_len=max_desc_len,
                    d_llm=d_llm,
                    total_samples=total_samples,
                    is_first_write=is_first_write,
                )
                is_first_write = False

                # Save checkpoint
                processed = (batch_idx + 1) * batch_size
                save_checkpoint(checkpoint_path, batch_idx + 1, min(processed, total_samples))

                # Clear buffer and free memory
                del buffer_embeddings, buffer_masks, emb_array, mask_array
                buffer_embeddings = []
                buffer_masks = []
                buffer_uids = []

                # Force garbage collection and GPU cache clear
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Update progress bar
                pbar.set_postfix({"saved": f"{batch_idx + 1}/{total_batches}"})

    print(f"\nCompleted! Processed {total_batches} batches")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute text embeddings from CSV using LLM"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file with uid, title, description columns",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (reduce if OOM)",
    )
    parser.add_argument(
        "--max-desc-len",
        type=int,
        default=256,
        help="Maximum description length in tokens",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N batches (default: 100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="text_embeddings.h5",
        help="Output filename",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name
    checkpoint_path = output_dir / f"{args.output_name}.checkpoint.json"
    device = torch.device(args.device)

    print("=" * 60)
    print("Pre-computing Text Embeddings from CSV")
    print("=" * 60)
    print(f"CSV file: {csv_path}")
    print(f"Output: {output_path}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max desc len: {args.max_desc_len}")
    print(f"Checkpoint every: {args.checkpoint_every} batches")
    print()

    # Check for resume
    start_batch = 0
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            start_batch = checkpoint["batch_idx"]
            print(f"Resuming from batch {start_batch} ({checkpoint['processed_samples']} samples)")
        else:
            print("No checkpoint found, starting from beginning")
    else:
        # Clear old checkpoint and output if not resuming
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if output_path.exists() and start_batch == 0:
            output_path.unlink()
            print("Cleared previous output file")
    print()

    # Load CSV data
    print("Loading CSV data...")
    samples = load_csv_data(csv_path)
    print(f"Loaded {len(samples)} samples")
    print()

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.float16 if args.fp16 else torch.float32
    model = AutoModel.from_pretrained(
        args.model,
        dtype=dtype,  # Fixed: use 'dtype' instead of deprecated 'torch_dtype'
        trust_remote_code=True,
    )

    model = model.to(device)
    model.eval()

    print(f"Model loaded: {args.model}")
    print(f"Hidden dimension: {model.config.hidden_size}")
    print()

    # Extract embeddings with checkpointing
    extract_embeddings_with_checkpoints(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        batch_size=args.batch_size,
        max_desc_len=args.max_desc_len,
        device=device,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        start_batch=start_batch,
    )

    # Clean up checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Cleaned up checkpoint file")

    # Print final stats
    with h5py.File(output_path, "r") as f:
        print("\n" + "=" * 60)
        print("Pre-computation complete!")
        print("=" * 60)
        print(f"Output: {output_path}")
        print(f"  - desc_embeddings: {f['desc_embeddings'].shape}")
        print(f"  - desc_masks: {f['desc_masks'].shape}")
        print(f"  - n_samples: {f.attrs['n_samples']}")


if __name__ == "__main__":
    main()
