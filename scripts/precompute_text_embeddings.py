#!/usr/bin/env python3
"""
Pre-compute text embeddings using LLM.

This script extracts LLM hidden states for all text data and saves them
as HDF5 files. During training, these cached embeddings are loaded instead
of running LLM inference, which significantly speeds up training.

Usage:
    python scripts/precompute_text_embeddings.py --data-root data/mmcad
    python scripts/precompute_text_embeddings.py --data-root data/mmcad --model microsoft/Phi-4-mini-instruct
    python scripts/precompute_text_embeddings.py --data-root data/mmcad --batch-size 16 --output-dir embeddings/

Output:
    Creates HDF5 files with structure:
    - {split}_text_embeddings.h5
        - title_embeddings: [N, d_llm]
        - desc_embeddings: [N, max_desc_len, d_llm]
        - desc_masks: [N, max_desc_len]
        - sample_ids: [N] (string dataset)
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_text_data(data_root: Path, split: str):
    """Load text data for a split."""
    text_dir = data_root / "text"
    split_file = data_root / "splits" / f"{split}.txt"

    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        return []

    with open(split_file, "r") as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    samples = []
    for sample_id in sample_ids:
        text_file = text_dir / f"{sample_id}.json"
        if text_file.exists():
            with open(text_file, "r") as f:
                data = json.load(f)
            samples.append({
                "sample_id": sample_id,
                "title": data.get("title", ""),
                "description": data.get("description", ""),
            })
        else:
            # Use empty text if file not found
            samples.append({
                "sample_id": sample_id,
                "title": "",
                "description": "",
            })

    return samples


def create_batches(samples, batch_size):
    """Create batches from samples."""
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]


@torch.no_grad()
def extract_embeddings(
    model,
    tokenizer,
    samples,
    batch_size: int,
    max_title_len: int,
    max_desc_len: int,
    device: torch.device,
):
    """
    Extract LLM embeddings for all samples.

    Returns:
        title_embeddings: [N, d_llm]
        desc_embeddings: [N, max_desc_len, d_llm]
        desc_masks: [N, max_desc_len]
        sample_ids: [N]
    """
    model.eval()

    d_llm = model.config.hidden_size

    all_title_embeddings = []
    all_desc_embeddings = []
    all_desc_masks = []
    all_sample_ids = []

    batches = list(create_batches(samples, batch_size))

    for batch in tqdm(batches, desc="Extracting embeddings"):
        # Get titles and descriptions
        titles = [s["title"] or "CAD model" for s in batch]
        descriptions = [s["description"] or "A CAD model" for s in batch]
        sample_ids = [s["sample_id"] for s in batch]

        # Tokenize titles
        title_tokens = tokenizer(
            titles,
            padding="max_length",
            truncation=True,
            max_length=max_title_len,
            return_tensors="pt",
        )

        # Tokenize descriptions
        desc_tokens = tokenizer(
            descriptions,
            padding="max_length",
            truncation=True,
            max_length=max_desc_len,
            return_tensors="pt",
        )

        # Move to device
        title_input_ids = title_tokens["input_ids"].to(device)
        title_attention_mask = title_tokens["attention_mask"].to(device)
        desc_input_ids = desc_tokens["input_ids"].to(device)
        desc_attention_mask = desc_tokens["attention_mask"].to(device)

        # Extract title embeddings (last token hidden state)
        title_outputs = model(
            input_ids=title_input_ids,
            attention_mask=title_attention_mask,
        )
        title_hidden = title_outputs.last_hidden_state  # [B, L, d_llm]

        # Get last non-padding token
        seq_lengths = title_attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(title_hidden.size(0), device=device)
        title_last = title_hidden[batch_idx, seq_lengths.long()]  # [B, d_llm]

        # Extract description embeddings (all token hidden states)
        desc_outputs = model(
            input_ids=desc_input_ids,
            attention_mask=desc_attention_mask,
        )
        desc_hidden = desc_outputs.last_hidden_state  # [B, L_desc, d_llm]

        # Store results
        all_title_embeddings.append(title_last.cpu().float().numpy())
        all_desc_embeddings.append(desc_hidden.cpu().float().numpy())
        all_desc_masks.append(desc_attention_mask.cpu().numpy())
        all_sample_ids.extend(sample_ids)

    # Concatenate all batches
    title_embeddings = np.concatenate(all_title_embeddings, axis=0)
    desc_embeddings = np.concatenate(all_desc_embeddings, axis=0)
    desc_masks = np.concatenate(all_desc_masks, axis=0)

    return title_embeddings, desc_embeddings, desc_masks, all_sample_ids


def save_embeddings_hdf5(
    output_path: Path,
    title_embeddings: np.ndarray,
    desc_embeddings: np.ndarray,
    desc_masks: np.ndarray,
    sample_ids: list,
    model_name: str,
):
    """Save embeddings to HDF5 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Save embeddings
        f.create_dataset(
            "title_embeddings",
            data=title_embeddings,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "desc_embeddings",
            data=desc_embeddings,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "desc_masks",
            data=desc_masks,
            compression="gzip",
            compression_opts=4,
        )

        # Save sample IDs as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        sample_ids_ds = f.create_dataset("sample_ids", (len(sample_ids),), dtype=dt)
        for i, sid in enumerate(sample_ids):
            sample_ids_ds[i] = sid

        # Save metadata
        f.attrs["model_name"] = model_name
        f.attrs["d_llm"] = title_embeddings.shape[1]
        f.attrs["max_desc_len"] = desc_embeddings.shape[1]
        f.attrs["n_samples"] = len(sample_ids)

    print(f"Saved embeddings to {output_path}")
    print(f"  - title_embeddings: {title_embeddings.shape}")
    print(f"  - desc_embeddings: {desc_embeddings.shape}")
    print(f"  - desc_masks: {desc_masks.shape}")
    print(f"  - n_samples: {len(sample_ids)}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute text embeddings using LLM"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to MMCAD data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for embeddings (default: {data_root}/embeddings)",
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
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-title-len",
        type=int,
        default=64,
        help="Maximum title length",
    )
    parser.add_argument(
        "--max-desc-len",
        type=int,
        default=256,
        help="Maximum description length",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process",
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

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "embeddings"
    device = torch.device(args.device)

    print("=" * 60)
    print("Pre-computing Text Embeddings")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max title len: {args.max_title_len}")
    print(f"Max desc len: {args.max_desc_len}")
    print()

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.float16 if args.fp16 else torch.float32
    try:
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        )
    except Exception as e:
        print(f"Flash attention not available: {e}")
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    model = model.to(device)
    model.eval()

    print(f"Model loaded: {args.model}")
    print(f"Hidden dimension: {model.config.hidden_size}")
    print()

    # Process each split
    for split in args.splits:
        print(f"\n{'=' * 60}")
        print(f"Processing {split} split")
        print("=" * 60)

        # Load text data
        samples = load_text_data(data_root, split)
        if not samples:
            print(f"No samples found for {split}")
            continue

        print(f"Found {len(samples)} samples")

        # Extract embeddings
        title_emb, desc_emb, desc_masks, sample_ids = extract_embeddings(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            batch_size=args.batch_size,
            max_title_len=args.max_title_len,
            max_desc_len=args.max_desc_len,
            device=device,
        )

        # Save to HDF5
        output_path = output_dir / f"{split}_text_embeddings.h5"
        save_embeddings_hdf5(
            output_path=output_path,
            title_embeddings=title_emb,
            desc_embeddings=desc_emb,
            desc_masks=desc_masks,
            sample_ids=sample_ids,
            model_name=args.model,
        )

    print("\n" + "=" * 60)
    print("Pre-computation complete!")
    print("=" * 60)
    print(f"\nEmbeddings saved to: {output_dir}")
    print("\nTo use during training, update your config:")
    print("  data:")
    print(f"    text_embeddings_dir: {output_dir}")
    print("    use_cached_text_embeddings: true")


if __name__ == "__main__":
    main()
