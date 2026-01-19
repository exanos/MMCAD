#!/usr/bin/env python3
"""
Analyze text token lengths in the dataset.

Samples 1000 items and computes token length statistics to validate
the max_desc_len parameter.

Usage:
    python scripts/analyze_text_lengths.py --csv ../data/169k.csv
    python scripts/analyze_text_lengths.py --csv ../data/169k.csv --num-samples 2000
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Analyze text token lengths")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="microsoft/Phi-4-mini-instruct", help="Tokenizer model")
    parser.add_argument("--save-uids", type=str, default=None, help="Save sampled UIDs to file")
    args = parser.parse_args()

    print("=" * 60)
    print("TEXT LENGTH ANALYSIS")
    print("=" * 60)
    print(f"CSV: {args.csv}")
    print(f"Samples: {args.num_samples}")
    print(f"Model: {args.model}")
    print()

    # Load CSV
    print("Loading CSV...")
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["uid"])
    print(f"Total samples: {len(df)}")

    # Sample
    sample_df = df.sample(n=args.num_samples, random_state=args.seed)
    print(f"Sampled: {len(sample_df)}")

    # Save UIDs if requested
    if args.save_uids:
        sample_df["uid"].to_csv(args.save_uids, index=False, header=True)
        print(f"Saved UIDs to: {args.save_uids}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"Tokenizer loaded: {args.model}")

    # Analyze
    print("\nAnalyzing token lengths...")
    title_lens = []
    desc_lens = []
    combined_lens = []

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        desc = str(row["description"]) if pd.notna(row["description"]) else ""
        combined = f"{title}. {desc}" if title else desc

        title_tokens = tokenizer.encode(title, add_special_tokens=False)
        desc_tokens = tokenizer.encode(desc, add_special_tokens=False)
        combined_tokens = tokenizer.encode(combined, add_special_tokens=False)

        title_lens.append(len(title_tokens))
        desc_lens.append(len(desc_tokens))
        combined_lens.append(len(combined_tokens))

    title_lens = np.array(title_lens)
    desc_lens = np.array(desc_lens)
    combined_lens = np.array(combined_lens)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nTitle token lengths:")
    print(f"  Min: {title_lens.min()}, Max: {title_lens.max()}, Mean: {title_lens.mean():.1f}")
    print(f"  Percentiles: 50%={np.percentile(title_lens, 50):.0f}, "
          f"90%={np.percentile(title_lens, 90):.0f}, "
          f"95%={np.percentile(title_lens, 95):.0f}, "
          f"99%={np.percentile(title_lens, 99):.0f}")

    print(f"\nDescription token lengths:")
    print(f"  Min: {desc_lens.min()}, Max: {desc_lens.max()}, Mean: {desc_lens.mean():.1f}")
    print(f"  Percentiles: 50%={np.percentile(desc_lens, 50):.0f}, "
          f"90%={np.percentile(desc_lens, 90):.0f}, "
          f"95%={np.percentile(desc_lens, 95):.0f}, "
          f"99%={np.percentile(desc_lens, 99):.0f}")

    print(f"\nCombined (title + desc) token lengths:")
    print(f"  Min: {combined_lens.min()}, Max: {combined_lens.max()}, Mean: {combined_lens.mean():.1f}")
    print(f"  Percentiles: 50%={np.percentile(combined_lens, 50):.0f}, "
          f"90%={np.percentile(combined_lens, 90):.0f}, "
          f"95%={np.percentile(combined_lens, 95):.0f}, "
          f"99%={np.percentile(combined_lens, 99):.0f}")

    print(f"\nCoverage at different max_len:")
    for max_len in [128, 192, 256, 320, 384, 512]:
        coverage = (combined_lens <= max_len).mean() * 100
        print(f"  max_len={max_len}: {coverage:.1f}%")

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    p95 = np.percentile(combined_lens, 95)
    p99 = np.percentile(combined_lens, 99)

    if p95 <= 256:
        print(f"✓ max_desc_len=256 covers 95th percentile ({p95:.0f} tokens)")
        if p99 <= 256:
            print(f"✓ max_desc_len=256 also covers 99th percentile ({p99:.0f} tokens)")
        else:
            print(f"⚠ 99th percentile is {p99:.0f} tokens - some truncation will occur")
    else:
        recommended = 320 if p95 <= 320 else 512
        print(f"⚠ 95th percentile is {p95:.0f} tokens - consider max_desc_len={recommended}")


if __name__ == "__main__":
    main()
