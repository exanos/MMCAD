"""
Quick diagnostic script for HUS embeddings.
Run this in the notebook after loading model and data.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

def diagnose_embeddings(model, val_loader, device="cuda", max_batches=10):
    """
    Diagnose embedding quality issues.

    Checks:
    1. Embedding statistics (mean, std, min, max)
    2. Embedding collapse (are all embeddings the same?)
    3. Cross-modal alignment (are modalities in same space?)
    4. Self-similarity (does same sample match itself?)
    """
    model.eval()

    all_z_brep = []
    all_z_pc = []
    all_z_text = []
    all_z_brep_global = []
    all_z_brep_detail = []

    print("Collecting embeddings...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            if i >= max_batches:
                break

            outputs = model(batch)

            all_z_brep.append(outputs["z_brep"].cpu())
            all_z_pc.append(outputs["z_pc"].cpu())
            all_z_text.append(outputs["z_text"].cpu())
            all_z_brep_global.append(outputs["z_brep_global"].cpu())
            all_z_brep_detail.append(outputs["z_brep_detail"].cpu())

    z_brep = torch.cat(all_z_brep)
    z_pc = torch.cat(all_z_pc)
    z_text = torch.cat(all_z_text)
    z_brep_global = torch.cat(all_z_brep_global)
    z_brep_detail = torch.cat(all_z_brep_detail)

    print("\n" + "=" * 70)
    print("EMBEDDING STATISTICS")
    print("=" * 70)

    for name, emb in [("z_brep", z_brep), ("z_pc", z_pc), ("z_text", z_text),
                      ("z_brep_global", z_brep_global), ("z_brep_detail", z_brep_detail)]:
        print(f"\n{name}:")
        print(f"  Shape: {emb.shape}")
        print(f"  Mean: {emb.mean():.4f}")
        print(f"  Std: {emb.std():.4f}")
        print(f"  Min: {emb.min():.4f}")
        print(f"  Max: {emb.max():.4f}")
        print(f"  L2 norm mean: {emb.norm(dim=-1).mean():.4f}")

    print("\n" + "=" * 70)
    print("EMBEDDING COLLAPSE CHECK")
    print("=" * 70)

    # Check if embeddings are collapsed (all very similar)
    for name, emb in [("z_brep", z_brep), ("z_pc", z_pc), ("z_text", z_text)]:
        emb_norm = F.normalize(emb, dim=-1)

        # Average pairwise similarity (should be low for diverse embeddings)
        # Sample 100 pairs to speed up
        n = min(100, len(emb))
        idx = torch.randperm(len(emb))[:n]
        sampled = emb_norm[idx]
        pairwise_sim = torch.mm(sampled, sampled.T)

        # Remove diagonal (self-similarity = 1)
        mask = ~torch.eye(n, dtype=torch.bool)
        avg_sim = pairwise_sim[mask].mean().item()

        collapse_status = "COLLAPSED!" if avg_sim > 0.9 else "OK" if avg_sim < 0.5 else "MODERATE"
        print(f"\n{name}:")
        print(f"  Avg pairwise similarity: {avg_sim:.4f}")
        print(f"  Status: {collapse_status}")

        if avg_sim > 0.8:
            print(f"  WARNING: Embeddings are too similar! Model may not be learning.")

    print("\n" + "=" * 70)
    print("CROSS-MODAL ALIGNMENT CHECK")
    print("=" * 70)

    # Check Text-BRep alignment
    z_text_norm = F.normalize(z_text, dim=-1)
    z_brep_norm = F.normalize(z_brep, dim=-1)
    z_pc_norm = F.normalize(z_pc, dim=-1)

    # Diagonal should be high (matched pairs)
    n = min(len(z_text), len(z_brep), 100)

    sim_text_brep = torch.mm(z_text_norm[:n], z_brep_norm[:n].T)
    sim_text_pc = torch.mm(z_text_norm[:n], z_pc_norm[:n].T)

    diag_sim_brep = sim_text_brep.diag().mean().item()
    diag_sim_pc = sim_text_pc.diag().mean().item()

    # Off-diagonal should be low (non-matched pairs)
    mask = ~torch.eye(n, dtype=torch.bool)
    offdiag_sim_brep = sim_text_brep[mask].mean().item()
    offdiag_sim_pc = sim_text_pc[mask].mean().item()

    print(f"\nText-BRep alignment:")
    print(f"  Matched pairs similarity: {diag_sim_brep:.4f}")
    print(f"  Non-matched pairs similarity: {offdiag_sim_brep:.4f}")
    print(f"  Gap (should be positive): {diag_sim_brep - offdiag_sim_brep:.4f}")

    print(f"\nText-PC alignment:")
    print(f"  Matched pairs similarity: {diag_sim_pc:.4f}")
    print(f"  Non-matched pairs similarity: {offdiag_sim_pc:.4f}")
    print(f"  Gap (should be positive): {diag_sim_pc - offdiag_sim_pc:.4f}")

    if diag_sim_brep - offdiag_sim_brep < 0.05:
        print("\n  WARNING: Text-BRep embeddings are NOT aligned!")
        print("  The model is not learning meaningful cross-modal relationships.")

    print("\n" + "=" * 70)
    print("GLOBAL vs DETAIL COMPARISON")
    print("=" * 70)

    z_brep_global_norm = F.normalize(z_brep_global, dim=-1)
    z_brep_detail_norm = F.normalize(z_brep_detail, dim=-1)

    # Are global and detail embeddings different?
    global_detail_sim = (z_brep_global_norm * z_brep_detail_norm).sum(dim=-1).mean().item()
    print(f"\nBRep Global-Detail similarity: {global_detail_sim:.4f}")

    if global_detail_sim > 0.95:
        print("  WARNING: Global and Detail embeddings are nearly identical!")
        print("  The hierarchical structure is not working.")
    elif global_detail_sim < 0.3:
        print("  OK: Global and Detail capture different information.")
    else:
        print("  MODERATE: Some differentiation between levels.")

    print("\n" + "=" * 70)
    print("RETRIEVAL SANITY CHECK")
    print("=" * 70)

    # Quick R@1 check
    sim = torch.mm(z_text_norm[:n], z_brep_norm[:n].T)
    rankings = sim.argmax(dim=1)
    correct = (rankings == torch.arange(n)).sum().item()
    r_at_1 = correct / n

    print(f"\nQuick Text->BRep R@1 (first {n} samples): {r_at_1:.4f}")

    if r_at_1 < 0.05:
        print("  CRITICAL: R@1 < 5% means embeddings are essentially random!")
    elif r_at_1 < 0.3:
        print("  WARNING: R@1 < 30% indicates poor alignment.")
    else:
        print("  OK: Reasonable retrieval performance.")

    return {
        "z_brep_stats": {"mean": z_brep.mean().item(), "std": z_brep.std().item()},
        "text_brep_gap": diag_sim_brep - offdiag_sim_brep,
        "global_detail_sim": global_detail_sim,
        "quick_r1": r_at_1,
    }


# Usage in notebook:
# results = diagnose_embeddings(model, val_loader, device=str(device))
