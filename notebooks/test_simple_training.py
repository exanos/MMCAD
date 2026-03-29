# === TEST SIMPLE TRAINING ===
# Run this in notebook after loading data
# Copy each section to a cell

# === CELL 1: Create simplified model ===
import sys
sys.path.insert(0, '..')

from clip4cad.models.clip4cad_gfa_v4_8_1_simple import (
    CLIP4CAD_GFA_v481_Simple,
    GFAv481SimpleConfig,
    SimpleLoss,
)

config = GFAv481SimpleConfig()
simple_model = CLIP4CAD_GFA_v481_Simple(config).to(device)
simple_criterion = SimpleLoss()

print(f"Simple model parameters: {simple_model.count_parameters():,}")


# === CELL 2: Quick training test (5 epochs) ===
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np

optimizer = AdamW(simple_model.parameters(), lr=1e-3, weight_decay=0.01)

print("\n" + "="*60)
print("TESTING SIMPLIFIED MODEL TRAINING")
print("="*60)
print("If this learns (gap decreases, cosine increases), the issue is in the complex architecture.")
print("If this doesn't learn, the issue is in the data.\n")

history = {'loss': [], 'gap': [], 'cosine': [], 'r_at_1': []}

for epoch in range(5):
    simple_model.train()
    epoch_metrics = {'loss': [], 'gap': [], 'cos': [], 'r1': []}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/5")
    for batch in pbar:
        batch = remap_batch(batch)

        # Forward - use brep_pc mode (simpler)
        outputs = simple_model.forward_brep_pc(batch)
        loss, losses = simple_criterion(outputs)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(simple_model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        epoch_metrics['loss'].append(losses['total'].item())
        epoch_metrics['gap'].append(losses['gap'].item())
        epoch_metrics['cos'].append(losses['cosine'].item())
        epoch_metrics['r1'].append(losses['r_at_1'].item())

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.3f}",
            'gap': f"{losses['gap'].item():.2f}",
            'cos': f"{losses['cosine'].item():.3f}",
            'R@1': f"{losses['r_at_1'].item()*100:.1f}%",
        })

    # Epoch summary
    avg_loss = np.mean(epoch_metrics['loss'])
    avg_gap = np.mean(epoch_metrics['gap'])
    avg_cos = np.mean(epoch_metrics['cos'])
    avg_r1 = np.mean(epoch_metrics['r1'])

    history['loss'].append(avg_loss)
    history['gap'].append(avg_gap)
    history['cosine'].append(avg_cos)
    history['r_at_1'].append(avg_r1)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Gap={avg_gap:.4f}, Cos={avg_cos:.4f}, R@1={avg_r1*100:.1f}%")

# Final analysis
print("\n" + "="*60)
print("RESULTS")
print("="*60)

gap_change = history['gap'][-1] - history['gap'][0]
cos_change = history['cosine'][-1] - history['cosine'][0]
r1_change = history['r_at_1'][-1] - history['r_at_1'][0]

print(f"Gap:    {history['gap'][0]:.4f} -> {history['gap'][-1]:.4f} (change: {gap_change:+.4f})")
print(f"Cosine: {history['cosine'][0]:.4f} -> {history['cosine'][-1]:.4f} (change: {cos_change:+.4f})")
print(f"R@1:    {history['r_at_1'][0]*100:.1f}% -> {history['r_at_1'][-1]*100:.1f}% (change: {r1_change*100:+.1f}%)")

if gap_change < -0.1 and cos_change > 0.05:
    print("\n✓ SIMPLE MODEL IS LEARNING!")
    print("  This means the issue is in the COMPLEX ARCHITECTURE (topology, codebook, etc.)")
    print("  Try: Simplify the TopologyBRepEncoder or disable message passing")
else:
    print("\n✗ SIMPLE MODEL IS NOT LEARNING")
    print("  This means the issue is likely in the DATA")
    print("  Check: BRep features may be wrong, or there's no learnable signal")


# === CELL 3: Visualize learning ===
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].plot(history['loss'])
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')

axes[1].plot(history['gap'])
axes[1].set_title('BRep-PC Gap')
axes[1].set_xlabel('Epoch')
axes[1].axhline(y=0, color='g', linestyle='--', alpha=0.5)

axes[2].plot(history['cosine'])
axes[2].set_title('BRep-PC Cosine')
axes[2].set_xlabel('Epoch')
axes[2].axhline(y=1, color='g', linestyle='--', alpha=0.5)

axes[3].plot([r*100 for r in history['r_at_1']])
axes[3].set_title('R@1 (%)')
axes[3].set_xlabel('Epoch')

plt.tight_layout()
plt.show()


# === CELL 4: Check what the model actually learned ===
print("\n" + "="*60)
print("CHECKING LEARNED REPRESENTATIONS")
print("="*60)

simple_model.eval()
with torch.no_grad():
    test_batch = remap_batch(next(iter(train_loader)))
    outputs = simple_model.forward_brep_pc(test_batch)

    z_brep = outputs['z_brep_raw']
    z_pc = outputs['z_pc_raw']

    # Check variation
    print(f"\nz_brep variance (should be > 0):")
    print(f"  Per-dim mean: {z_brep.var(dim=0).mean():.6f}")
    print(f"  Per-sample mean: {z_brep.var(dim=1).mean():.6f}")

    print(f"\nz_pc variance (should be > 0):")
    print(f"  Per-dim mean: {z_pc.var(dim=0).mean():.6f}")
    print(f"  Per-sample mean: {z_pc.var(dim=1).mean():.6f}")

    # Check if BRep and PC are now aligned
    cos_sim = torch.nn.functional.cosine_similarity(z_brep, z_pc, dim=-1)
    print(f"\nTrue-pair cosine similarity:")
    print(f"  Mean: {cos_sim.mean():.4f}")
    print(f"  Std: {cos_sim.std():.4f}")
    print(f"  Min: {cos_sim.min():.4f}")
    print(f"  Max: {cos_sim.max():.4f}")
