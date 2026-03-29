# === TEST AUTOBREP-STYLE ENCODER ===
# Run this in notebook after loading data
# Copy each section to a cell

# === CELL 1: Setup and imports ===
import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

from clip4cad.models.brep_encoder_autobrep import (
    AutoBrepEncoderConfig,
    AutoBrepEncoder,
    SimpleCLIP4CAD_AutoBrep,
    SimpleContrastiveLoss,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# === CELL 2: Batch remapping function ===
def remap_batch_autobrep(batch):
    """Remap batch keys to match the encoder's expected format."""
    return {
        'face_features': batch['brep_face_features'],
        'edge_features': batch['brep_edge_features'],
        'face_mask': batch['brep_face_mask'],
        'edge_mask': batch['brep_edge_mask'],
        'face_centroids': batch.get('face_centroids', torch.zeros_like(batch['brep_face_features'][..., :3])),
        # PC features: split into local and global
        'pc_local_features': batch['pc_features'][:, :-1, :],  # All but last
        'pc_global_features': batch['pc_features'][:, -1, :],   # Last token
    }


# === CELL 3: Create model ===
config = AutoBrepEncoderConfig(
    d_face=48,
    d_edge=12,
    d_model=256,
    d_proj=128,
    num_layers=4,
    num_heads=8,
    dropout=0.1,
)

model = SimpleCLIP4CAD_AutoBrep(config).to(device)
criterion = SimpleContrastiveLoss()

print(f"Model parameters: {model.count_parameters():,}")
print(f"BRep encoder: {model.brep_encoder.count_parameters():,}")


# === CELL 4: Test forward pass ===
print("\n" + "="*60)
print("TESTING FORWARD PASS")
print("="*60)

model.eval()
with torch.no_grad():
    test_batch = remap_batch_autobrep(next(iter(train_loader)))
    outputs = model(test_batch)

    print(f"z_brep: {outputs['z_brep'].shape}")
    print(f"z_pc: {outputs['z_pc'].shape}")

    # Check variance (should not be collapsed)
    print(f"\nz_brep variance:")
    print(f"  Per-dim: {outputs['z_brep_raw'].var(dim=0).mean():.6f}")
    print(f"  Per-sample: {outputs['z_brep_raw'].var(dim=1).mean():.6f}")

    print(f"\nz_pc variance:")
    print(f"  Per-dim: {outputs['z_pc_raw'].var(dim=0).mean():.6f}")
    print(f"  Per-sample: {outputs['z_pc_raw'].var(dim=1).mean():.6f}")

    # Initial cosine similarity
    cos = F.cosine_similarity(outputs['z_brep_raw'], outputs['z_pc_raw'], dim=-1)
    print(f"\nInitial cosine similarity: {cos.mean():.4f} ± {cos.std():.4f}")


# === CELL 5: Training loop ===
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
num_epochs = 10

print("\n" + "="*60)
print("TRAINING AUTOBREP-STYLE ENCODER")
print("="*60)
print("This should show learning (gap decreases, cosine increases)")
print("If not learning, the issue may be in the data.\n")

history = {'loss': [], 'gap': [], 'cosine': [], 'r_at_1': []}

for epoch in range(num_epochs):
    model.train()
    epoch_metrics = {'loss': [], 'gap': [], 'cos': [], 'r1': []}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in pbar:
        batch = remap_batch_autobrep(batch)

        # Forward
        outputs = model(batch)
        loss, losses = criterion(outputs)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track
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


# === CELL 6: Analysis ===
print("\n" + "="*60)
print("TRAINING RESULTS")
print("="*60)

gap_change = history['gap'][-1] - history['gap'][0]
cos_change = history['cosine'][-1] - history['cosine'][0]
r1_change = history['r_at_1'][-1] - history['r_at_1'][0]

print(f"Gap:    {history['gap'][0]:.4f} -> {history['gap'][-1]:.4f} (change: {gap_change:+.4f})")
print(f"Cosine: {history['cosine'][0]:.4f} -> {history['cosine'][-1]:.4f} (change: {cos_change:+.4f})")
print(f"R@1:    {history['r_at_1'][0]*100:.1f}% -> {history['r_at_1'][-1]*100:.1f}% (change: {r1_change*100:+.1f}%)")

if gap_change < -0.1 or cos_change > 0.05 or r1_change > 0.02:
    print("\n✓ MODEL IS LEARNING!")
    print("  The AutoBrep-style encoder (without broken topology fields) works.")
    print("  Next steps:")
    print("    1. Increase training epochs")
    print("    2. Add text modality")
    print("    3. Tune hyperparameters")
else:
    print("\n✗ MODEL IS NOT LEARNING")
    print("  Possible issues:")
    print("    1. Learning rate too low/high")
    print("    2. FSQ features may not have enough signal")
    print("    3. Try more epochs")


# === CELL 7: Visualize ===
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].plot(history['loss'])
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')

axes[1].plot(history['gap'])
axes[1].set_title('BRep-PC Gap (L2)')
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


# === CELL 8: Check learned representations ===
print("\n" + "="*60)
print("FINAL REPRESENTATIONS")
print("="*60)

model.eval()
with torch.no_grad():
    test_batch = remap_batch_autobrep(next(iter(train_loader)))
    outputs = model(test_batch)

    z_brep = outputs['z_brep_raw']
    z_pc = outputs['z_pc_raw']

    # Variance
    print(f"z_brep variance: {z_brep.var(dim=0).mean():.6f}")
    print(f"z_pc variance: {z_pc.var(dim=0).mean():.6f}")

    # True-pair similarity
    cos_sim = F.cosine_similarity(z_brep, z_pc, dim=-1)
    print(f"\nTrue-pair cosine similarity:")
    print(f"  Mean: {cos_sim.mean():.4f}")
    print(f"  Std: {cos_sim.std():.4f}")
    print(f"  Min: {cos_sim.min():.4f}")
    print(f"  Max: {cos_sim.max():.4f}")

    # Cross similarity matrix
    z_brep_norm = F.normalize(z_brep, dim=-1)
    z_pc_norm = F.normalize(z_pc, dim=-1)
    sim_matrix = z_brep_norm @ z_pc_norm.T

    # R@1 on batch
    pred = sim_matrix.argmax(dim=1)
    labels = torch.arange(z_brep.size(0), device=z_brep.device)
    r1 = (pred == labels).float().mean()
    print(f"\nBatch R@1: {r1*100:.1f}%")


# === CELL 9: Compare with broken topology encoder (optional) ===
# Uncomment to compare with the original v4.8.1 encoder
#
# from clip4cad.models.clip4cad_gfa_v4_8_1 import CLIP4CAD_GFA_v481, GFAv481Config
#
# old_config = GFAv481Config()
# old_model = CLIP4CAD_GFA_v481(old_config).to(device)
#
# # Run same batch through both
# with torch.no_grad():
#     # New encoder
#     new_outputs = model(test_batch)
#
#     # Old encoder (needs edge_to_faces, bfs_level which are broken)
#     old_batch = {
#         'face_features': test_batch['face_features'],
#         'edge_features': test_batch['edge_features'],
#         'face_mask': test_batch['face_mask'],
#         'edge_mask': test_batch['edge_mask'],
#         'edge_to_faces': batch.get('edge_to_faces', torch.full((test_batch['face_mask'].size(0), 512, 2), -1, dtype=torch.long)).to(device),
#         'bfs_level': batch.get('bfs_level', torch.zeros((test_batch['face_mask'].size(0), 192), dtype=torch.long)).to(device),
#         'face_centroids': test_batch['face_centroids'],
#     }
#     old_outputs = old_model.forward_stage0(old_batch)
#
#     print("New encoder cosine:", F.cosine_similarity(new_outputs['z_brep_raw'], new_outputs['z_pc_raw'], dim=-1).mean().item())
#     print("Old encoder cosine:", F.cosine_similarity(old_outputs['z_brep_raw'], old_outputs['z_pc_raw'], dim=-1).mean().item())
