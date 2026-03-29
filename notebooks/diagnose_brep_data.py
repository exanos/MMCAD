# Run this in notebook cells to diagnose BRep data issues
# Copy each section to a cell

# === CELL 1: Check BRep data statistics ===
import torch
import numpy as np

# Get a batch
test_batch = remap_batch(next(iter(train_loader)))

print("="*60)
print("BREP DATA DIAGNOSTICS")
print("="*60)

# Check face features
face_feats = test_batch['face_features']
face_mask = test_batch['face_mask']
print(f"\nFace Features: {face_feats.shape}")
print(f"  Min: {face_feats.min():.4f}")
print(f"  Max: {face_feats.max():.4f}")
print(f"  Mean: {face_feats.mean():.4f}")
print(f"  Std: {face_feats.std():.4f}")
print(f"  % Zero: {(face_feats == 0).float().mean()*100:.1f}%")
print(f"  % NaN: {face_feats.isnan().float().mean()*100:.1f}%")

# Check mask
print(f"\nFace Mask: {face_mask.shape}")
print(f"  Valid faces per sample: {face_mask.sum(dim=1).mean():.1f} (of {face_mask.shape[1]})")

# Check edge features
edge_feats = test_batch['edge_features']
edge_mask = test_batch['edge_mask']
print(f"\nEdge Features: {edge_feats.shape}")
print(f"  Min: {edge_feats.min():.4f}")
print(f"  Max: {edge_feats.max():.4f}")
print(f"  Mean: {edge_feats.mean():.4f}")
print(f"  Std: {edge_feats.std():.4f}")
print(f"  % Zero: {(edge_feats == 0).float().mean()*100:.1f}%")

print(f"\nEdge Mask: {edge_mask.shape}")
print(f"  Valid edges per sample: {edge_mask.sum(dim=1).mean():.1f} (of {edge_mask.shape[1]})")

# Check topology
edge_to_faces = test_batch['edge_to_faces']
print(f"\nEdge to Faces: {edge_to_faces.shape}")
print(f"  Min idx: {edge_to_faces.min()}")
print(f"  Max idx: {edge_to_faces.max()}")
print(f"  % = -1 (invalid): {(edge_to_faces == -1).float().mean()*100:.1f}%")

# Check BFS level
bfs_level = test_batch['bfs_level']
print(f"\nBFS Level: {bfs_level.shape}")
print(f"  Min: {bfs_level.min()}")
print(f"  Max: {bfs_level.max()}")
print(f"  Mean: {bfs_level.float().mean():.2f}")

# Check spatial fields
print(f"\nFace Centroids: {test_batch['face_centroids'].shape}")
print(f"  Range: [{test_batch['face_centroids'].min():.2f}, {test_batch['face_centroids'].max():.2f}]")
print(f"  % Zero: {(test_batch['face_centroids'] == 0).float().mean()*100:.1f}%")

print(f"\nFace Normals: {test_batch['face_normals'].shape}")
print(f"  Range: [{test_batch['face_normals'].min():.4f}, {test_batch['face_normals'].max():.4f}]")
print(f"  % Zero: {(test_batch['face_normals'] == 0).float().mean()*100:.1f}%")

print(f"\nFace Areas: {test_batch['face_areas'].shape}")
print(f"  Range: [{test_batch['face_areas'].min():.4f}, {test_batch['face_areas'].max():.4f}]")
print(f"  % Zero: {(test_batch['face_areas'] == 0).float().mean()*100:.1f}%")


# === CELL 2: Check PC features ===
print("\n" + "="*60)
print("PC DATA DIAGNOSTICS")
print("="*60)

pc_local = test_batch['pc_local_features']
pc_global = test_batch['pc_global_features']

print(f"\nPC Local Features: {pc_local.shape}")
print(f"  Min: {pc_local.min():.4f}")
print(f"  Max: {pc_local.max():.4f}")
print(f"  Mean: {pc_local.mean():.4f}")
print(f"  Std: {pc_local.std():.4f}")
print(f"  % Zero: {(pc_local == 0).float().mean()*100:.1f}%")

print(f"\nPC Global Features: {pc_global.shape}")
print(f"  Min: {pc_global.min():.4f}")
print(f"  Max: {pc_global.max():.4f}")
print(f"  Mean: {pc_global.mean():.4f}")
print(f"  Std: {pc_global.std():.4f}")


# === CELL 3: Check model encoder outputs and gradients ===
print("\n" + "="*60)
print("MODEL OUTPUT DIAGNOSTICS")
print("="*60)

model.train()

# Forward pass with gradient tracking
test_batch_cuda = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in test_batch.items()}

# Don't use autocast for this diagnostic
outputs = model.forward_stage0(test_batch)

print(f"\nz_brep_raw: {outputs['z_brep_raw'].shape}")
print(f"  Min: {outputs['z_brep_raw'].min():.4f}")
print(f"  Max: {outputs['z_brep_raw'].max():.4f}")
print(f"  Std: {outputs['z_brep_raw'].std():.4f}")
print(f"  % NaN: {outputs['z_brep_raw'].isnan().float().mean()*100:.1f}%")

print(f"\nz_pc_raw: {outputs['z_pc_raw'].shape}")
print(f"  Min: {outputs['z_pc_raw'].min():.4f}")
print(f"  Max: {outputs['z_pc_raw'].max():.4f}")
print(f"  Std: {outputs['z_pc_raw'].std():.4f}")

# Check cosine similarity distribution
z_brep_norm = torch.nn.functional.normalize(outputs['z_brep_raw'], dim=-1)
z_pc_norm = torch.nn.functional.normalize(outputs['z_pc_raw'], dim=-1)
cos_sim = (z_brep_norm * z_pc_norm).sum(dim=-1)
print(f"\nTrue-pair cosine similarity:")
print(f"  Mean: {cos_sim.mean():.4f}")
print(f"  Std: {cos_sim.std():.4f}")
print(f"  Min: {cos_sim.min():.4f}")
print(f"  Max: {cos_sim.max():.4f}")

# Check if all outputs look the same (collapse)
z_brep_var = outputs['z_brep_raw'].var(dim=0).mean()
z_pc_var = outputs['z_pc_raw'].var(dim=0).mean()
print(f"\nOutput variance (low = collapse):")
print(f"  z_brep variance: {z_brep_var:.6f}")
print(f"  z_pc variance: {z_pc_var:.6f}")


# === CELL 4: Check gradient flow ===
print("\n" + "="*60)
print("GRADIENT FLOW DIAGNOSTICS")
print("="*60)

# Compute loss
loss, loss_dict = criterion(outputs, stage=0)
print(f"Total loss: {loss.item():.4f}")

# Backward
loss.backward()

# Check gradient norms
def check_grad_norm(name, module):
    total_norm = 0
    n_params = 0
    for p in module.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
            n_params += 1
    if n_params > 0:
        total_norm = total_norm ** 0.5
        return total_norm, n_params
    return 0, 0

print("\nGradient norms by component:")
brep_grad, n = check_grad_norm("brep_encoder", model.brep_encoder)
print(f"  brep_encoder: {brep_grad:.6f} ({n} params)")

pc_grad, n = check_grad_norm("pc_encoder", model.pc_encoder)
print(f"  pc_encoder: {pc_grad:.6f} ({n} params)")

brep_proj_grad, n = check_grad_norm("brep_direct_proj", model.brep_direct_proj)
print(f"  brep_direct_proj: {brep_proj_grad:.6f} ({n} params)")

pc_proj_grad, n = check_grad_norm("pc_direct_proj", model.pc_direct_proj)
print(f"  pc_direct_proj: {pc_proj_grad:.6f} ({n} params)")

proj_head_grad, n = check_grad_norm("proj_head", model.proj_head)
print(f"  proj_head: {proj_head_grad:.6f} ({n} params)")

# Check for zero/nan gradients
zero_grads = 0
nan_grads = 0
total_params = 0
for name, p in model.named_parameters():
    if p.grad is not None:
        total_params += 1
        if p.grad.abs().max() == 0:
            zero_grads += 1
        if p.grad.isnan().any():
            nan_grads += 1

print(f"\nGradient issues:")
print(f"  Zero gradient params: {zero_grads}/{total_params}")
print(f"  NaN gradient params: {nan_grads}/{total_params}")

model.zero_grad()


# === CELL 5: Test a simplified forward to isolate issues ===
print("\n" + "="*60)
print("SIMPLIFIED FORWARD TEST")
print("="*60)

# Test if BRep encoder produces varying outputs
face_feats = test_batch['face_features'].to(device).float()
edge_feats = test_batch['edge_features'].to(device).float()

# Just test the projection layers
with torch.no_grad():
    # Face projection only
    F = model.brep_encoder.face_proj(face_feats)
    print(f"\nFace projection output:")
    print(f"  Shape: {F.shape}")
    print(f"  Std: {F.std():.4f}")
    print(f"  Per-sample variance: {F.var(dim=[1,2]).mean():.4f}")

    # Edge projection only
    E = model.brep_encoder.edge_proj(edge_feats)
    print(f"\nEdge projection output:")
    print(f"  Shape: {E.shape}")
    print(f"  Std: {E.std():.4f}")
    print(f"  Per-sample variance: {E.var(dim=[1,2]).mean():.4f}")

    # PC projection
    pc_all = torch.cat([
        test_batch['pc_local_features'].to(device),
        test_batch['pc_global_features'].to(device).unsqueeze(1)
    ], dim=1)
    X_pc = model.pc_encoder.proj(pc_all.float())
    print(f"\nPC projection output:")
    print(f"  Shape: {X_pc.shape}")
    print(f"  Std: {X_pc.std():.4f}")
    print(f"  Per-sample variance: {X_pc.var(dim=[1,2]).mean():.4f}")
