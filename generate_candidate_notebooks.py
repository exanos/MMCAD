"""Generate 5 CLIP4CAD architecture candidate notebooks for Google Colab.

Candidates:
  A: VIB-Contrastive
  B: Sheaf B-Rep Encoder
  C: Hierarchical Nested Contrastive
  D: SIGReg + Predictive Auxiliary
  E: Full Stack (A+B+C+D)

Each notebook is fully self-contained (no external imports from clip4cad package).
"""
import json
import os

# ============================================================
# Cell building helpers
# ============================================================

def make_code_cell(source):
    return {
        "cell_type": "code", "metadata": {},
        "source": [line + "\n" for line in source.split("\n")],
        "execution_count": None, "outputs": []
    }

def make_md_cell(source):
    return {
        "cell_type": "markdown", "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

# ============================================================
# SHARED CELLS — identical across all 5 notebooks
# ============================================================

CELL_INSTALL = '''\
# Cell: Install Packages
print("Installing packages...")

!pip install -q plyfile tensorboard h5py einops

# Try Gated DeltaNet (optional — falls back to standard attention if unavailable)
import subprocess, sys
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gated-deltanet"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("gated-deltanet installed!")
except Exception:
    print("gated-deltanet not available — using standard attention fallback")

import torch
print(f"\\n{'='*60}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB")
print(f"{'='*60}")'''

CELL_IMPORTS = '''\
# Cell: Imports
import os, sys, gc, time, math, warnings, itertools, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import h5py
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from plyfile import PlyData
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Gated DeltaNet availability
try:
    from gated_deltanet import GatedDeltaNetLayer
    HAS_GDN = True
    print("Gated DeltaNet: AVAILABLE (using NVIDIA implementation)")
except ImportError:
    HAS_GDN = False
    print("Gated DeltaNet: NOT AVAILABLE (using standard attention fallback)")

print("All libraries imported!")'''

CELL_MOUNT = '''\
# Cell: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
print("\\nGoogle Drive mounted at /content/drive/")'''

CELL_DATA = '''\
# Cell: Extract/Copy Data
import os, time

EXTRACT_DIR = "/content/mmcad_data"
os.makedirs(EXTRACT_DIR, exist_ok=True)

# --- Point Clouds (lz4 archive) ---
print("=" * 60)
print("EXTRACTING POINT CLOUDS")
print("=" * 60)

PLY_TAR_LZ4 = "/content/drive/MyDrive/MMCAD/ply.tar.lz4"
ply_dir = f"{EXTRACT_DIR}/abc_ply_organized"
if os.path.exists(ply_dir) and len(os.listdir(ply_dir)) > 100:
    print(f"Point clouds already extracted ({len(os.listdir(ply_dir))} files)")
else:
    print(f"Extracting from: {PLY_TAR_LZ4}")
    start = time.time()
    !apt-get install -qq lz4 > /dev/null 2>&1
    !lz4 -dc "{PLY_TAR_LZ4}" | tar -xf - -C "{EXTRACT_DIR}"
    print(f"Extracted in {time.time()-start:.1f}s")

# --- CSV ---
print("\\nCopying CSV...")
!cp /content/drive/MyDrive/MMCAD/abc_dataset_clean.csv /content/mmcad_data/
print("CSV copied!")

# --- B-Rep features (copy to local SSD for speed) ---
print("\\nCopying B-Rep features...")
BREP_H5_SRC = "/content/drive/MyDrive/MMCAD/brep_autobrep.h5"
BREP_H5_DST = f"{EXTRACT_DIR}/brep_autobrep.h5"
if os.path.exists(BREP_H5_DST):
    print(f"B-Rep HDF5 already exists")
else:
    !cp "{BREP_H5_SRC}" "{BREP_H5_DST}"
    print(f"B-Rep HDF5 copied!")

# --- Text features (keep on Drive — files are very large) ---
print("\\nText embeddings will be read directly from Drive (too large to copy)")
TRAIN_TEXT_H5 = "/content/drive/MyDrive/MMCAD/train_text_embeddings.h5"
VAL_TEXT_H5 = "/content/drive/MyDrive/MMCAD/val_text_embeddings.h5"
TEST_TEXT_H5 = "/content/drive/MyDrive/MMCAD/test_text_embeddings.h5"

for path, name in [(TRAIN_TEXT_H5, "Train"), (VAL_TEXT_H5, "Val"), (TEST_TEXT_H5, "Test")]:
    if os.path.exists(path):
        sz = os.path.getsize(path) / 1e9
        print(f"  {name}: {path} ({sz:.1f} GB)")
    else:
        print(f"  WARNING: {name} text file not found at {path}")

print("\\n" + "=" * 60)
print("ALL DATA READY!")
print("=" * 60)'''

CELL_SPLIT = '''\
# Cell: Data Loading with Train/Val/Test Split (Combined train+val for training)
df = pd.read_csv(config.CSV_PATH)
print(f"Total entries in CSV: {len(df)}")

valid_df = df[df['completeness_score'] == 7].copy()
print(f"Complete entries (score=7): {len(valid_df)}")

valid_df = valid_df[
    (valid_df['has_iso1'] == True) &
    (valid_df['has_iso2'] == True) &
    (valid_df['has_ply'] == True)
].copy()
print(f"Valid entries after verification: {len(valid_df)}")

if config.SUBSET_SIZE:
    valid_df = valid_df.sample(config.SUBSET_SIZE, random_state=config.RANDOM_SEED)
    print(f"Using subset of {config.SUBSET_SIZE} samples")

train_split_path = os.path.join(config.SPLIT_DIR, 'train_uids.txt')
val_split_path = os.path.join(config.SPLIT_DIR, 'val_uids.txt')
test_split_path = os.path.join(config.SPLIT_DIR, 'test_uids.txt')

if os.path.exists(train_split_path) and os.path.exists(val_split_path) and os.path.exists(test_split_path):
    print("Loading existing split files...")
    with open(train_split_path) as f: train_uids = set(int(line.strip()) for line in f if line.strip())
    with open(val_split_path) as f: val_uids = set(int(line.strip()) for line in f if line.strip())
    with open(test_split_path) as f: test_uids = set(int(line.strip()) for line in f if line.strip())
    train_df = valid_df[valid_df['uid'].isin(train_uids)]
    val_df = valid_df[valid_df['uid'].isin(val_uids)]
    test_df = valid_df[valid_df['uid'].isin(test_uids)]
else:
    print("Creating new train/val/test splits...")
    train_df, temp_df = train_test_split(valid_df, train_size=config.TRAIN_RATIO, random_state=config.RANDOM_SEED)
    val_ratio_adjusted = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio_adjusted, random_state=config.RANDOM_SEED)
    os.makedirs(config.SPLIT_DIR, exist_ok=True)
    for path, split_df in [(train_split_path, train_df), (val_split_path, val_df), (test_split_path, test_df)]:
        with open(path, 'w') as f:
            for uid in split_df['uid']: f.write(f"{uid}\\n")
    print(f"Split files saved to {config.SPLIT_DIR}")

print(f"\\nOriginal Splits:")
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Combine train + val for training (no memory constraints on Colab)
trainval_df = pd.concat([train_df, val_df], ignore_index=True)
eval_df = test_df.copy()
print(f"\\nCombined for training:")
print(f"  Train+Val: {len(trainval_df)} samples")
print(f"  Test (eval): {len(eval_df)} samples")'''

CELL_UID_INDEX = '''\
# Cell: Build HDF5 UID Index Maps & Filter to Trimodal Samples
print("Building HDF5 UID index maps...")

# --- B-Rep features ---
brep_h5 = h5py.File(config.BREP_H5_PATH, 'r')
brep_uids_raw = brep_h5['uids'][:]
brep_uid_to_idx = {}
for i, uid in enumerate(brep_uids_raw):
    uid_str = uid.decode('utf-8') if isinstance(uid, bytes) else str(uid)
    brep_uid_to_idx[uid_str] = i
print(f"B-Rep features: {len(brep_uid_to_idx)} samples")
print(f"  face_features: {brep_h5['face_features'].shape}")
print(f"  edge_features: {brep_h5['edge_features'].shape}")
brep_h5.close()

# --- Text features (3 separate H5 files) ---
def build_text_uid_map(h5_path, label):
    """Build UID -> (h5_path, index) mapping for a text H5 file."""
    h5 = h5py.File(h5_path, 'r')
    uid_map = {}
    # Check for UIDs in the file
    uid_key = None
    for key in ['uids', 'sample_ids']:
        if key in h5:
            uid_key = key
            break
    if uid_key:
        uids_raw = h5[uid_key][:]
        for i, uid in enumerate(uids_raw):
            uid_str = uid.decode('utf-8') if isinstance(uid, bytes) else str(uid)
            uid_map[uid_str] = (h5_path, i)
        print(f"  {label}: {len(uid_map)} samples (UIDs from H5)")
    else:
        print(f"  {label}: {h5['desc_embeddings'].shape[0]} samples (no UIDs in H5 — will use B-Rep UID order)")
    h5.close()
    return uid_map

text_uid_to_source = {}
for h5_path, label in [(config.TRAIN_TEXT_H5, "Train text"), (config.VAL_TEXT_H5, "Val text")]:
    if os.path.exists(h5_path):
        text_uid_to_source.update(build_text_uid_map(h5_path, label))

test_text_uid_to_source = {}
if os.path.exists(config.TEST_TEXT_H5):
    test_text_uid_to_source = build_text_uid_map(config.TEST_TEXT_H5, "Test text")

# If text H5 files don't have UIDs, build mapping from B-Rep UID order + split files
if not text_uid_to_source:
    print("\\n  Building text UID maps from split UIDs (positional indexing)...")
    # Train text
    train_text_h5 = h5py.File(config.TRAIN_TEXT_H5, 'r')
    n_train_text = train_text_h5['desc_embeddings'].shape[0]
    train_text_h5.close()
    # Match train+val UIDs to train text (first N entries are train)
    train_uids_list = sorted(train_df['uid'].astype(str).tolist())[:n_train_text]
    for i, uid in enumerate(train_uids_list):
        text_uid_to_source[uid] = (config.TRAIN_TEXT_H5, i)
    # Val text
    val_text_h5 = h5py.File(config.VAL_TEXT_H5, 'r')
    n_val_text = val_text_h5['desc_embeddings'].shape[0]
    val_text_h5.close()
    val_uids_list = sorted(val_df['uid'].astype(str).tolist())[:n_val_text]
    for i, uid in enumerate(val_uids_list):
        text_uid_to_source[uid] = (config.VAL_TEXT_H5, i)
    # Test text
    test_text_h5 = h5py.File(config.TEST_TEXT_H5, 'r')
    n_test_text = test_text_h5['desc_embeddings'].shape[0]
    test_text_h5.close()
    test_uids_list = sorted(test_df['uid'].astype(str).tolist())[:n_test_text]
    for i, uid in enumerate(test_uids_list):
        test_text_uid_to_source[uid] = (config.TEST_TEXT_H5, i)
    print(f"  Train+Val text: {len(text_uid_to_source)}, Test text: {len(test_text_uid_to_source)}")

all_text_uid_to_source = {**text_uid_to_source, **test_text_uid_to_source}
print(f"\\nTotal text UIDs: {len(all_text_uid_to_source)}")

# --- Filter to trimodal samples ---
print("\\nFiltering to trimodal samples...")
def filter_trimodal(df, text_map):
    mask = df['uid'].astype(str).isin(brep_uid_to_idx) & df['uid'].astype(str).isin(text_map)
    return df[mask].reset_index(drop=True)

trainval_df_tri = filter_trimodal(trainval_df, text_uid_to_source)
eval_df_tri = filter_trimodal(eval_df, test_text_uid_to_source if test_text_uid_to_source else all_text_uid_to_source)

print(f"\\nTrimodal samples:")
print(f"  Train+Val: {len(trainval_df_tri)} (from {len(trainval_df)})")
print(f"  Test: {len(eval_df_tri)} (from {len(eval_df)})")'''

CELL_DGCNN = '''\
# Cell: DGCNN Encoder (same as baseline notebooks)
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    return (-xx - inner - xx.transpose(2, 1)).topk(k=k, dim=-1)[1]

def get_graph_feature(x, k=20):
    bs, d, n = x.size()
    idx = knn(x, k)
    idx_base = torch.arange(0, bs, device=x.device).view(-1, 1, 1) * n
    idx = (idx + idx_base).view(-1)
    x = x.transpose(2, 1).contiguous()
    feat = x.view(bs * n, -1)[idx].view(bs, n, k, d)
    x = x.view(bs, n, 1, d).repeat(1, 1, k, 1)
    return torch.cat((feat - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

class DGCNNEncoder(nn.Module):
    def __init__(self, latent_size=1024, k=20):
        super().__init__()
        self.k = k
        self.bn1, self.bn2 = nn.BatchNorm2d(64), nn.BatchNorm2d(64)
        self.bn3, self.bn4 = nn.BatchNorm2d(128), nn.BatchNorm2d(256)
        self.bn5, self.bn6, self.bn7 = nn.BatchNorm1d(latent_size), nn.BatchNorm1d(512), nn.BatchNorm1d(latent_size)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), self.bn1, nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), self.bn2, nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), self.bn3, nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), self.bn4, nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, latent_size, 1, bias=False), self.bn5, nn.LeakyReLU(0.2))
        self.linear1 = nn.Linear(latent_size * 2, 512, bias=False)
        self.linear2 = nn.Linear(512, latent_size)
        self.dp = nn.Dropout(0.3)

    def forward(self, x):
        bs = x.size(0)
        x1 = self.conv1(get_graph_feature(x, self.k)).max(dim=-1)[0]
        x2 = self.conv2(get_graph_feature(x1, self.k)).max(dim=-1)[0]
        x3 = self.conv3(get_graph_feature(x2, self.k)).max(dim=-1)[0]
        x4 = self.conv4(get_graph_feature(x3, self.k)).max(dim=-1)[0]
        x = self.conv5(torch.cat((x1, x2, x3, x4), dim=1))
        x = torch.cat((x.max(2)[0], x.mean(2)), 1)
        x = self.dp(F.leaky_relu(self.bn6(self.linear1(x)), 0.2))
        return self.dp(self.bn7(self.linear2(x)))

print("DGCNN encoder defined!")'''

CELL_PLY = '''\
# Cell: PLY Loading
def load_ply_fast(path):
    plydata = PlyData.read(path)
    return np.stack([plydata['vertex']['x'],
                    plydata['vertex']['y'],
                    plydata['vertex']['z']], axis=1).astype(np.float32)

print("PLY loader defined!")'''

CELL_ATTN_POOL = '''\
# Cell: Attention Pooling (v4.9 style — replaces codebook)
class AttentionPooling(nn.Module):
    def __init__(self, d, num_queries=16, num_heads=8, dropout=0.1, num_layers=2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, d) * 0.02)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 4, d))
            for _ in range(num_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])
        self.self_attn = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d)
        self.pool_weights = nn.Linear(d, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        B = X.shape[0]
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = ~mask.bool() if mask is not None else None
        for i in range(len(self.cross_attn_layers)):
            attn_out, _ = self.cross_attn_layers[i](
                query=Q, key=X, value=X,
                key_padding_mask=key_padding_mask, need_weights=False
            )
            Q = self.norms1[i](Q + self.dropout(attn_out))
            Q = self.norms2[i](Q + self.dropout(self.ffn_layers[i](Q)))
        self_attn_out, _ = self.self_attn(Q, Q, Q, need_weights=False)
        Q = self.self_attn_norm(Q + self.dropout(self_attn_out))
        weights = torch.softmax(self.pool_weights(Q).squeeze(-1), dim=-1)
        return torch.einsum('bk,bkd->bd', weights, Q)

print("AttentionPooling defined!")'''

# ============================================================
# Universal Infrastructure: mHC-lite + Gated DeltaNet Hybrid
# ============================================================

CELL_INFRASTRUCTURE = '''\
# Cell: Universal Infrastructure — mHC-lite + Gated DeltaNet Hybrid
import itertools

class mHCLiteConnection(nn.Module):
    """Manifold Hyper-Connection Lite (Birkhoff-von Neumann reparameterization).
    Replaces standard residual connections. n=3 streams, 6 learnable params per connection.
    """
    def __init__(self, d_model, n_streams=3):
        super().__init__()
        self.n_streams = n_streams
        perms = list(itertools.permutations(range(n_streams)))
        perm_tensors = []
        for p in perms:
            P = torch.zeros(n_streams, n_streams)
            for i, j in enumerate(p):
                P[i, j] = 1.0
            perm_tensors.append(P)
        self.register_buffer('perm_matrices', torch.stack(perm_tensors))
        self.coeffs = nn.Parameter(torch.zeros(len(perms)))

    def get_mixing_matrix(self):
        alpha = F.softmax(self.coeffs, dim=0)
        return torch.einsum('p,pij->ij', alpha, self.perm_matrices)

    def forward(self, streams):
        M = self.get_mixing_matrix()
        return torch.einsum('ij,bsnj->bsni', M, streams)


class GatedDeltaNetBlock(nn.Module):
    """Wrapper: uses NVIDIA GatedDeltaNet if available, else standard TransformerEncoderLayer."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if HAS_GDN:
            self.layer = GatedDeltaNetLayer(d_model=d_model, num_heads=n_heads)
            self.is_gdn = True
        else:
            self.layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True
            )
            self.is_gdn = False

    def forward(self, x, mask=None):
        if self.is_gdn:
            return self.layer(x)
        else:
            kpm = ~mask.bool() if mask is not None else None
            return self.layer(x, src_key_padding_mask=kpm)


class FullAttentionBlock(nn.Module):
    """Standard multi-head self-attention + FFN block."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
    def forward(self, x, mask=None):
        kpm = ~mask.bool() if mask is not None else None
        return self.layer(x, src_key_padding_mask=kpm)


class HybridBRepTransformer(nn.Module):
    """6-layer hybrid: 4 Gated DeltaNet + 2 Full Attention (at layers 3,6), with mHC-lite."""
    def __init__(self, d_model, n_heads, n_streams=3, dropout=0.1):
        super().__init__()
        self.n_streams = n_streams
        self.layers = nn.ModuleList()
        self.mhc_connections = nn.ModuleList()
        for i in range(6):
            if i in [2, 5]:  # Full attention at layers 3 and 6
                self.layers.append(FullAttentionBlock(d_model, n_heads, dropout))
            else:
                self.layers.append(GatedDeltaNetBlock(d_model, n_heads, dropout))
            self.mhc_connections.append(mHCLiteConnection(d_model, n_streams))

    def forward(self, x, mask=None):
        B, S, d = x.shape
        streams = x.unsqueeze(2).expand(-1, -1, self.n_streams, -1).clone()
        for layer, mhc in zip(self.layers, self.mhc_connections):
            streams = mhc(streams)
            x = streams[:, :, 0, :]
            x = layer(x, mask=mask)
            streams = streams.clone()
            streams[:, :, 0, :] = x
        return streams[:, :, 0, :]


class HybridTextTransformer(nn.Module):
    """4-layer hybrid: 3 Gated DeltaNet + 1 Full Attention (layer 4), with mHC-lite."""
    def __init__(self, d_model, n_heads, n_streams=3, dropout=0.1):
        super().__init__()
        self.n_streams = n_streams
        self.layers = nn.ModuleList()
        self.mhc_connections = nn.ModuleList()
        for i in range(4):
            if i == 3:  # Full attention at last layer
                self.layers.append(FullAttentionBlock(d_model, n_heads, dropout))
            else:
                self.layers.append(GatedDeltaNetBlock(d_model, n_heads, dropout))
            self.mhc_connections.append(mHCLiteConnection(d_model, n_streams))

    def forward(self, x, mask=None):
        B, S, d = x.shape
        streams = x.unsqueeze(2).expand(-1, -1, self.n_streams, -1).clone()
        for layer, mhc in zip(self.layers, self.mhc_connections):
            streams = mhc(streams)
            x = streams[:, :, 0, :]
            x = layer(x, mask=mask)
            streams = streams.clone()
            streams[:, :, 0, :] = x
        return streams[:, :, 0, :]

print("Infrastructure defined: mHC-lite + Hybrid Transformers")
print(f"  Gated DeltaNet: {'NVIDIA implementation' if HAS_GDN else 'Standard attention fallback'}")'''

# ============================================================
# B-Rep Encoders (2 variants)
# ============================================================

CELL_BREP_ENCODER_TOPOLOGY = '''\
# Cell: Topology-Aware B-Rep Encoder (EdgeMessageLayer + HybridBRepTransformer)

class EdgeMessageLayer(nn.Module):
    """Message passing between faces and edges through B-Rep topology."""
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.face_to_edge = nn.Sequential(
            nn.Linear(d * 3, d * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 2, d))
        self.edge_to_face = nn.Sequential(
            nn.Linear(d * 2, d * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 2, d))
        self.norm_f = nn.LayerNorm(d)
        self.norm_e = nn.LayerNorm(d)
        self.gate_f = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.gate_e = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())

    def forward(self, F_feat, E, edge_to_faces, face_mask, edge_mask):
        B, N_e, d = E.shape
        N_f = F_feat.shape[1]
        f1_idx = edge_to_faces[:, :, 0].clamp(0, N_f - 1).long()
        f2_idx = edge_to_faces[:, :, 1].clamp(0, N_f - 1).long()
        valid_edge = edge_mask.bool() & (edge_to_faces[:, :, 0] >= 0) & (edge_to_faces[:, :, 1] >= 0)
        f1 = torch.gather(F_feat, 1, f1_idx.unsqueeze(-1).expand(-1, -1, d))
        f2 = torch.gather(F_feat, 1, f2_idx.unsqueeze(-1).expand(-1, -1, d))
        msg_e = self.face_to_edge(torch.cat([E, f1, f2], dim=-1))
        msg_e = msg_e * valid_edge.unsqueeze(-1).float()
        gate_e = self.gate_e(torch.cat([E, msg_e], dim=-1))
        E_new = self.norm_e(E + gate_e * msg_e)
        face_msg = torch.zeros_like(F_feat)
        face_count = torch.zeros(B, N_f, 1, device=F_feat.device)
        edge_contrib = E_new * valid_edge.unsqueeze(-1).float()
        count_contrib = valid_edge.unsqueeze(-1).float()
        face_msg.scatter_add_(1, f1_idx.unsqueeze(-1).expand(-1, -1, d), edge_contrib)
        face_count.scatter_add_(1, f1_idx.unsqueeze(-1), count_contrib)
        face_msg.scatter_add_(1, f2_idx.unsqueeze(-1).expand(-1, -1, d), edge_contrib)
        face_count.scatter_add_(1, f2_idx.unsqueeze(-1), count_contrib)
        face_msg = face_msg / (face_count + 1e-8)
        msg_f = self.edge_to_face(torch.cat([F_feat, face_msg], dim=-1))
        gate_f = self.gate_f(torch.cat([F_feat, msg_f], dim=-1))
        F_new = self.norm_f(F_feat + gate_f * msg_f * face_mask.unsqueeze(-1).float())
        return F_new, E_new


class TopologyBRepEncoder(nn.Module):
    """B-Rep encoder: EdgeMessageLayer x3 + HybridBRepTransformer (6 layers)."""
    def __init__(self, config):
        super().__init__()
        d = config.D_MODEL
        self.face_proj = nn.Sequential(
            nn.Linear(config.D_FACE, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(config.DROPOUT))
        self.edge_proj = nn.Sequential(
            nn.Linear(config.D_EDGE, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(config.DROPOUT))
        self.face_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.edge_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.level_emb = nn.Embedding(config.MAX_BFS_LEVELS, d)
        self.msg_layers = nn.ModuleList([EdgeMessageLayer(d, config.DROPOUT) for _ in range(config.NUM_MSG_LAYERS)])
        self.transformer = HybridBRepTransformer(d, config.NUM_HEADS, n_streams=3, dropout=config.DROPOUT)
        self.norm = nn.LayerNorm(d)

    def forward(self, face_feats, edge_feats, face_mask, edge_mask, edge_to_faces, bfs_level):
        F_feat = self.face_proj(face_feats.float()) + self.face_type
        F_feat = F_feat + self.level_emb(bfs_level.clamp(0, 31).long())
        E = self.edge_proj(edge_feats.float()) + self.edge_type
        for layer in self.msg_layers:
            F_feat, E = layer(F_feat, E, edge_to_faces, face_mask, edge_mask)
        X = torch.cat([F_feat, E], dim=1)
        mask = torch.cat([face_mask, edge_mask], dim=1)
        X = self.transformer(X, mask=mask)
        X = self.norm(X)
        X = torch.nan_to_num(X, nan=0.0)
        return X, mask

print("TopologyBRepEncoder defined!")'''

CELL_BREP_ENCODER_SHEAF = '''\
# Cell: Sheaf B-Rep Encoder (SheafBRepLayer + HybridBRepTransformer)

class SheafBRepLayer(nn.Module):
    """Sheaf neural network layer for B-Rep face-adjacency graphs.
    Learns diagonal restriction maps for anisotropic diffusion.
    Handles heterophilic adjacency (planar face next to cylindrical fillet).
    """
    def __init__(self, d_model, d_stalk=64, dropout=0.1):
        super().__init__()
        self.d_stalk = d_stalk
        self.proj_in = nn.Linear(d_model, d_stalk)
        self.proj_out = nn.Linear(d_stalk, d_model)
        self.restriction_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_stalk))
        self.W1 = nn.Linear(d_stalk, d_stalk)
        self.W2 = nn.Linear(d_stalk, d_stalk)
        self.norm = nn.LayerNorm(d_stalk)
        self.gate = nn.Sequential(nn.Linear(d_stalk * 2, d_stalk), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, F_feat, edge_to_faces, face_mask, edge_mask):
        B, N_f, d = F_feat.shape
        N_e = edge_to_faces.shape[1]
        X = self.proj_in(F_feat)
        f1_idx = edge_to_faces[:, :, 0].clamp(0, N_f - 1).long()
        f2_idx = edge_to_faces[:, :, 1].clamp(0, N_f - 1).long()
        valid = edge_mask.bool() & (edge_to_faces[:, :, 0] >= 0) & (edge_to_faces[:, :, 1] >= 0)
        f1_feat = torch.gather(F_feat, 1, f1_idx.unsqueeze(-1).expand(-1, -1, d))
        f2_feat = torch.gather(F_feat, 1, f2_idx.unsqueeze(-1).expand(-1, -1, d))
        R = self.restriction_mlp(torch.cat([f1_feat, f2_feat], dim=-1))
        R = R * valid.unsqueeze(-1).float()
        X_f1 = torch.gather(X, 1, f1_idx.unsqueeze(-1).expand(-1, -1, self.d_stalk))
        X_f2 = torch.gather(X, 1, f2_idx.unsqueeze(-1).expand(-1, -1, self.d_stalk))
        diff = X_f1 - X_f2
        weighted_diff = R * diff
        face_update = torch.zeros_like(X)
        update_count = torch.zeros(B, N_f, 1, device=X.device)
        edge_msg = R * weighted_diff * valid.unsqueeze(-1).float()
        count_one = valid.unsqueeze(-1).float()
        face_update.scatter_add_(1, f1_idx.unsqueeze(-1).expand(-1, -1, self.d_stalk), edge_msg)
        face_update.scatter_add_(1, f2_idx.unsqueeze(-1).expand(-1, -1, self.d_stalk), -edge_msg)
        update_count.scatter_add_(1, f1_idx.unsqueeze(-1), count_one)
        update_count.scatter_add_(1, f2_idx.unsqueeze(-1), count_one)
        face_update = face_update / (update_count + 1e-8)
        transformed = self.W2(F.gelu(self.W1(face_update)))
        transformed = self.dropout(transformed)
        gate_val = self.gate(torch.cat([X, transformed], dim=-1))
        X_new = self.norm(X - gate_val * transformed)
        return self.proj_out(X_new) * face_mask.unsqueeze(-1).float()


class SheafBRepEncoder(nn.Module):
    """B-Rep encoder: SheafBRepLayer x2 + HybridBRepTransformer (6 layers)."""
    def __init__(self, config):
        super().__init__()
        d = config.D_MODEL
        self.face_proj = nn.Sequential(
            nn.Linear(config.D_FACE, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(config.DROPOUT))
        self.edge_proj = nn.Sequential(
            nn.Linear(config.D_EDGE, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(config.DROPOUT))
        self.face_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.edge_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.level_emb = nn.Embedding(config.MAX_BFS_LEVELS, d)
        d_stalk = getattr(config, 'D_STALK', 64)
        n_sheaf = getattr(config, 'NUM_SHEAF_LAYERS', 2)
        self.sheaf_layers = nn.ModuleList([SheafBRepLayer(d, d_stalk, config.DROPOUT) for _ in range(n_sheaf)])
        self.transformer = HybridBRepTransformer(d, config.NUM_HEADS, n_streams=3, dropout=config.DROPOUT)
        self.norm = nn.LayerNorm(d)

    def forward(self, face_feats, edge_feats, face_mask, edge_mask, edge_to_faces, bfs_level):
        F_feat = self.face_proj(face_feats.float()) + self.face_type
        F_feat = F_feat + self.level_emb(bfs_level.clamp(0, 31).long())
        E = self.edge_proj(edge_feats.float()) + self.edge_type
        # Sheaf message passing on face-adjacency graph
        for sheaf in self.sheaf_layers:
            F_feat = sheaf(F_feat, edge_to_faces, face_mask, edge_mask)
        # Concatenate face+edge tokens, run hybrid transformer
        X = torch.cat([F_feat, E], dim=1)
        mask = torch.cat([face_mask, edge_mask], dim=1)
        X = self.transformer(X, mask=mask)
        X = self.norm(X)
        X = torch.nan_to_num(X, nan=0.0)
        return X, mask

print("SheafBRepEncoder defined!")'''

# ============================================================
# Text Encoder + Projection Heads
# ============================================================

CELL_TEXT_ENCODER = '''\
# Cell: Text Encoder + Projection Head

class TextEncoder(nn.Module):
    """Text encoder: 3072 -> 768 -> 384 with HybridTextTransformer."""
    def __init__(self, config):
        super().__init__()
        d = config.D_MODEL
        d_hidden = config.D_TEXT_HIDDEN
        self.proj_stage1 = nn.Sequential(
            nn.Linear(config.D_TEXT, d_hidden), nn.LayerNorm(d_hidden),
            nn.GELU(), nn.Dropout(config.DROPOUT))
        self.proj_stage2 = nn.Sequential(
            nn.Linear(d_hidden, d), nn.LayerNorm(d))
        self.encoder = HybridTextTransformer(d, config.NUM_HEADS, n_streams=3, dropout=config.DROPOUT)
        self.norm = nn.LayerNorm(d)

    def forward(self, X, mask=None):
        X = self.proj_stage1(X.float())
        X = self.proj_stage2(X)
        X = self.encoder(X, mask=mask)
        X = self.norm(X)
        X = torch.nan_to_num(X, nan=0.0)
        return X, mask


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_in), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_in, d_out))
    def forward(self, x):
        return self.proj(x)

print("TextEncoder and ProjectionHead defined!")'''

# ============================================================
# Candidate-specific components
# ============================================================

CELL_VIB_COMPONENTS = '''\
# Cell: VIB Components (Variational Information Bottleneck)

class VIBProjectionHead(nn.Module):
    """Projection head that outputs (mu, log_var) for VIB. Replaces ProjectionHead."""
    def __init__(self, d_in, d_proj, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_in, d_in), nn.GELU(), nn.Dropout(dropout))
        self.mu_head = nn.Linear(d_in, d_proj)
        self.logvar_head = nn.Linear(d_in, d_proj)
        nn.init.constant_(self.logvar_head.bias, -5.0)

    def forward(self, x, sample=True):
        h = self.shared(x)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h)
        if sample and self.training:
            std = (0.5 * log_var).exp()
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, log_var

def kl_divergence(mu, log_var):
    """KL(N(mu, sigma^2) || N(0, I)), closed form."""
    return 0.5 * torch.mean(torch.sum(mu.pow(2) + log_var.exp() - 1 - log_var, dim=-1))

print("VIB components defined!")'''

CELL_SIGREG_COMPONENTS = '''\
# Cell: SIGReg + Predictive Auxiliary Components

class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (Epps-Pulley statistic)."""
    def __init__(self, d_model, n_projections=512):
        super().__init__()
        directions = torch.randn(d_model, n_projections)
        directions = F.normalize(directions, dim=0)
        self.register_buffer('directions', directions)

    def epps_pulley(self, h):
        h = (h - h.mean()) / (h.std() + 1e-8)
        N = h.shape[0]
        diff = h.unsqueeze(0) - h.unsqueeze(1)
        pairwise = torch.exp(-0.5 * diff.pow(2)).sum() * (2.0 / (N * N))
        marginal = torch.exp(-0.25 * h.pow(2)).sum() * (2.0 * 1.41421356 / N)
        return pairwise - marginal + 1.0

    def forward(self, Z):
        projections = Z @ self.directions
        # Vectorized over first 64 projections for speed, loop the rest
        n_proj = projections.shape[1]
        stats = []
        for m in range(min(n_proj, 64)):
            stats.append(self.epps_pulley(projections[:, m]))
        return torch.stack(stats).mean()


class CrossModalPredictor(nn.Module):
    """Predicts z_brep from z_text. 2-layer MLP with stop-gradient on target."""
    def __init__(self, d_proj, hidden_mult=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_proj, d_proj * hidden_mult), nn.GELU(),
            nn.Linear(d_proj * hidden_mult, d_proj))
    def forward(self, z_text):
        return self.net(z_text)

print("SIGReg + CrossModalPredictor defined!")'''

CELL_HIERARCHICAL_COMPONENTS = '''\
# Cell: Hierarchical Nested Contrastive Components

class HierarchicalProjectionHead(nn.Module):
    """Three projection heads for hierarchical latent space: 128, 256, 384 dims."""
    def __init__(self, d_in, levels=[128, 256, 384], dropout=0.1):
        super().__init__()
        self.levels = levels
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_in, l))
            for l in levels
        ])
    def forward(self, x):
        return [head(x) for head in self.heads]


def nested_dropout_mask(d_proj, levels=[128, 256, 384], probs=[0.3, 0.3, 0.4]):
    """Sample a truncation level for nested dropout."""
    level = random.choices(levels, weights=probs, k=1)[0]
    mask = torch.ones(d_proj)
    mask[level:] = 0.0
    return mask, level

print("Hierarchical Nested Contrastive components defined!")'''

# ============================================================
# Loss functions (per candidate)
# ============================================================

CELL_LOSS_BASE = '''\
# Cell: Base Loss Functions

def info_nce_loss(features_a, features_b, temperature=0.07):
    features_a = F.normalize(features_a, dim=1)
    features_b = F.normalize(features_b, dim=1)
    labels = torch.arange(len(features_a), device=features_a.device)
    logits_a2b = torch.mm(features_a, features_b.t()) / temperature
    logits_b2a = torch.mm(features_b, features_a.t()) / temperature
    return (F.cross_entropy(logits_a2b, labels) + F.cross_entropy(logits_b2a, labels)) / 2.0

def uniformity_loss(z, t=2.0):
    z = F.normalize(z, dim=-1)
    sq_pdist = torch.cdist(z, z, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

def variance_loss(z, min_var=0.5):
    var = z.var(dim=0).mean()
    return F.relu(min_var - var)

def compute_margins(z_text, z_brep, z_pc):
    """Compute contrastive margins for diagnostics."""
    B = z_text.shape[0]
    neg_mask = ~torch.eye(B, dtype=torch.bool, device=z_text.device)
    sim_tb = z_text @ z_brep.T
    sim_tp = z_text @ z_pc.T
    sim_bp = z_brep @ z_pc.T
    m_tb = sim_tb.diag().mean() - sim_tb[neg_mask].mean()
    m_tp = sim_tp.diag().mean() - sim_tp[neg_mask].mean()
    m_bp = sim_bp.diag().mean() - sim_bp[neg_mask].mean()
    return {'margin_tb': m_tb, 'margin_tp': m_tp, 'margin_bp': m_bp,
            'margin': (m_tb + m_tp + m_bp) / 3}

print("Base loss functions defined!")'''


def make_loss_cell(candidate):
    """Generate candidate-specific loss function cell."""
    if candidate == 'a':
        return '''\
# Cell: Candidate A Loss — VIB-Contrastive
def clip4cad_loss(outputs, config, stage=1):
    losses = {}
    tau = outputs['tau']
    z_brep = F.normalize(outputs['z_brep'], dim=-1)
    z_pc = F.normalize(outputs['z_pc'], dim=-1)
    B = z_brep.shape[0]
    labels = torch.arange(B, device=z_brep.device)
    ls = config.LABEL_SMOOTHING
    logits_b2p = (z_brep.float() @ z_pc.float().T) / tau
    loss_b2p = (F.cross_entropy(logits_b2p, labels, label_smoothing=ls) +
                F.cross_entropy(logits_b2p.T, labels, label_smoothing=ls)) / 2
    losses['infonce_b2p'] = loss_b2p
    # VIB KL terms
    kl_brep = kl_divergence(outputs['mu_brep'], outputs['logvar_brep'])
    kl_pc = kl_divergence(outputs['mu_pc'], outputs['logvar_pc'])
    losses['kl_brep'] = kl_brep
    losses['kl_pc'] = kl_pc
    if stage >= 1 and 'z_text' in outputs:
        z_text = F.normalize(outputs['z_text'], dim=-1)
        logits_t2b = (z_text.float() @ z_brep.float().T) / tau
        loss_t2b = (F.cross_entropy(logits_t2b, labels, label_smoothing=ls) +
                    F.cross_entropy(logits_t2b.T, labels, label_smoothing=ls)) / 2
        logits_t2p = (z_text.float() @ z_pc.float().T) / tau
        loss_t2p = (F.cross_entropy(logits_t2p, labels, label_smoothing=ls) +
                    F.cross_entropy(logits_t2p.T, labels, label_smoothing=ls)) / 2
        losses['infonce_t2b'] = loss_t2b
        losses['infonce_t2p'] = loss_t2p
        tw = config.TEXT_LOSS_WEIGHT
        losses['infonce'] = (tw * loss_t2b + tw * loss_t2p + loss_b2p) / (2 * tw + 1)
        kl_text = kl_divergence(outputs['mu_text'], outputs['logvar_text'])
        losses['kl_text'] = kl_text
        losses['uniformity'] = (uniformity_loss(z_text) + uniformity_loss(z_brep) + uniformity_loss(z_pc)) / 3
        losses['variance'] = (variance_loss(z_text) + variance_loss(z_brep)) / 2
        with torch.no_grad():
            losses.update(compute_margins(z_text, z_brep, z_pc))
    else:
        losses['infonce'] = loss_b2p
        losses['uniformity'] = uniformity_loss(z_brep)
        losses['variance'] = variance_loss(z_brep)
        kl_text = torch.tensor(0.0, device=z_brep.device)
    # Asymmetric beta (OMIB): BRep gets LESS compression (weaker encoder)
    beta_brep = config.BETA_MAX * 0.3
    beta_pc = config.BETA_MAX * 0.7
    beta_text = config.BETA_MAX * 0.1
    losses['total'] = (losses['infonce']
        + config.UNIFORMITY_WEIGHT * losses['uniformity']
        + config.VARIANCE_WEIGHT * losses['variance']
        + beta_brep * kl_brep + beta_pc * kl_pc + beta_text * kl_text)
    return losses['total'], losses

print("Candidate A loss defined!")'''

    elif candidate == 'b':
        return '''\
# Cell: Candidate B Loss — Standard InfoNCE (novelty is in the Sheaf encoder)
def clip4cad_loss(outputs, config, stage=1):
    losses = {}
    tau = outputs['tau']
    z_brep = F.normalize(outputs['z_brep'], dim=-1)
    z_pc = F.normalize(outputs['z_pc'], dim=-1)
    B = z_brep.shape[0]
    labels = torch.arange(B, device=z_brep.device)
    ls = config.LABEL_SMOOTHING
    logits_b2p = (z_brep.float() @ z_pc.float().T) / tau
    loss_b2p = (F.cross_entropy(logits_b2p, labels, label_smoothing=ls) +
                F.cross_entropy(logits_b2p.T, labels, label_smoothing=ls)) / 2
    losses['infonce_b2p'] = loss_b2p
    if stage >= 1 and 'z_text' in outputs:
        z_text = F.normalize(outputs['z_text'], dim=-1)
        logits_t2b = (z_text.float() @ z_brep.float().T) / tau
        loss_t2b = (F.cross_entropy(logits_t2b, labels, label_smoothing=ls) +
                    F.cross_entropy(logits_t2b.T, labels, label_smoothing=ls)) / 2
        logits_t2p = (z_text.float() @ z_pc.float().T) / tau
        loss_t2p = (F.cross_entropy(logits_t2p, labels, label_smoothing=ls) +
                    F.cross_entropy(logits_t2p.T, labels, label_smoothing=ls)) / 2
        losses['infonce_t2b'] = loss_t2b
        losses['infonce_t2p'] = loss_t2p
        tw = config.TEXT_LOSS_WEIGHT
        losses['infonce'] = (tw * loss_t2b + tw * loss_t2p + loss_b2p) / (2 * tw + 1)
        losses['uniformity'] = (uniformity_loss(z_text) + uniformity_loss(z_brep) + uniformity_loss(z_pc)) / 3
        losses['variance'] = (variance_loss(z_text) + variance_loss(z_brep)) / 2
        with torch.no_grad():
            losses.update(compute_margins(z_text, z_brep, z_pc))
    else:
        losses['infonce'] = loss_b2p
        losses['uniformity'] = uniformity_loss(z_brep)
        losses['variance'] = variance_loss(z_brep)
    losses['total'] = (losses['infonce']
        + config.UNIFORMITY_WEIGHT * losses['uniformity']
        + config.VARIANCE_WEIGHT * losses['variance'])
    return losses['total'], losses

print("Candidate B loss defined!")'''

    elif candidate == 'c':
        return '''\
# Cell: Candidate C Loss — Hierarchical Nested Contrastive
def clip4cad_loss(outputs, config, stage=1):
    losses = {}
    tau = outputs['tau']
    z_brep_full = outputs['z_brep']  # 384-dim
    z_pc_full = outputs['z_pc']      # 384-dim
    B = z_brep_full.shape[0]
    labels = torch.arange(B, device=z_brep_full.device)
    ls = config.LABEL_SMOOTHING
    # BRep <-> PC at full dimension
    z_b = F.normalize(z_brep_full, dim=-1)
    z_p = F.normalize(z_pc_full, dim=-1)
    logits_b2p = (z_b.float() @ z_p.float().T) / tau
    loss_b2p = (F.cross_entropy(logits_b2p, labels, label_smoothing=ls) +
                F.cross_entropy(logits_b2p.T, labels, label_smoothing=ls)) / 2
    losses['infonce_b2p'] = loss_b2p
    if stage >= 1 and 'z_text_levels' in outputs:
        text_levels = outputs['z_text_levels']  # list of [B,128], [B,256], [B,384]
        levels = config.HIERARCHY_LEVELS
        weights = config.HIERARCHY_WEIGHTS
        # Nested dropout: sample truncation level
        mask, trunc_level = nested_dropout_mask(levels[-1], levels, config.NESTED_DROPOUT_PROBS)
        mask = mask.to(z_brep_full.device)
        z_brep_masked = z_brep_full * mask.unsqueeze(0)
        z_pc_masked = z_pc_full * mask.unsqueeze(0)
        # Hierarchical InfoNCE at each level
        total_hier = 0.0
        for li, (level, w) in enumerate(zip(levels, weights)):
            z_t_l = F.normalize(text_levels[li], dim=-1)
            z_b_l = F.normalize(z_brep_masked[:, :level], dim=-1)
            z_p_l = F.normalize(z_pc_masked[:, :level], dim=-1)
            logits_tb = (z_t_l.float() @ z_b_l.float().T) / tau
            loss_tb = (F.cross_entropy(logits_tb, labels, label_smoothing=ls) +
                       F.cross_entropy(logits_tb.T, labels, label_smoothing=ls)) / 2
            logits_tp = (z_t_l.float() @ z_p_l.float().T) / tau
            loss_tp = (F.cross_entropy(logits_tp, labels, label_smoothing=ls) +
                       F.cross_entropy(logits_tp.T, labels, label_smoothing=ls)) / 2
            total_hier += w * (loss_tb + loss_tp) / 2
            losses[f'infonce_l{level}'] = (loss_tb + loss_tp) / 2
        losses['infonce'] = total_hier / sum(weights) + 0.5 * loss_b2p
        z_text_full = F.normalize(text_levels[-1], dim=-1)
        losses['uniformity'] = (uniformity_loss(z_text_full) + uniformity_loss(z_b) + uniformity_loss(z_p)) / 3
        losses['variance'] = (variance_loss(z_text_full) + variance_loss(z_b)) / 2
        with torch.no_grad():
            losses.update(compute_margins(z_text_full, z_b, z_p))
            losses['trunc_level'] = torch.tensor(float(trunc_level))
    else:
        losses['infonce'] = loss_b2p
        losses['uniformity'] = uniformity_loss(z_b)
        losses['variance'] = variance_loss(z_b)
    losses['total'] = (losses['infonce']
        + config.UNIFORMITY_WEIGHT * losses['uniformity']
        + config.VARIANCE_WEIGHT * losses['variance'])
    return losses['total'], losses

print("Candidate C loss defined!")'''

    elif candidate == 'd':
        return '''\
# Cell: Candidate D Loss — SIGReg + Predictive Auxiliary
def clip4cad_loss(outputs, config, stage=1):
    losses = {}
    tau = outputs['tau']
    z_brep = F.normalize(outputs['z_brep'], dim=-1)
    z_pc = F.normalize(outputs['z_pc'], dim=-1)
    B = z_brep.shape[0]
    labels = torch.arange(B, device=z_brep.device)
    ls = config.LABEL_SMOOTHING
    logits_b2p = (z_brep.float() @ z_pc.float().T) / tau
    loss_b2p = (F.cross_entropy(logits_b2p, labels, label_smoothing=ls) +
                F.cross_entropy(logits_b2p.T, labels, label_smoothing=ls)) / 2
    losses['infonce_b2p'] = loss_b2p
    if stage >= 1 and 'z_text' in outputs:
        z_text = F.normalize(outputs['z_text'], dim=-1)
        logits_t2b = (z_text.float() @ z_brep.float().T) / tau
        loss_t2b = (F.cross_entropy(logits_t2b, labels, label_smoothing=ls) +
                    F.cross_entropy(logits_t2b.T, labels, label_smoothing=ls)) / 2
        logits_t2p = (z_text.float() @ z_pc.float().T) / tau
        loss_t2p = (F.cross_entropy(logits_t2p, labels, label_smoothing=ls) +
                    F.cross_entropy(logits_t2p.T, labels, label_smoothing=ls)) / 2
        losses['infonce_t2b'] = loss_t2b
        losses['infonce_t2p'] = loss_t2p
        tw = config.TEXT_LOSS_WEIGHT
        losses['infonce'] = (tw * loss_t2b + tw * loss_t2p + loss_b2p) / (2 * tw + 1)
        losses['uniformity'] = (uniformity_loss(z_text) + uniformity_loss(z_brep) + uniformity_loss(z_pc)) / 3
        losses['variance'] = (variance_loss(z_text) + variance_loss(z_brep)) / 2
        # SIGReg on concatenated embeddings
        sigreg_module = outputs.get('sigreg_module')
        if sigreg_module is not None:
            all_z = torch.cat([outputs['z_brep_raw'], outputs['z_pc_raw'], outputs['z_text_raw']], dim=0)
            losses['sigreg'] = sigreg_module(all_z)
        else:
            losses['sigreg'] = torch.tensor(0.0, device=z_brep.device)
        # Predictive auxiliary: predict z_brep from z_text
        if 'z_brep_hat' in outputs:
            losses['pred_mse'] = F.mse_loss(outputs['z_brep_hat'], outputs['z_brep_raw'].detach())
        else:
            losses['pred_mse'] = torch.tensor(0.0, device=z_brep.device)
        with torch.no_grad():
            losses.update(compute_margins(z_text, z_brep, z_pc))
    else:
        losses['infonce'] = loss_b2p
        losses['uniformity'] = uniformity_loss(z_brep)
        losses['variance'] = variance_loss(z_brep)
        losses['sigreg'] = torch.tensor(0.0, device=z_brep.device)
        losses['pred_mse'] = torch.tensor(0.0, device=z_brep.device)
    losses['total'] = (losses['infonce']
        + config.UNIFORMITY_WEIGHT * losses['uniformity']
        + config.VARIANCE_WEIGHT * losses['variance']
        + config.LAMBDA_SIG * losses['sigreg']
        + config.LAMBDA_PRED * losses['pred_mse'])
    return losses['total'], losses

print("Candidate D loss defined!")'''

    elif candidate == 'e':
        return '''\
# Cell: Candidate E Loss — Full Stack (VIB + Hierarchical + SIGReg + Predictive)
def clip4cad_loss(outputs, config, stage=0):
    losses = {}
    tau = outputs['tau']
    z_brep_full = outputs['z_brep']
    z_pc_full = outputs['z_pc']
    B = z_brep_full.shape[0]
    labels = torch.arange(B, device=z_brep_full.device)
    ls = config.LABEL_SMOOTHING
    z_b = F.normalize(z_brep_full, dim=-1)
    z_p = F.normalize(z_pc_full, dim=-1)
    logits_b2p = (z_b.float() @ z_p.float().T) / tau
    loss_b2p = (F.cross_entropy(logits_b2p, labels, label_smoothing=ls) +
                F.cross_entropy(logits_b2p.T, labels, label_smoothing=ls)) / 2
    losses['infonce_b2p'] = loss_b2p
    # VIB KL (active in stages 1+2)
    kl_brep = kl_divergence(outputs['mu_brep'], outputs['logvar_brep']) if 'mu_brep' in outputs else torch.tensor(0.0, device=z_b.device)
    kl_pc = kl_divergence(outputs['mu_pc'], outputs['logvar_pc']) if 'mu_pc' in outputs else torch.tensor(0.0, device=z_b.device)
    losses['kl_brep'] = kl_brep
    losses['kl_pc'] = kl_pc
    if stage >= 1 and 'z_text_levels' in outputs:
        text_levels = outputs['z_text_levels']
        z_text_full = F.normalize(text_levels[-1], dim=-1)
        if stage >= 2:
            # Hierarchical loss with nested dropout
            levels = config.HIERARCHY_LEVELS
            weights = config.HIERARCHY_WEIGHTS
            mask, trunc_level = nested_dropout_mask(levels[-1], levels, config.NESTED_DROPOUT_PROBS)
            mask = mask.to(z_brep_full.device)
            z_brep_masked = z_brep_full * mask.unsqueeze(0)
            z_pc_masked = z_pc_full * mask.unsqueeze(0)
            total_hier = 0.0
            for li, (level, w) in enumerate(zip(levels, weights)):
                z_t_l = F.normalize(text_levels[li], dim=-1)
                z_b_l = F.normalize(z_brep_masked[:, :level], dim=-1)
                z_p_l = F.normalize(z_pc_masked[:, :level], dim=-1)
                logits_tb = (z_t_l.float() @ z_b_l.float().T) / tau
                loss_tb = (F.cross_entropy(logits_tb, labels, label_smoothing=ls) +
                           F.cross_entropy(logits_tb.T, labels, label_smoothing=ls)) / 2
                logits_tp = (z_t_l.float() @ z_p_l.float().T) / tau
                loss_tp = (F.cross_entropy(logits_tp, labels, label_smoothing=ls) +
                           F.cross_entropy(logits_tp.T, labels, label_smoothing=ls)) / 2
                total_hier += w * (loss_tb + loss_tp) / 2
            losses['infonce'] = total_hier / sum(weights) + 0.5 * loss_b2p
        else:
            # Flat InfoNCE (stage 1)
            logits_t2b = (z_text_full.float() @ z_b.float().T) / tau
            loss_t2b = (F.cross_entropy(logits_t2b, labels, label_smoothing=ls) +
                        F.cross_entropy(logits_t2b.T, labels, label_smoothing=ls)) / 2
            logits_t2p = (z_text_full.float() @ z_p.float().T) / tau
            loss_t2p = (F.cross_entropy(logits_t2p, labels, label_smoothing=ls) +
                        F.cross_entropy(logits_t2p.T, labels, label_smoothing=ls)) / 2
            tw = config.TEXT_LOSS_WEIGHT
            losses['infonce'] = (tw * loss_t2b + tw * loss_t2p + loss_b2p) / (2 * tw + 1)
        kl_text = kl_divergence(outputs['mu_text'], outputs['logvar_text']) if 'mu_text' in outputs else torch.tensor(0.0, device=z_b.device)
        losses['kl_text'] = kl_text
        losses['uniformity'] = (uniformity_loss(z_text_full) + uniformity_loss(z_b) + uniformity_loss(z_p)) / 3
        losses['variance'] = (variance_loss(z_text_full) + variance_loss(z_b)) / 2
        # SIGReg (stages 1+2)
        sigreg_module = outputs.get('sigreg_module')
        if sigreg_module is not None:
            all_z = torch.cat([outputs.get('z_brep_raw', z_brep_full), outputs.get('z_pc_raw', z_pc_full), outputs.get('z_text_raw', text_levels[-1])], dim=0)
            losses['sigreg'] = sigreg_module(all_z)
        else:
            losses['sigreg'] = torch.tensor(0.0, device=z_b.device)
        # Predictive (stage 2 only)
        if stage >= 2 and 'z_brep_hat' in outputs:
            losses['pred_mse'] = F.mse_loss(outputs['z_brep_hat'], outputs.get('z_brep_raw', z_brep_full).detach())
        else:
            losses['pred_mse'] = torch.tensor(0.0, device=z_b.device)
        with torch.no_grad():
            losses.update(compute_margins(z_text_full, z_b, z_p))
        # Staged beta
        beta_max = config.BETA_MAX if stage >= 1 else 0.0
        if stage >= 2:
            beta_max = config.BETA_MAX * 10  # increase in stage 2
        beta_brep = beta_max * 0.3
        beta_pc = beta_max * 0.7
        beta_text = beta_max * 0.1
    else:
        losses['infonce'] = loss_b2p
        losses['uniformity'] = uniformity_loss(z_b)
        losses['variance'] = variance_loss(z_b)
        losses['sigreg'] = torch.tensor(0.0, device=z_b.device)
        losses['pred_mse'] = torch.tensor(0.0, device=z_b.device)
        kl_text = torch.tensor(0.0, device=z_b.device)
        beta_brep = beta_pc = beta_text = 0.0
    losses['total'] = (losses['infonce']
        + config.UNIFORMITY_WEIGHT * losses['uniformity']
        + config.VARIANCE_WEIGHT * losses['variance']
        + beta_brep * kl_brep + beta_pc * kl_pc + beta_text * kl_text
        + config.LAMBDA_SIG * losses['sigreg']
        + config.LAMBDA_PRED * losses['pred_mse'])
    return losses['total'], losses

print("Candidate E loss defined!")'''


# ============================================================
# Model classes (per candidate)
# ============================================================

def make_model_cell(candidate):
    """Generate candidate-specific model cell."""
    if candidate == 'a':
        return '''\
# Cell: Candidate A Model — VIB-Contrastive
class CLIP4CADModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.D_MODEL
        self.text_encoder = TextEncoder(config)
        self.brep_encoder = TopologyBRepEncoder(config)
        self.pc_encoder = DGCNNEncoder(latent_size=config.D_PC, k=config.DGCNN_K)
        self.pc_proj_layer = nn.Sequential(
            nn.Linear(config.D_PC, d), nn.LayerNorm(d), nn.GELU(),
            nn.Dropout(config.DROPOUT), nn.Linear(d, d), nn.LayerNorm(d))
        self.text_pool = AttentionPooling(d, config.NUM_POOL_QUERIES, num_heads=4, dropout=config.DROPOUT)
        self.brep_pool = AttentionPooling(d, config.NUM_POOL_QUERIES, num_heads=4, dropout=config.DROPOUT)
        # VIB projection heads (output mu + logvar)
        self.text_proj = VIBProjectionHead(d, config.D_PROJ, config.DROPOUT)
        self.brep_proj = VIBProjectionHead(d, config.D_PROJ, config.DROPOUT)
        self.pc_proj = VIBProjectionHead(d, config.D_PROJ, config.DROPOUT)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(config.TEMPERATURE)))

    @property
    def tau(self): return self.log_tau.exp().clamp(0.01, 1.0)

    def forward(self, batch, stage=1):
        device = next(self.parameters()).device
        pc_feat = self.pc_encoder(batch['point_cloud'].to(device))
        pc_feat = self.pc_proj_layer(pc_feat)
        z_pc, mu_pc, logvar_pc = self.pc_proj(pc_feat)
        X_brep, brep_mask = self.brep_encoder(
            batch['face_features'].to(device), batch['edge_features'].to(device),
            batch['face_mask'].to(device), batch['edge_mask'].to(device),
            batch['edge_to_faces'].to(device), batch['bfs_level'].to(device))
        brep_pooled = self.brep_pool(X_brep, brep_mask)
        z_brep, mu_brep, logvar_brep = self.brep_proj(brep_pooled)
        result = {'z_brep': z_brep, 'z_pc': z_pc, 'tau': self.tau,
                  'mu_brep': mu_brep, 'logvar_brep': logvar_brep,
                  'mu_pc': mu_pc, 'logvar_pc': logvar_pc}
        if stage >= 1:
            X_text, text_mask = self.text_encoder(batch['text_features'].to(device), batch['text_mask'].to(device))
            text_pooled = self.text_pool(X_text, text_mask)
            z_text, mu_text, logvar_text = self.text_proj(text_pooled)
            result.update({'z_text': z_text, 'mu_text': mu_text, 'logvar_text': logvar_text})
        return result

print("Candidate A model defined!")'''

    elif candidate == 'b':
        return '''\
# Cell: Candidate B Model — Sheaf B-Rep Encoder
class CLIP4CADModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.D_MODEL
        self.text_encoder = TextEncoder(config)
        self.brep_encoder = SheafBRepEncoder(config)  # Sheaf instead of Topology
        self.pc_encoder = DGCNNEncoder(latent_size=config.D_PC, k=config.DGCNN_K)
        self.pc_proj_layer = nn.Sequential(
            nn.Linear(config.D_PC, d), nn.LayerNorm(d), nn.GELU(),
            nn.Dropout(config.DROPOUT), nn.Linear(d, d), nn.LayerNorm(d))
        self.text_pool = AttentionPooling(d, config.NUM_POOL_QUERIES, num_heads=4, dropout=config.DROPOUT)
        self.brep_pool = AttentionPooling(d, config.NUM_POOL_QUERIES, num_heads=4, dropout=config.DROPOUT)
        self.text_proj = ProjectionHead(d, config.D_PROJ, config.DROPOUT)
        self.brep_proj = ProjectionHead(d, config.D_PROJ, config.DROPOUT)
        self.pc_proj = ProjectionHead(d, config.D_PROJ, config.DROPOUT)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(config.TEMPERATURE)))

    @property
    def tau(self): return self.log_tau.exp().clamp(0.01, 1.0)

    def forward(self, batch, stage=1):
        device = next(self.parameters()).device
        pc_feat = self.pc_encoder(batch['point_cloud'].to(device))
        z_pc = self.pc_proj(self.pc_proj_layer(pc_feat))
        X_brep, brep_mask = self.brep_encoder(
            batch['face_features'].to(device), batch['edge_features'].to(device),
            batch['face_mask'].to(device), batch['edge_mask'].to(device),
            batch['edge_to_faces'].to(device), batch['bfs_level'].to(device))
        z_brep = self.brep_proj(self.brep_pool(X_brep, brep_mask))
        result = {'z_brep': z_brep, 'z_pc': z_pc, 'tau': self.tau}
        if stage >= 1:
            X_text, text_mask = self.text_encoder(batch['text_features'].to(device), batch['text_mask'].to(device))
            z_text = self.text_proj(self.text_pool(X_text, text_mask))
            result['z_text'] = z_text
        return result

print("Candidate B model defined!")'''

    elif candidate == 'c':
        return '''\
# Cell: Candidate C Model — Hierarchical Nested Contrastive
class CLIP4CADModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.D_MODEL
        d_proj = config.D_PROJ  # 384 for hierarchical
        self.text_encoder = TextEncoder(config)
        self.brep_encoder = TopologyBRepEncoder(config)
        self.pc_encoder = DGCNNEncoder(latent_size=config.D_PC, k=config.DGCNN_K)
        self.pc_proj_layer = nn.Sequential(
            nn.Linear(config.D_PC, d), nn.LayerNorm(d), nn.GELU(),
            nn.Dropout(config.DROPOUT), nn.Linear(d, d), nn.LayerNorm(d))
        self.text_pool = AttentionPooling(d, config.NUM_POOL_QUERIES, num_heads=4, dropout=config.DROPOUT)
        self.brep_pool = AttentionPooling(d, config.NUM_POOL_QUERIES, num_heads=4, dropout=config.DROPOUT)
        # BRep/PC project to full 384-dim
        self.brep_proj = ProjectionHead(d, d_proj, config.DROPOUT)
        self.pc_proj = ProjectionHead(d, d_proj, config.DROPOUT)
        # Text: 3 hierarchical projection heads (128, 256, 384)
        self.text_hier_proj = HierarchicalProjectionHead(d, config.HIERARCHY_LEVELS, config.DROPOUT)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(config.TEMPERATURE)))

    @property
    def tau(self): return self.log_tau.exp().clamp(0.01, 1.0)

    def forward(self, batch, stage=1):
        device = next(self.parameters()).device
        pc_feat = self.pc_encoder(batch['point_cloud'].to(device))
        z_pc = self.pc_proj(self.pc_proj_layer(pc_feat))
        X_brep, brep_mask = self.brep_encoder(
            batch['face_features'].to(device), batch['edge_features'].to(device),
            batch['face_mask'].to(device), batch['edge_mask'].to(device),
            batch['edge_to_faces'].to(device), batch['bfs_level'].to(device))
        z_brep = self.brep_proj(self.brep_pool(X_brep, brep_mask))
        result = {'z_brep': z_brep, 'z_pc': z_pc, 'tau': self.tau}
        if stage >= 1:
            X_text, text_mask = self.text_encoder(batch['text_features'].to(device), batch['text_mask'].to(device))
            text_pooled = self.text_pool(X_text, text_mask)
            z_text_levels = self.text_hier_proj(text_pooled)
            result['z_text_levels'] = z_text_levels
            result['z_text'] = z_text_levels[-1]  # full 384-dim for eval
        return result

print("Candidate C model defined!")'''

    elif candidate == 'd':
        return '''\
# Cell: Candidate D Model — SIGReg + Predictive Auxiliary
class CLIP4CADModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.D_MODEL
        self.text_encoder = TextEncoder(config)
        self.brep_encoder = TopologyBRepEncoder(config)
        self.pc_encoder = DGCNNEncoder(latent_size=config.D_PC, k=config.DGCNN_K)
        self.pc_proj_layer = nn.Sequential(
            nn.Linear(config.D_PC, d), nn.LayerNorm(d), nn.GELU(),
            nn.Dropout(config.DROPOUT), nn.Linear(d, d), nn.LayerNorm(d))
        self.text_pool = AttentionPooling(d, config.NUM_POOL_QUERIES, num_heads=4, dropout=config.DROPOUT)
        self.brep_pool = AttentionPooling(d, config.NUM_POOL_QUERIES, num_heads=4, dropout=config.DROPOUT)
        self.text_proj = ProjectionHead(d, config.D_PROJ, config.DROPOUT)
        self.brep_proj = ProjectionHead(d, config.D_PROJ, config.DROPOUT)
        self.pc_proj = ProjectionHead(d, config.D_PROJ, config.DROPOUT)
        # SIGReg + Predictor
        self.sigreg = SIGReg(config.D_PROJ, getattr(config, 'SIGREG_PROJECTIONS', 512))
        self.predictor = CrossModalPredictor(config.D_PROJ)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(config.TEMPERATURE)))

    @property
    def tau(self): return self.log_tau.exp().clamp(0.01, 1.0)

    def forward(self, batch, stage=1):
        device = next(self.parameters()).device
        pc_feat = self.pc_encoder(batch['point_cloud'].to(device))
        z_pc_raw = self.pc_proj(self.pc_proj_layer(pc_feat))
        X_brep, brep_mask = self.brep_encoder(
            batch['face_features'].to(device), batch['edge_features'].to(device),
            batch['face_mask'].to(device), batch['edge_mask'].to(device),
            batch['edge_to_faces'].to(device), batch['bfs_level'].to(device))
        z_brep_raw = self.brep_proj(self.brep_pool(X_brep, brep_mask))
        result = {'z_brep': z_brep_raw, 'z_pc': z_pc_raw, 'tau': self.tau,
                  'z_brep_raw': z_brep_raw, 'z_pc_raw': z_pc_raw, 'sigreg_module': self.sigreg}
        if stage >= 1:
            X_text, text_mask = self.text_encoder(batch['text_features'].to(device), batch['text_mask'].to(device))
            z_text_raw = self.text_proj(self.text_pool(X_text, text_mask))
            z_brep_hat = self.predictor(z_text_raw)
            result.update({'z_text': z_text_raw, 'z_text_raw': z_text_raw, 'z_brep_hat': z_brep_hat})
        return result

print("Candidate D model defined!")'''

    elif candidate == 'e':
        return '''\
# Cell: Candidate E Model — Full Stack (Sheaf + VIB + Hierarchical + SIGReg + Predictor)
class CLIP4CADModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.D_MODEL
        d_proj = config.D_PROJ  # 384
        self.text_encoder = TextEncoder(config)
        self.brep_encoder = SheafBRepEncoder(config)
        self.pc_encoder = DGCNNEncoder(latent_size=config.D_PC, k=config.DGCNN_K)
        self.pc_proj_layer = nn.Sequential(
            nn.Linear(config.D_PC, d), nn.LayerNorm(d), nn.GELU(),
            nn.Dropout(config.DROPOUT), nn.Linear(d, d), nn.LayerNorm(d))
        self.text_pool = AttentionPooling(d, config.NUM_POOL_QUERIES, num_heads=4, dropout=config.DROPOUT)
        self.brep_pool = AttentionPooling(d, config.NUM_POOL_QUERIES, num_heads=4, dropout=config.DROPOUT)
        # VIB projection heads outputting 384-dim
        self.brep_proj = VIBProjectionHead(d, d_proj, config.DROPOUT)
        self.pc_proj = VIBProjectionHead(d, d_proj, config.DROPOUT)
        # Hierarchical text projection heads
        self.text_hier_proj = HierarchicalProjectionHead(d, config.HIERARCHY_LEVELS, config.DROPOUT)
        # SIGReg + Predictor
        self.sigreg = SIGReg(d_proj, getattr(config, 'SIGREG_PROJECTIONS', 512))
        self.predictor = CrossModalPredictor(d_proj)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(config.TEMPERATURE)))

    @property
    def tau(self): return self.log_tau.exp().clamp(0.01, 1.0)

    def forward(self, batch, stage=0):
        device = next(self.parameters()).device
        pc_feat = self.pc_encoder(batch['point_cloud'].to(device))
        z_pc, mu_pc, logvar_pc = self.pc_proj(self.pc_proj_layer(pc_feat))
        X_brep, brep_mask = self.brep_encoder(
            batch['face_features'].to(device), batch['edge_features'].to(device),
            batch['face_mask'].to(device), batch['edge_mask'].to(device),
            batch['edge_to_faces'].to(device), batch['bfs_level'].to(device))
        z_brep, mu_brep, logvar_brep = self.brep_proj(self.brep_pool(X_brep, brep_mask))
        result = {'z_brep': z_brep, 'z_pc': z_pc, 'tau': self.tau,
                  'mu_brep': mu_brep, 'logvar_brep': logvar_brep,
                  'mu_pc': mu_pc, 'logvar_pc': logvar_pc,
                  'z_brep_raw': z_brep, 'z_pc_raw': z_pc, 'sigreg_module': self.sigreg}
        if stage >= 1:
            X_text, text_mask = self.text_encoder(batch['text_features'].to(device), batch['text_mask'].to(device))
            text_pooled = self.text_pool(X_text, text_mask)
            z_text_levels = self.text_hier_proj(text_pooled)
            z_brep_hat = self.predictor(z_text_levels[-1])
            result.update({'z_text_levels': z_text_levels, 'z_text': z_text_levels[-1],
                           'z_text_raw': z_text_levels[-1], 'z_brep_hat': z_brep_hat,
                           'mu_text': z_text_levels[-1], 'logvar_text': torch.zeros_like(z_text_levels[-1])})
        return result

print("Candidate E model defined!")'''


# ============================================================
# Metrics, training utilities, dataset, loaders, training loop
# ============================================================

CELL_METRICS = '''\
# Cell: Metrics Calculation
def calc_metrics(sim_matrix, k_values=[1, 5, 10]):
    sim = sim_matrix.cpu().numpy() if isinstance(sim_matrix, torch.Tensor) else sim_matrix
    n = sim.shape[0]
    sorted_idx = np.argsort(-sim, axis=1)
    metrics = {}
    for k in k_values:
        recalls, aps = [], []
        for i in range(n):
            top_k = sorted_idx[i, :k]
            hit = int(i in top_k)
            recalls.append(hit)
            aps.append(1.0 / (np.where(top_k == i)[0][0] + 1) if hit else 0.0)
        metrics[f'recall@{k}'] = np.mean(recalls) * 100
        metrics[f'mAP@{k}'] = np.mean(aps) * 100
    return metrics

@torch.no_grad()
def evaluate_clip4cad(model, loader, device, k_values=[1, 5, 10]):
    model.eval()
    all_z_text, all_z_brep, all_z_pc = [], [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        outputs = model(batch, stage=1)
        z_text = outputs.get('z_text', outputs.get('z_text_levels', [None])[-1] if 'z_text_levels' in outputs else None)
        if z_text is not None:
            all_z_text.append(z_text.cpu())
        all_z_brep.append(outputs['z_brep'].cpu())
        all_z_pc.append(outputs['z_pc'].cpu())
    z_brep = F.normalize(torch.cat(all_z_brep), dim=1)
    z_pc = F.normalize(torch.cat(all_z_pc), dim=1)
    metrics = {}
    b2p = calc_metrics(z_brep @ z_pc.T, k_values)
    for k, v in b2p.items(): metrics[f'brep2pc_{k}'] = v
    if all_z_text:
        z_text = F.normalize(torch.cat(all_z_text), dim=1)
        # Truncate to same dim if needed
        d = min(z_text.shape[1], z_brep.shape[1])
        t2b = calc_metrics(z_text[:, :d] @ z_brep[:, :d].T, k_values)
        for k, v in t2b.items(): metrics[f'text2brep_{k}'] = v
        d = min(z_text.shape[1], z_pc.shape[1])
        t2p = calc_metrics(z_text[:, :d] @ z_pc[:, :d].T, k_values)
        for k, v in t2p.items(): metrics[f'text2pc_{k}'] = v
    return metrics

print("Metrics functions defined!")'''

CELL_TRAIN_UTILS = '''\
# Cell: Training & Checkpoint Utilities
def train_epoch(model, loader, optimizer, scheduler, device, config, stage=1):
    model.train()
    total_loss = 0
    total_margin = 0
    pbar = tqdm(loader, desc=f'Training (Stage {stage})', leave=False)
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(batch, stage=stage)
            loss, loss_dict = clip4cad_loss(outputs, config, stage=stage)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        if scheduler is not None: scheduler.step()
        total_loss += loss.item()
        margin = loss_dict.get('margin', torch.tensor(0.0))
        total_margin += margin.item() if isinstance(margin, torch.Tensor) else margin
        if i % 20 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'margin': f'{margin.item() if isinstance(margin, torch.Tensor) else margin:.4f}'})
    n = len(loader)
    return total_loss / n, total_margin / n

@torch.no_grad()
def validate_epoch(model, loader, device, config, stage=1):
    model.eval()
    total_loss = 0
    total_margin = 0
    for batch in loader:
        with torch.amp.autocast('cuda'):
            outputs = model(batch, stage=stage)
            loss, loss_dict = clip4cad_loss(outputs, config, stage=stage)
        total_loss += loss.item()
        margin = loss_dict.get('margin', torch.tensor(0.0))
        total_margin += margin.item() if isinstance(margin, torch.Tensor) else margin
    n = len(loader)
    return total_loss / n, total_margin / n

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_dir, filename='checkpoint.pth'):
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics}
    if scheduler is not None: state['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state, os.path.join(save_dir, filename))

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if not os.path.exists(checkpoint_path): return 0
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint.get('model_state_dict', checkpoint).items()}
    model.load_state_dict(state_dict, strict=False)
    if optimizer and 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0) + 1

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"Memory cleaned. GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("Training utilities defined!")'''

CELL_DATASET = '''\
# Cell: CLIP4CAD Dataset (multi-file text H5 support)

class CLIP4CADDataset(Dataset):
    """Trimodal dataset: PC (raw PLY) + B-Rep (HDF5) + Text (multi-file HDF5)."""
    def __init__(self, df, config, brep_uid_to_idx, text_uid_map, cache_ply=True):
        self.config = config
        self.data_root = config.DATA_ROOT
        self.samples = df.reset_index(drop=True)
        self.brep_uid_to_idx = brep_uid_to_idx
        self.text_uid_map = text_uid_map  # {uid_str: (h5_path, index)}
        self.brep_h5 = h5py.File(config.BREP_H5_PATH, 'r')
        # Open all unique text H5 files
        self.text_h5_handles = {}
        for uid_str, (h5_path, idx) in text_uid_map.items():
            if h5_path not in self.text_h5_handles:
                self.text_h5_handles[h5_path] = h5py.File(h5_path, 'r')
        # Cache PLY in RAM
        self.cache_ply = cache_ply
        if cache_ply:
            print(f"Caching PLY data ({len(self.samples)} samples)...")
            self.ply_cache = {}
            for idx in tqdm(range(len(self.samples)), desc="Caching PLY"):
                row = self.samples.iloc[idx]
                try:
                    ply_path = os.path.join(self.data_root, row['ply_path'])
                    xyz = load_ply_fast(ply_path)
                    xyz = xyz - xyz.mean(0)
                    norm = np.max(np.linalg.norm(xyz, axis=1))
                    if norm > 0: xyz = xyz / norm
                    self.ply_cache[idx] = xyz
                except:
                    self.ply_cache[idx] = np.zeros((10000, 3), dtype=np.float32)
            mem_gb = sum(p.nbytes for p in self.ply_cache.values()) / 1e9
            print(f"Cached {len(self.ply_cache)} PLY files ({mem_gb:.1f} GB)")
        print(f"CLIP4CADDataset ready: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        row = self.samples.iloc[i]
        uid_str = str(row['uid'])
        # Point Cloud
        if self.cache_ply:
            pc = self.ply_cache[i]
        else:
            ply_path = os.path.join(self.data_root, row['ply_path'])
            pc = load_ply_fast(ply_path)
            pc = pc - pc.mean(0)
            norm = np.max(np.linalg.norm(pc, axis=1))
            if norm > 0: pc = pc / norm
        idx_pts = np.random.choice(len(pc), self.config.NUM_POINTS, replace=len(pc) < self.config.NUM_POINTS)
        pc = torch.from_numpy(pc[idx_pts]).T
        # B-Rep
        brep_idx = self.brep_uid_to_idx[uid_str]
        face_features = torch.from_numpy(self.brep_h5['face_features'][brep_idx].astype(np.float32))
        edge_features = torch.from_numpy(self.brep_h5['edge_features'][brep_idx].astype(np.float32))
        face_mask = torch.from_numpy(self.brep_h5['face_masks'][brep_idx].astype(np.float32))
        edge_mask = torch.from_numpy(self.brep_h5['edge_masks'][brep_idx].astype(np.float32))
        edge_to_faces = torch.from_numpy(self.brep_h5['edge_to_faces'][brep_idx].astype(np.int64))
        bfs_level = torch.from_numpy(self.brep_h5['bfs_level'][brep_idx].astype(np.int64))
        # Text (multi-file)
        h5_path, text_idx = self.text_uid_map[uid_str]
        h5 = self.text_h5_handles[h5_path]
        text_features = torch.from_numpy(h5['desc_embeddings'][text_idx].astype(np.float32))
        text_mask = torch.from_numpy(h5['desc_masks'][text_idx].astype(np.float32))
        return {'point_cloud': pc, 'face_features': face_features, 'edge_features': edge_features,
                'face_mask': face_mask, 'edge_mask': edge_mask, 'edge_to_faces': edge_to_faces,
                'bfs_level': bfs_level, 'text_features': text_features, 'text_mask': text_mask, 'idx': i}

    def __del__(self):
        if hasattr(self, 'brep_h5') and self.brep_h5: self.brep_h5.close()
        if hasattr(self, 'text_h5_handles'):
            for h5 in self.text_h5_handles.values(): h5.close()

print("CLIP4CADDataset defined!")'''

CELL_CREATE_LOADERS = '''\
# Cell: Create Datasets & DataLoaders
print("Creating datasets...")
train_dataset = CLIP4CADDataset(trainval_df_tri, config, brep_uid_to_idx, text_uid_to_source, cache_ply=True)
eval_dataset = CLIP4CADDataset(eval_df_tri, config, brep_uid_to_idx,
    test_text_uid_to_source if test_text_uid_to_source else all_text_uid_to_source, cache_ply=True)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
    num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
eval_loader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
    num_workers=config.NUM_WORKERS, pin_memory=True)

print(f"\\nDataLoaders created:")
print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"  Eval:  {len(eval_dataset)} samples, {len(eval_loader)} batches")
print(f"  Batch size: {config.BATCH_SIZE}")'''


def make_device_cell(candidate):
    """Generate model instantiation + optimizer cell."""
    return f'''\
# Cell: Device Setup + Model Creation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {{device}}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {{i}}: {{torch.cuda.get_device_name(i)}}")

model = CLIP4CADModel(config).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using DataParallel with {{torch.cuda.device_count()}} GPUs")
    model = nn.DataParallel(model)

params = count_parameters(model)
print(f"\\nModel parameters:")
print(f"  Total: {{params['total']:,}}")
print(f"  Trainable: {{params['trainable']:,}}")

core_model = model.module if hasattr(model, 'module') else model
text_params = list(core_model.text_encoder.parameters()) + list(core_model.text_pool.parameters())
# Add text projection params (handle different head types)
for attr in ['text_proj', 'text_hier_proj']:
    if hasattr(core_model, attr):
        text_params += list(getattr(core_model, attr).parameters())
text_param_ids = set(id(p) for p in text_params)
other_params = [p for p in core_model.parameters() if id(p) not in text_param_ids and p.requires_grad]

optimizer = optim.AdamW([
    {{'params': other_params, 'lr': config.LEARNING_RATE, 'weight_decay': config.WEIGHT_DECAY}},
    {{'params': text_params, 'lr': config.LEARNING_RATE * config.TEXT_LR_MULT, 'weight_decay': config.WEIGHT_DECAY}},
])

print(f"\\nOptimizer: AdamW")
print(f"  Base LR: {{config.LEARNING_RATE}}")
print(f"  Text LR: {{config.LEARNING_RATE * config.TEXT_LR_MULT}}")'''


def make_training_loop(candidate):
    """Generate training loop cell."""
    model_name = {
        'a': 'CLIP4CAD-CandA-VIB',
        'b': 'CLIP4CAD-CandB-Sheaf',
        'c': 'CLIP4CAD-CandC-Hierarchical',
        'd': 'CLIP4CAD-CandD-SIGReg',
        'e': 'CLIP4CAD-CandE-FullStack',
    }[candidate]

    stage_logic = ''
    if candidate == 'e':
        stage_logic = '''    if epoch < config.STAGE0_EPOCHS:
        stage = 0
    elif epoch < config.STAGE0_EPOCHS + config.STAGE1_EPOCHS:
        stage = 1
    else:
        stage = 2'''
    else:
        stage_logic = '    stage = 0 if epoch < config.STAGE0_EPOCHS else 1'

    return f'''\
# Cell: Training Loop
model_name = '{model_name}'
save_dir = os.path.join(config.OUTPUT_DIR, model_name.replace(' ', '_'))
os.makedirs(save_dir, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs')) if config.ENABLE_TENSORBOARD else None

total_steps = config.NUM_EPOCHS * len(train_loader)
warmup_steps = config.WARMUP_EPOCHS * len(train_loader)

def lr_lambda(step):
    if step < warmup_steps: return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.01 + 0.5 * 0.99 * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
metrics_history = []
best_recall = 0.0

print(f"\\n{{'='*80}}")
print(f"Training: {{model_name}}")
print(f"{{'='*80}}")

for epoch in range(config.NUM_EPOCHS):
{stage_logic}
    print(f"\\nEpoch {{epoch+1}}/{{config.NUM_EPOCHS}} [Stage {{stage}}]")

    train_loss, train_margin = train_epoch(model, train_loader, optimizer, scheduler, device, config, stage=stage)
    val_loss, val_margin = validate_epoch(model, eval_loader, device, config, stage=stage)

    retrieval_metrics = {{}}
    if stage >= 1:
        retrieval_metrics = evaluate_clip4cad(model, eval_loader, device, config.K_VALUES)

    print(f"  Train Loss: {{train_loss:.4f}}, Margin: {{train_margin:.4f}}")
    print(f"  Val   Loss: {{val_loss:.4f}}, Margin: {{val_margin:.4f}}")
    if retrieval_metrics:
        print(f"  Text->BRep R@1: {{retrieval_metrics.get('text2brep_recall@1', 0):.2f}}%")
        print(f"  Text->PC  R@1: {{retrieval_metrics.get('text2pc_recall@1', 0):.2f}}%")
        print(f"  BRep->PC  R@1: {{retrieval_metrics.get('brep2pc_recall@1', 0):.2f}}%")

    epoch_metrics = {{'epoch': epoch + 1, 'stage': stage,
                     'train_loss': train_loss, 'val_loss': val_loss,
                     'train_margin': train_margin, 'val_margin': val_margin,
                     **{{f'val_{{k}}': v for k, v in retrieval_metrics.items()}}}}
    metrics_history.append(epoch_metrics)

    if writer:
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Margin/train', train_margin, epoch)
        writer.add_scalar('Margin/val', val_margin, epoch)
        for k, v in retrieval_metrics.items(): writer.add_scalar(f'Val/{{k}}', v, epoch)

    r1 = retrieval_metrics.get('text2brep_recall@1', 0)
    if r1 > best_recall:
        best_recall = r1
        save_checkpoint(model, optimizer, scheduler, epoch, retrieval_metrics, save_dir, 'best.pth')
        print(f"  ** New best Text->BRep R@1: {{best_recall:.2f}}% **")

    save_checkpoint(model, optimizer, scheduler, epoch, retrieval_metrics, save_dir, 'latest.pth')
    if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
        save_checkpoint(model, optimizer, scheduler, epoch, retrieval_metrics, save_dir, f'epoch_{{epoch+1}}.pth')

    if stage >= 1 and epoch > config.STAGE0_EPOCHS + 3 and val_margin < 0.01:
        print("  !! WARNING: Margin near 0 - possible collapse!")

metrics_df = pd.DataFrame(metrics_history)
metrics_df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
if writer: writer.close()

print(f"\\n{{'='*80}}")
print(f"{{model_name}} training complete!")
print(f"Best Text->BRep R@1: {{best_recall:.2f}}%")
print(f"{{'='*80}}")'''


CELL_FINAL_EVAL = '''\
# Cell: Final Evaluation
print(f"\\nFinal evaluation on test set")
print("=" * 80)

best_path = os.path.join(save_dir, 'best.pth')
if os.path.exists(best_path):
    load_checkpoint(model, None, None, best_path)
    print(f"Loaded best checkpoint")

final_metrics = evaluate_clip4cad(model, eval_loader, device, config.K_VALUES)
print(f"\\nFinal Retrieval Results:")
print("=" * 50)
for direction in ['text2brep', 'text2pc', 'brep2pc']:
    print(f"\\n  {direction}:")
    for k in config.K_VALUES:
        r = final_metrics.get(f'{direction}_recall@{k}', 0)
        m = final_metrics.get(f'{direction}_mAP@{k}', 0)
        print(f"    R@{k}: {r:.2f}%  mAP@{k}: {m:.2f}%")'''

CELL_COPY_DRIVE = '''\
# Cell: Copy Checkpoints to Google Drive
import shutil
DRIVE_SAVE_DIR = f"/content/drive/MyDrive/MMCAD/clip4cad_checkpoints/{model_name}"
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
print("Copying checkpoints to Drive...")
for fname in ['best.pth', 'latest.pth', 'metrics.csv']:
    src = os.path.join(save_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(DRIVE_SAVE_DIR, fname))
        print(f"  Copied {fname}")
print(f"\\nCheckpoints saved to: {DRIVE_SAVE_DIR}")'''

CELL_CLEANUP = '''\
# Cell: Cleanup
cleanup_memory()'''


# ============================================================
# Config cells per candidate
# ============================================================

def make_config_cell(candidate):
    base = '''\
# Cell: Configuration
class Config:
    DATA_ROOT = "/content/mmcad_data"
    CSV_PATH = os.path.join(DATA_ROOT, "abc_dataset_clean.csv")
    BREP_H5_PATH = os.path.join(DATA_ROOT, "brep_autobrep.h5")
    TRAIN_TEXT_H5 = "/content/drive/MyDrive/MMCAD/train_text_embeddings.h5"
    VAL_TEXT_H5 = "/content/drive/MyDrive/MMCAD/val_text_embeddings.h5"
    TEST_TEXT_H5 = "/content/drive/MyDrive/MMCAD/test_text_embeddings.h5"
    OUTPUT_DIR = "/content/clip4cad_checkpoints"
    SPLIT_DIR = "/content/mmcad_splits"
    TRAIN_RATIO = 0.80; VAL_RATIO = 0.10; TEST_RATIO = 0.10; RANDOM_SEED = 42
    BATCH_SIZE = 128; LEARNING_RATE = 1e-4; TEXT_LR_MULT = 3.0
    WEIGHT_DECAY = 0.01; WARMUP_EPOCHS = 2; TEMPERATURE = 0.07
    GRAD_CLIP = 1.0; LABEL_SMOOTHING = 0.01
    UNIFORMITY_WEIGHT = 0.5; VARIANCE_WEIGHT = 0.1; TEXT_LOSS_WEIGHT = 1.5
    D_FACE = 48; D_EDGE = 12; D_TEXT = 3072; D_PC = 1024
    D_MODEL = 384; D_TEXT_HIDDEN = 768
    NUM_POOL_QUERIES = 16; NUM_HEADS = 8; DROPOUT = 0.1
    NUM_MSG_LAYERS = 3; NUM_BREP_TF_LAYERS = 6; NUM_TEXT_TF_LAYERS = 4
    MAX_BFS_LEVELS = 32; MAX_FACES = 192; MAX_EDGES = 512; DGCNN_K = 20
    NUM_POINTS = 2048; SAVE_EVERY_N_EPOCHS = 5; NUM_WORKERS = 4
    K_VALUES = [1, 5, 10]; SUBSET_SIZE = None; ENABLE_TENSORBOARD = True'''

    extras = {
        'a': '''
    # VIB-specific
    D_PROJ = 256
    NUM_EPOCHS = 30; STAGE0_EPOCHS = 8
    BETA_MAX = 1e-3''',
        'b': '''
    # Sheaf-specific
    D_PROJ = 256
    NUM_EPOCHS = 30; STAGE0_EPOCHS = 8
    D_STALK = 64; NUM_SHEAF_LAYERS = 2''',
        'c': '''
    # Hierarchical-specific
    D_PROJ = 384  # Full hierarchical dimension
    NUM_EPOCHS = 30; STAGE0_EPOCHS = 8
    HIERARCHY_LEVELS = [128, 256, 384]
    HIERARCHY_WEIGHTS = [1.0, 1.0, 1.5]
    NESTED_DROPOUT_PROBS = [0.3, 0.3, 0.4]''',
        'd': '''
    # SIGReg-specific
    D_PROJ = 256
    NUM_EPOCHS = 30; STAGE0_EPOCHS = 8
    LAMBDA_SIG = 0.1; LAMBDA_PRED = 0.01; SIGREG_PROJECTIONS = 512''',
        'e': '''
    # Full Stack
    D_PROJ = 384; D_STALK = 64; NUM_SHEAF_LAYERS = 2
    NUM_EPOCHS = 35; STAGE0_EPOCHS = 8; STAGE1_EPOCHS = 12; STAGE2_EPOCHS = 15
    BETA_MAX = 1e-4
    HIERARCHY_LEVELS = [128, 256, 384]
    HIERARCHY_WEIGHTS = [1.0, 1.0, 1.5]
    NESTED_DROPOUT_PROBS = [0.3, 0.3, 0.4]
    LAMBDA_SIG = 0.1; LAMBDA_PRED = 0.01; SIGREG_PROJECTIONS = 512''',
    }

    footer = '''

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.SPLIT_DIR, exist_ok=True)
print(f"Config: D_MODEL={config.D_MODEL}, D_PROJ={config.D_PROJ}, Batch={config.BATCH_SIZE}")
print(f"Epochs: {config.NUM_EPOCHS}, Stage0: {config.STAGE0_EPOCHS}")'''

    return base + extras[candidate] + footer


# ============================================================
# Notebook builder
# ============================================================

CANDIDATE_INFO = {
    'a': {
        'name': 'VIB-Contrastive',
        'desc': 'Variational Information Bottleneck compression + asymmetric KL (OMIB-inspired)',
        'components': 'mHC-lite + Gated DeltaNet Hybrid + VIB Projection Heads',
    },
    'b': {
        'name': 'Sheaf B-Rep Encoder',
        'desc': 'Sheaf neural network message passing on B-Rep face-adjacency graph',
        'components': 'mHC-lite + Gated DeltaNet Hybrid + Sheaf Laplacian Diffusion',
    },
    'c': {
        'name': 'Hierarchical Nested Contrastive',
        'desc': 'Nested dropout with 3-level text hierarchy (128/256/384 dim)',
        'components': 'mHC-lite + Gated DeltaNet Hybrid + Hierarchical Projection + Nested Dropout',
    },
    'd': {
        'name': 'SIGReg + Predictive Auxiliary',
        'desc': 'Epps-Pulley isotropic Gaussian regularizer + cross-modal predictor',
        'components': 'mHC-lite + Gated DeltaNet Hybrid + SIGReg + CrossModalPredictor',
    },
    'e': {
        'name': 'Full Stack (A+B+C+D)',
        'desc': 'All components combined with 3-stage curriculum training',
        'components': 'Sheaf + VIB + Hierarchical + SIGReg + Predictor + 3-stage curriculum',
    },
}

def build_notebook(candidate):
    info = CANDIDATE_INFO[candidate]
    cells = []

    # Title
    cells.append(make_md_cell(
        f"# CLIP4CAD Candidate {candidate.upper()}: {info['name']}\n\n"
        f"**Architecture:** {info['desc']}\n\n"
        f"**Components:** {info['components']}\n\n"
        f"**Data:** Train+Val combined for training, Test for evaluation\n\n"
        f"**Hardware:** 2x A100 GPUs (Colab)\n\n"
        f"**Source:** `clip4cad_architecture_candidates_v2.md`"))

    # Shared setup
    cells.append(make_code_cell(CELL_INSTALL))
    cells.append(make_code_cell(CELL_IMPORTS))
    cells.append(make_code_cell(CELL_MOUNT))
    cells.append(make_code_cell(CELL_DATA))
    cells.append(make_code_cell(make_config_cell(candidate)))
    cells.append(make_code_cell(CELL_SPLIT))
    cells.append(make_code_cell(CELL_UID_INDEX))
    cells.append(make_code_cell(CELL_DGCNN))
    cells.append(make_code_cell(CELL_PLY))
    cells.append(make_code_cell(CELL_ATTN_POOL))
    cells.append(make_code_cell(CELL_INFRASTRUCTURE))

    # B-Rep encoder
    if candidate in ['b', 'e']:
        cells.append(make_code_cell(CELL_BREP_ENCODER_SHEAF))
    else:
        cells.append(make_code_cell(CELL_BREP_ENCODER_TOPOLOGY))

    # Text encoder
    cells.append(make_code_cell(CELL_TEXT_ENCODER))

    # Candidate-specific components
    if candidate in ['a', 'e']:
        cells.append(make_code_cell(CELL_VIB_COMPONENTS))
    if candidate in ['c', 'e']:
        cells.append(make_code_cell(CELL_HIERARCHICAL_COMPONENTS))
    if candidate in ['d', 'e']:
        cells.append(make_code_cell(CELL_SIGREG_COMPONENTS))

    # Model + Loss
    cells.append(make_code_cell(make_model_cell(candidate)))
    cells.append(make_code_cell(CELL_LOSS_BASE))
    cells.append(make_code_cell(make_loss_cell(candidate)))
    cells.append(make_code_cell(CELL_METRICS))
    cells.append(make_code_cell(CELL_TRAIN_UTILS))
    cells.append(make_code_cell(CELL_DATASET))
    cells.append(make_code_cell(CELL_CREATE_LOADERS))

    # Training
    cells.append(make_md_cell("---\n# Training"))
    cells.append(make_code_cell(make_device_cell(candidate)))
    cells.append(make_code_cell(make_training_loop(candidate)))
    cells.append(make_code_cell(CELL_FINAL_EVAL))
    cells.append(make_code_cell(CELL_COPY_DRIVE))
    cells.append(make_code_cell(CELL_CLEANUP))

    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU", "gpuClass": "standard"
        },
        "cells": cells
    }


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    os.makedirs('notebooks', exist_ok=True)

    for cand_id in ['a', 'b', 'c', 'd', 'e']:
        info = CANDIDATE_INFO[cand_id]
        short_names = {'a': 'vib', 'b': 'sheaf', 'c': 'hierarchical', 'd': 'sigreg', 'e': 'fullstack'}
        filename = f"notebooks/clip4cad_candidate_{cand_id}_{short_names[cand_id]}.ipynb"
        nb = build_notebook(cand_id)
        with open(filename, 'w') as f:
            json.dump(nb, f, indent=1)
        n_cells = len(nb['cells'])
        print(f"Generated: {filename} ({n_cells} cells)")

    print(f"\nAll 5 notebooks generated successfully!")
