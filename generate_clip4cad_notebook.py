"""Generate the CLIP4CAD Colab training notebook."""
import json

def make_code_cell(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")],
        "execution_count": None,
        "outputs": []
    }

def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

cells = []

# ============================================================
# Cell 0: Title (markdown)
# ============================================================
cells.append(make_md_cell("""\
# CLIP4CAD: 3-Way Contrastive CAD Representation Learning

**Architecture:** v4.9 - Direct contrastive alignment (no codebook)
- **B-Rep branch:** Pre-extracted AutoBrep FSQ features + Topology Transformer + AttentionPooling
- **Point Cloud branch:** DGCNN encoder (same as baseline notebooks)
- **Text branch:** Pre-extracted Phi-4-mini features + Transformer + AttentionPooling

**Loss:** 3-way symmetric InfoNCE (Text <-> BRep <-> PC)

**Comparison:** Same DGCNN + metrics as `mmcad_training_colab.ipynb` for fair comparison against BERT-DGCNN, CLIP-DGCNN baselines.

**Data requirements (upload to Drive):**
- `abc_dataset_clean.csv` (already there)
- `ply.tar.lz4` (already there)
- `brep_autobrep.h5` (pre-extracted with `scripts/precompute_brep_autobrep.py`)
- `text_embeddings.h5` (pre-extracted Phi-4-mini features)"""))

# ============================================================
# Cell 1: Install packages
# ============================================================
cells.append(make_code_cell("""\
# Cell 1: Install Packages
print("Installing packages...")

!pip install -q plyfile tensorboard h5py

print("Packages installed!")

# Check GPU
import torch
print(f"\\n{'='*60}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB")
print(f"{'='*60}")"""))

# ============================================================
# Cell 2: Imports
# ============================================================
cells.append(make_code_cell("""\
# Cell 2: Imports
import os, sys, gc, time, math, warnings
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

print("Libraries imported!")"""))

# ============================================================
# Cell 3: Mount Drive
# ============================================================
cells.append(make_code_cell("""\
# Cell 3: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

print("\\nGoogle Drive mounted at /content/drive/")"""))

# ============================================================
# Cell 4: Extract/Copy Data
# ============================================================
cells.append(make_code_cell("""\
# Cell 4: Extract/Copy Data
import os, time

EXTRACT_DIR = "/content/mmcad_data"
os.makedirs(EXTRACT_DIR, exist_ok=True)

# --- Point Clouds (lz4 archive) ---
print("="*60)
print("EXTRACTING POINT CLOUDS")
print("="*60)

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

# --- Pre-extracted B-Rep features (HDF5) ---
print("\\nCopying B-Rep features...")
BREP_H5_SRC = "/content/drive/MyDrive/MMCAD/brep_autobrep.h5"
BREP_H5_DST = f"{EXTRACT_DIR}/brep_autobrep.h5"
if os.path.exists(BREP_H5_DST):
    print(f"B-Rep HDF5 already exists")
else:
    !cp "{BREP_H5_SRC}" "{BREP_H5_DST}"
    print(f"B-Rep HDF5 copied!")

# --- Pre-extracted Text features (HDF5) ---
print("\\nCopying Text features...")
TEXT_H5_SRC = "/content/drive/MyDrive/MMCAD/text_embeddings.h5"
TEXT_H5_DST = f"{EXTRACT_DIR}/text_embeddings.h5"
if os.path.exists(TEXT_H5_DST):
    print(f"Text HDF5 already exists")
else:
    !cp "{TEXT_H5_SRC}" "{TEXT_H5_DST}"
    print(f"Text HDF5 copied!")

print("\\n" + "="*60)
print("ALL DATA READY!")
print("="*60)"""))

# ============================================================
# Cell 5: Configuration
# ============================================================
cells.append(make_code_cell("""\
# Cell 5: Configuration
class Config:
    # === DATA PATHS ===
    DATA_ROOT = "/content/mmcad_data"
    CSV_PATH = os.path.join(DATA_ROOT, "abc_dataset_clean.csv")
    BREP_H5_PATH = os.path.join(DATA_ROOT, "brep_autobrep.h5")
    TEXT_H5_PATH = os.path.join(DATA_ROOT, "text_embeddings.h5")
    OUTPUT_DIR = "/content/clip4cad_checkpoints"
    SPLIT_DIR = "/content/mmcad_splits"

    # === DATA SPLIT ===
    TRAIN_RATIO = 0.80
    VAL_RATIO = 0.10
    TEST_RATIO = 0.10
    RANDOM_SEED = 42

    # === TRAINING ===
    NUM_EPOCHS = 30          # Total epochs (Stage 0 + Stage 1)
    STAGE0_EPOCHS = 8        # BRep-PC anchoring
    BATCH_SIZE = 128         # A100 can handle this
    LEARNING_RATE = 1e-4
    TEXT_LR_MULT = 3.0       # Text encoder needs higher LR
    WEIGHT_DECAY = 0.01
    WARMUP_EPOCHS = 2
    TEMPERATURE = 0.07
    GRAD_CLIP = 1.0
    LABEL_SMOOTHING = 0.01
    UNIFORMITY_WEIGHT = 0.5
    VARIANCE_WEIGHT = 0.1
    TEXT_LOSS_WEIGHT = 1.5

    # === MODEL (v4.9 architecture) ===
    D_FACE = 48              # AutoBrep face FSQ features
    D_EDGE = 12              # AutoBrep edge FSQ features
    D_TEXT = 3072             # Phi-4-mini features
    D_PC = 1024              # DGCNN output
    D_MODEL = 384            # Internal unified dimension
    D_PROJ = 256             # Projection output dimension
    D_TEXT_HIDDEN = 768      # Intermediate text dimension
    NUM_POOL_QUERIES = 16
    NUM_HEADS = 8
    DROPOUT = 0.1
    NUM_MSG_LAYERS = 3       # Topology message passing
    NUM_BREP_TF_LAYERS = 6   # BRep transformer
    NUM_TEXT_TF_LAYERS = 4   # Text transformer
    MAX_BFS_LEVELS = 32
    MAX_FACES = 192
    MAX_EDGES = 512
    DGCNN_K = 20

    # === POINT CLOUD ===
    NUM_POINTS = 2048

    # === CHECKPOINT & LOGGING ===
    SAVE_EVERY_N_EPOCHS = 5
    NUM_WORKERS = 4
    K_VALUES = [1, 5, 10]

    # === SUBSET (for quick validation, set None for full dataset) ===
    SUBSET_SIZE = None       # Set to e.g. 5000 for smoke test
    ENABLE_TENSORBOARD = True

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.SPLIT_DIR, exist_ok=True)
print(f"Output: {config.OUTPUT_DIR}")
print(f"Batch size: {config.BATCH_SIZE}")
print(f"Epochs: {config.NUM_EPOCHS} (Stage 0: {config.STAGE0_EPOCHS})")
print(f"Model dim: {config.D_MODEL}, Proj dim: {config.D_PROJ}")"""))

# ============================================================
# Cell 6: CSV + Train/Val/Test Split
# ============================================================
cells.append(make_code_cell("""\
# Cell 6: Data Loading with Train/Val/Test Split
df = pd.read_csv(config.CSV_PATH)
print(f"Total entries in CSV: {len(df)}")

# Filter for complete entries
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
    with open(train_split_path) as f: train_uids = set(int(line.strip()) for line in f)
    with open(val_split_path) as f: val_uids = set(int(line.strip()) for line in f)
    with open(test_split_path) as f: test_uids = set(int(line.strip()) for line in f)
    train_df = valid_df[valid_df['uid'].isin(train_uids)]
    val_df = valid_df[valid_df['uid'].isin(val_uids)]
    test_df = valid_df[valid_df['uid'].isin(test_uids)]
else:
    print("Creating new train/val/test splits...")
    train_df, temp_df = train_test_split(valid_df, train_size=config.TRAIN_RATIO, random_state=config.RANDOM_SEED)
    val_ratio_adjusted = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio_adjusted, random_state=config.RANDOM_SEED)
    with open(train_split_path, 'w') as f:
        for uid in train_df['uid']: f.write(f"{uid}\\n")
    with open(val_split_path, 'w') as f:
        for uid in val_df['uid']: f.write(f"{uid}\\n")
    with open(test_split_path, 'w') as f:
        for uid in test_df['uid']: f.write(f"{uid}\\n")
    print(f"Split files saved to {config.SPLIT_DIR}")

print(f"\\nData Split:")
print(f"  Train: {len(train_df)} samples ({len(train_df)/len(valid_df)*100:.1f}%)")
print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(valid_df)*100:.1f}%)")
print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(valid_df)*100:.1f}%)")"""))

# ============================================================
# Cell 7: Build HDF5 UID index maps
# ============================================================
cells.append(make_code_cell("""\
# Cell 7: Build HDF5 UID Index Maps & Filter to Trimodal Samples
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

# --- Text features ---
text_h5 = h5py.File(config.TEXT_H5_PATH, 'r')
# Try different possible key names
text_uid_key = 'uids' if 'uids' in text_h5 else 'sample_ids'
text_feat_key = 'desc_embeddings' if 'desc_embeddings' in text_h5 else 'embeddings'
text_mask_key = 'desc_masks' if 'desc_masks' in text_h5 else 'masks'
text_uids_raw = text_h5[text_uid_key][:]
text_uid_to_idx = {}
for i, uid in enumerate(text_uids_raw):
    uid_str = uid.decode('utf-8') if isinstance(uid, bytes) else str(uid)
    text_uid_to_idx[uid_str] = i
print(f"\\nText features: {len(text_uid_to_idx)} samples")
print(f"  {text_feat_key}: {text_h5[text_feat_key].shape}")
text_h5.close()

# --- Filter to trimodal samples (have PLY + BRep + Text) ---
def has_ply(row):
    ply_path = os.path.join(config.DATA_ROOT, row['ply_path'])
    return os.path.isfile(ply_path)

print("\\nFiltering to trimodal samples...")
trimodal_mask_train = train_df['uid'].astype(str).isin(brep_uid_to_idx) & \\
                      train_df['uid'].astype(str).isin(text_uid_to_idx)
trimodal_mask_val = val_df['uid'].astype(str).isin(brep_uid_to_idx) & \\
                    val_df['uid'].astype(str).isin(text_uid_to_idx)
trimodal_mask_test = test_df['uid'].astype(str).isin(brep_uid_to_idx) & \\
                     test_df['uid'].astype(str).isin(text_uid_to_idx)

train_df_tri = train_df[trimodal_mask_train].reset_index(drop=True)
val_df_tri = val_df[trimodal_mask_val].reset_index(drop=True)
test_df_tri = test_df[trimodal_mask_test].reset_index(drop=True)

print(f"\\nTrimodal samples:")
print(f"  Train: {len(train_df_tri)} (from {len(train_df)})")
print(f"  Val:   {len(val_df_tri)} (from {len(val_df)})")
print(f"  Test:  {len(test_df_tri)} (from {len(test_df)})")"""))

# ============================================================
# Cell 8: DGCNN Encoder (verbatim from existing notebook)
# ============================================================
cells.append(make_code_cell("""\
# Cell 8: DGCNN Encoder (same as baseline notebooks)
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

print("DGCNN encoder defined!")"""))

# ============================================================
# Cell 9: PLY Loading (verbatim from existing notebook)
# ============================================================
cells.append(make_code_cell("""\
# Cell 9: PLY Loading (same as baseline notebooks)
def load_ply_fast(path):
    \"\"\"Fast PLY loading\"\"\"
    plydata = PlyData.read(path)
    return np.stack([plydata['vertex']['x'],
                    plydata['vertex']['y'],
                    plydata['vertex']['z']], axis=1).astype(np.float32)

print("PLY loader defined!")"""))

# ============================================================
# Cell 10: AttentionPooling (from v4.9)
# ============================================================
cells.append(make_code_cell("""\
# Cell 10: Attention Pooling (v4.9 - replaces codebook)
class AttentionPooling(nn.Module):
    \"\"\"
    Multi-layer cross-attention pooling with learnable queries.
    Replaces the codebook that caused 0% retrieval in v4.8.x.
    \"\"\"

    def __init__(self, d, num_queries=16, num_heads=8, dropout=0.1, num_layers=2):
        super().__init__()
        self.d = d
        self.num_queries = num_queries
        self.num_layers = num_layers

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

        for i in range(self.num_layers):
            attn_out, _ = self.cross_attn_layers[i](
                query=Q, key=X, value=X,
                key_padding_mask=key_padding_mask, need_weights=False
            )
            Q = self.norms1[i](Q + self.dropout(attn_out))
            Q = self.norms2[i](Q + self.dropout(self.ffn_layers[i](Q)))

        self_attn_out, _ = self.self_attn(Q, Q, Q, need_weights=False)
        Q = self.self_attn_norm(Q + self.dropout(self_attn_out))

        weights = torch.softmax(self.pool_weights(Q).squeeze(-1), dim=-1)
        pooled = torch.einsum('bk,bkd->bd', weights, Q)
        return pooled

print("AttentionPooling defined!")"""))

# ============================================================
# Cell 11: Edge Message Layer + BRep Encoder (from v4.9)
# ============================================================
cells.append(make_code_cell("""\
# Cell 11: Topology-Aware B-Rep Encoder (v4.9)

class EdgeMessageLayer(nn.Module):
    \"\"\"Message passing between faces and edges through topology.\"\"\"

    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.face_to_edge = nn.Sequential(
            nn.Linear(d * 3, d * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 2, d)
        )
        self.edge_to_face = nn.Sequential(
            nn.Linear(d * 2, d * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 2, d)
        )
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

        # Face -> Edge
        f1 = torch.gather(F_feat, 1, f1_idx.unsqueeze(-1).expand(-1, -1, d))
        f2 = torch.gather(F_feat, 1, f2_idx.unsqueeze(-1).expand(-1, -1, d))
        msg_e = self.face_to_edge(torch.cat([E, f1, f2], dim=-1))
        msg_e = msg_e * valid_edge.unsqueeze(-1).float()
        gate_e = self.gate_e(torch.cat([E, msg_e], dim=-1))
        E_new = self.norm_e(E + gate_e * msg_e)

        # Edge -> Face
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
    \"\"\"Topology-aware B-Rep encoder with message passing + transformer.\"\"\"

    def __init__(self, config):
        super().__init__()
        d = config.D_MODEL

        self.face_proj = nn.Sequential(
            nn.Linear(config.D_FACE, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(config.DROPOUT)
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(config.D_EDGE, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(config.DROPOUT)
        )
        self.face_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.edge_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.level_emb = nn.Embedding(config.MAX_BFS_LEVELS, d)

        self.msg_layers = nn.ModuleList([
            EdgeMessageLayer(d, config.DROPOUT) for _ in range(config.NUM_MSG_LAYERS)
        ])

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d, nhead=config.NUM_HEADS, dim_feedforward=d * 4,
                dropout=config.DROPOUT, activation='gelu', batch_first=True, norm_first=True
            ),
            num_layers=config.NUM_BREP_TF_LAYERS
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, face_feats, edge_feats, face_mask, edge_mask, edge_to_faces, bfs_level):
        F_feat = self.face_proj(face_feats.float()) + self.face_type
        F_feat = F_feat + self.level_emb(bfs_level.clamp(0, 31).long())
        E = self.edge_proj(edge_feats.float()) + self.edge_type

        for layer in self.msg_layers:
            F_feat, E = layer(F_feat, E, edge_to_faces, face_mask, edge_mask)

        X = torch.cat([F_feat, E], dim=1)
        mask = torch.cat([face_mask, edge_mask], dim=1).bool()
        X = self.transformer(X, src_key_padding_mask=~mask)
        X = self.norm(X)
        X = torch.nan_to_num(X, nan=0.0)
        return X, mask

print("TopologyBRepEncoder defined!")"""))

# ============================================================
# Cell 12: Text Encoder (from v4.9)
# ============================================================
cells.append(make_code_cell("""\
# Cell 12: Text Encoder (v4.9 - gradual dim reduction)

class TextEncoder(nn.Module):
    \"\"\"Enhanced text encoder: 3072 -> 768 -> d with transformer.\"\"\"

    def __init__(self, config):
        super().__init__()
        d = config.D_MODEL
        d_hidden = config.D_TEXT_HIDDEN

        self.proj_stage1 = nn.Sequential(
            nn.Linear(config.D_TEXT, d_hidden), nn.LayerNorm(d_hidden),
            nn.GELU(), nn.Dropout(config.DROPOUT),
        )
        self.proj_stage2 = nn.Sequential(
            nn.Linear(d_hidden, d), nn.LayerNorm(d),
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d, nhead=config.NUM_HEADS, dim_feedforward=d * 4,
                dropout=config.DROPOUT, activation='gelu', batch_first=True, norm_first=True
            ),
            num_layers=config.NUM_TEXT_TF_LAYERS
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, X, mask=None):
        X = self.proj_stage1(X.float())
        X = self.proj_stage2(X)
        if mask is not None:
            X = self.encoder(X, src_key_padding_mask=~mask.bool())
        else:
            X = self.encoder(X)
        X = self.norm(X)
        X = torch.nan_to_num(X, nan=0.0)
        return X, mask


class ProjectionHead(nn.Module):
    \"\"\"MLP projection head for contrastive learning.\"\"\"
    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_in), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_in, d_out)
        )
    def forward(self, x):
        return self.proj(x)

print("TextEncoder and ProjectionHead defined!")"""))

# ============================================================
# Cell 13: CLIP4CAD Model (main model)
# ============================================================
cells.append(make_code_cell("""\
# Cell 13: CLIP4CAD Model (v4.9 - 3-way contrastive, no codebook)

class CLIP4CADModel(nn.Module):
    \"\"\"
    3-way contrastive alignment: Text <-> BRep <-> PointCloud
    Architecture v4.9: AttentionPooling (no codebook), separate projection heads.
    \"\"\"

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.D_MODEL

        # Encoders
        self.text_encoder = TextEncoder(config)
        self.brep_encoder = TopologyBRepEncoder(config)
        self.pc_encoder = DGCNNEncoder(latent_size=config.D_PC, k=config.DGCNN_K)
        self.pc_proj_layer = nn.Sequential(
            nn.Linear(config.D_PC, d), nn.LayerNorm(d), nn.GELU(),
            nn.Dropout(config.DROPOUT), nn.Linear(d, d), nn.LayerNorm(d)
        )

        # Attention Pooling (replaces codebook)
        self.text_pool = AttentionPooling(d, config.NUM_POOL_QUERIES, num_heads=4, dropout=config.DROPOUT)
        self.brep_pool = AttentionPooling(d, config.NUM_POOL_QUERIES, num_heads=4, dropout=config.DROPOUT)

        # Projection Heads (separate per modality)
        self.text_proj = ProjectionHead(d, config.D_PROJ, config.DROPOUT)
        self.brep_proj = ProjectionHead(d, config.D_PROJ, config.DROPOUT)
        self.pc_proj = ProjectionHead(d, config.D_PROJ, config.DROPOUT)

        # Learnable temperature
        self.log_tau = nn.Parameter(torch.log(torch.tensor(config.TEMPERATURE)))

    @property
    def tau(self):
        return self.log_tau.exp().clamp(0.01, 1.0)

    def encode_pc(self, point_cloud):
        \"\"\"Encode point cloud with DGCNN.\"\"\"
        pc_feat = self.pc_encoder(point_cloud)  # (B, D_PC)
        pc_feat = self.pc_proj_layer(pc_feat)    # (B, d)
        z_pc = self.pc_proj(pc_feat)             # (B, D_PROJ)
        return z_pc

    def encode_brep(self, face_feats, edge_feats, face_mask, edge_mask, edge_to_faces, bfs_level):
        \"\"\"Encode B-Rep with topology transformer + attention pooling.\"\"\"
        X_brep, brep_mask = self.brep_encoder(face_feats, edge_feats, face_mask, edge_mask, edge_to_faces, bfs_level)
        z_brep_pooled = self.brep_pool(X_brep, brep_mask)
        z_brep = self.brep_proj(z_brep_pooled)
        return z_brep

    def encode_text(self, text_features, text_mask):
        \"\"\"Encode text with transformer + attention pooling.\"\"\"
        X_text, text_mask = self.text_encoder(text_features, text_mask)
        z_text_pooled = self.text_pool(X_text, text_mask)
        z_text = self.text_proj(z_text_pooled)
        return z_text

    def forward(self, batch, stage=1):
        device = next(self.parameters()).device

        # Encode PC (DGCNN on raw points)
        z_pc = self.encode_pc(batch['point_cloud'].to(device))

        # Encode BRep
        z_brep = self.encode_brep(
            batch['face_features'].to(device),
            batch['edge_features'].to(device),
            batch['face_mask'].to(device),
            batch['edge_mask'].to(device),
            batch['edge_to_faces'].to(device),
            batch['bfs_level'].to(device),
        )

        result = {'z_brep': z_brep, 'z_pc': z_pc, 'tau': self.tau}

        if stage >= 1:
            z_text = self.encode_text(
                batch['text_features'].to(device),
                batch['text_mask'].to(device),
            )
            result['z_text'] = z_text

        return result

print("CLIP4CADModel defined!")"""))

# ============================================================
# Cell 14: Loss Functions
# ============================================================
cells.append(make_code_cell("""\
# Cell 14: Loss Functions

# --- InfoNCE (same as baseline notebooks) ---
def info_nce_loss(features_a, features_b, temperature=0.07):
    features_a = F.normalize(features_a, dim=1)
    features_b = F.normalize(features_b, dim=1)
    labels = torch.arange(len(features_a), device=features_a.device)
    logits_a2b = torch.mm(features_a, features_b.t()) / temperature
    logits_b2a = torch.mm(features_b, features_a.t()) / temperature
    return (F.cross_entropy(logits_a2b, labels) + F.cross_entropy(logits_b2a, labels)) / 2.0

def uniformity_loss(z, t=2.0):
    \"\"\"Uniformity loss to prevent collapse.\"\"\"
    z = F.normalize(z, dim=-1)
    sq_pdist = torch.cdist(z, z, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

def variance_loss(z, min_var=0.5):
    \"\"\"Variance regularization - prevent embeddings from collapsing.\"\"\"
    var = z.var(dim=0).mean()
    return F.relu(min_var - var)

def clip4cad_loss(outputs, config, stage=1):
    \"\"\"
    Compute CLIP4CAD loss based on training stage.

    Stage 0: BRep <-> PC (anchoring)
    Stage 1: 3-way Text <-> BRep <-> PC
    \"\"\"
    losses = {}
    tau = outputs['tau']

    z_brep = F.normalize(outputs['z_brep'], dim=-1)
    z_pc = F.normalize(outputs['z_pc'], dim=-1)
    B = z_brep.shape[0]
    labels = torch.arange(B, device=z_brep.device)
    neg_mask = ~torch.eye(B, dtype=torch.bool, device=z_brep.device)

    # BRep <-> PC
    logits_b2p = (z_brep.float() @ z_pc.float().T) / tau
    loss_b2p = (F.cross_entropy(logits_b2p, labels, label_smoothing=config.LABEL_SMOOTHING)
              + F.cross_entropy(logits_b2p.T, labels, label_smoothing=config.LABEL_SMOOTHING)) / 2
    losses['infonce_b2p'] = loss_b2p

    if stage >= 1 and 'z_text' in outputs:
        z_text = F.normalize(outputs['z_text'], dim=-1)

        # Text <-> BRep
        logits_t2b = (z_text.float() @ z_brep.float().T) / tau
        loss_t2b = (F.cross_entropy(logits_t2b, labels, label_smoothing=config.LABEL_SMOOTHING)
                  + F.cross_entropy(logits_t2b.T, labels, label_smoothing=config.LABEL_SMOOTHING)) / 2

        # Text <-> PC
        logits_t2p = (z_text.float() @ z_pc.float().T) / tau
        loss_t2p = (F.cross_entropy(logits_t2p, labels, label_smoothing=config.LABEL_SMOOTHING)
                  + F.cross_entropy(logits_t2p.T, labels, label_smoothing=config.LABEL_SMOOTHING)) / 2

        losses['infonce_t2b'] = loss_t2b
        losses['infonce_t2p'] = loss_t2p

        # Weighted average
        tw = config.TEXT_LOSS_WEIGHT
        losses['infonce'] = (tw * loss_t2b + tw * loss_t2p + loss_b2p) / (2 * tw + 1)

        # Uniformity (all 3)
        losses['uniformity'] = (uniformity_loss(z_text) + uniformity_loss(z_brep) + uniformity_loss(z_pc)) / 3
        losses['variance'] = (variance_loss(z_text) + variance_loss(z_brep)) / 2

        # Margins (for monitoring)
        with torch.no_grad():
            sim_tb = z_text @ z_brep.T
            losses['margin_tb'] = sim_tb.diag().mean() - sim_tb[neg_mask].mean()
            sim_tp = z_text @ z_pc.T
            losses['margin_tp'] = sim_tp.diag().mean() - sim_tp[neg_mask].mean()
            sim_bp = z_brep @ z_pc.T
            losses['margin_bp'] = sim_bp.diag().mean() - sim_bp[neg_mask].mean()
            losses['margin'] = (losses['margin_tb'] + losses['margin_tp'] + losses['margin_bp']) / 3
    else:
        losses['infonce'] = loss_b2p
        losses['uniformity'] = uniformity_loss(z_brep)
        losses['variance'] = variance_loss(z_brep)

        with torch.no_grad():
            sim_bp = z_brep @ z_pc.T
            losses['margin'] = sim_bp.diag().mean() - sim_bp[neg_mask].mean()

    # Total
    losses['total'] = (
        losses['infonce']
        + config.UNIFORMITY_WEIGHT * losses['uniformity']
        + config.VARIANCE_WEIGHT * losses['variance']
    )

    return losses['total'], losses

print("Loss functions defined!")"""))

# ============================================================
# Cell 15: Metrics (same calc_metrics + CLIP4CAD evaluate)
# ============================================================
cells.append(make_code_cell("""\
# Cell 15: Metrics Calculation (same calc_metrics as baselines)

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
    \"\"\"Evaluate CLIP4CAD on all 3 retrieval directions.\"\"\"
    model.eval()
    all_z_text, all_z_brep, all_z_pc = [], [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        outputs = model(batch, stage=1)
        all_z_text.append(outputs['z_text'].cpu())
        all_z_brep.append(outputs['z_brep'].cpu())
        all_z_pc.append(outputs['z_pc'].cpu())

    z_text = F.normalize(torch.cat(all_z_text), dim=1)
    z_brep = F.normalize(torch.cat(all_z_brep), dim=1)
    z_pc = F.normalize(torch.cat(all_z_pc), dim=1)

    # 3 retrieval directions
    metrics = {}

    t2b = calc_metrics(z_text @ z_brep.T, k_values)
    for k, v in t2b.items(): metrics[f'text2brep_{k}'] = v

    t2p = calc_metrics(z_text @ z_pc.T, k_values)
    for k, v in t2p.items(): metrics[f'text2pc_{k}'] = v

    b2p = calc_metrics(z_brep @ z_pc.T, k_values)
    for k, v in b2p.items(): metrics[f'brep2pc_{k}'] = v

    return metrics

print("Metrics functions defined!")"""))

# ============================================================
# Cell 16: Training Utilities
# ============================================================
cells.append(make_code_cell("""\
# Cell 16: Training & Checkpoint Utilities

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
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        margin = loss_dict.get('margin', torch.tensor(0.0))
        total_margin += margin.item() if isinstance(margin, torch.Tensor) else margin

        if i % 20 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'margin': f'{margin.item() if isinstance(margin, torch.Tensor) else margin:.4f}'
            })

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
        'metrics': metrics
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state, os.path.join(save_dir, filename))


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return 0
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint.get('model_state_dict', checkpoint).items()}
    model.load_state_dict(state_dict, strict=False)
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0) + 1


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def plot_training_curves(metrics_df, save_dir):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train')
    axes[0].plot(metrics_df['epoch'], metrics_df['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss'); axes[0].legend()

    for k in [1, 5, 10]:
        col = f'val_text2pc_recall@{k}'
        if col in metrics_df.columns:
            axes[1].plot(metrics_df['epoch'], metrics_df[col], label=f'R@{k}')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Recall'); axes[1].set_title('Text->PC Recall'); axes[1].legend()

    for k in [1, 5, 10]:
        col = f'val_text2brep_recall@{k}'
        if col in metrics_df.columns:
            axes[2].plot(metrics_df['epoch'], metrics_df[col], label=f'R@{k}')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Recall'); axes[2].set_title('Text->BRep Recall'); axes[2].legend()

    if 'val_margin' in metrics_df.columns:
        axes[3].plot(metrics_df['epoch'], metrics_df['val_margin'], label='Margin', color='green')
        axes[3].set_xlabel('Epoch'); axes[3].set_ylabel('Margin'); axes[3].set_title('Contrastive Margin'); axes[3].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"Memory cleaned. GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("Training utilities defined!")"""))

# ============================================================
# Cell 17: Dataset
# ============================================================
cells.append(make_code_cell("""\
# Cell 17: CLIP4CAD Dataset (PLY from disk/cache + HDF5 for BRep & Text)

class CLIP4CADDataset(Dataset):
    \"\"\"
    Trimodal dataset: Point Cloud (raw PLY) + B-Rep (HDF5) + Text (HDF5).
    Follows the same caching pattern as OptimizedTextDataset.
    \"\"\"

    def __init__(self, df, config, brep_uid_to_idx, text_uid_to_idx,
                 text_feat_key, text_mask_key, cache_ply=True):
        self.config = config
        self.data_root = config.DATA_ROOT
        self.samples = df.reset_index(drop=True)
        self.brep_uid_to_idx = brep_uid_to_idx
        self.text_uid_to_idx = text_uid_to_idx
        self.text_feat_key = text_feat_key
        self.text_mask_key = text_mask_key

        # Open HDF5 files (keep handles open for speed)
        self.brep_h5 = h5py.File(config.BREP_H5_PATH, 'r')
        self.text_h5 = h5py.File(config.TEXT_H5_PATH, 'r')

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
                    if norm > 0:
                        xyz = xyz / norm
                    self.ply_cache[idx] = xyz
                except:
                    self.ply_cache[idx] = np.zeros((10000, 3), dtype=np.float32)

            mem_gb = sum(p.nbytes for p in self.ply_cache.values()) / 1e9
            print(f"Cached {len(self.ply_cache)} PLY files ({mem_gb:.1f} GB)")

        print(f"CLIP4CADDataset ready: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        row = self.samples.iloc[i]
        uid_str = str(row['uid'])

        # --- Point Cloud (from cache or disk) ---
        if self.cache_ply:
            pc = self.ply_cache[i]
        else:
            ply_path = os.path.join(self.data_root, row['ply_path'])
            pc = load_ply_fast(ply_path)
            pc = pc - pc.mean(0)
            norm = np.max(np.linalg.norm(pc, axis=1))
            if norm > 0:
                pc = pc / norm

        idx_pts = np.random.choice(len(pc), self.config.NUM_POINTS,
                                   replace=len(pc) < self.config.NUM_POINTS)
        pc = torch.from_numpy(pc[idx_pts]).T  # (3, N)

        # --- B-Rep (from HDF5) ---
        brep_idx = self.brep_uid_to_idx[uid_str]
        face_features = torch.from_numpy(self.brep_h5['face_features'][brep_idx].astype(np.float32))
        edge_features = torch.from_numpy(self.brep_h5['edge_features'][brep_idx].astype(np.float32))
        face_mask = torch.from_numpy(self.brep_h5['face_masks'][brep_idx].astype(np.float32))
        edge_mask = torch.from_numpy(self.brep_h5['edge_masks'][brep_idx].astype(np.float32))
        edge_to_faces = torch.from_numpy(self.brep_h5['edge_to_faces'][brep_idx].astype(np.int64))
        bfs_level = torch.from_numpy(self.brep_h5['bfs_level'][brep_idx].astype(np.int64))

        # --- Text (from HDF5) ---
        text_idx = self.text_uid_to_idx[uid_str]
        text_features = torch.from_numpy(self.text_h5[self.text_feat_key][text_idx].astype(np.float32))
        text_mask = torch.from_numpy(self.text_h5[self.text_mask_key][text_idx].astype(np.float32))

        return {
            'point_cloud': pc,
            'face_features': face_features,
            'edge_features': edge_features,
            'face_mask': face_mask,
            'edge_mask': edge_mask,
            'edge_to_faces': edge_to_faces,
            'bfs_level': bfs_level,
            'text_features': text_features,
            'text_mask': text_mask,
            'idx': i,
        }

    def __del__(self):
        if hasattr(self, 'brep_h5') and self.brep_h5:
            self.brep_h5.close()
        if hasattr(self, 'text_h5') and self.text_h5:
            self.text_h5.close()

print("CLIP4CADDataset defined!")"""))

# ============================================================
# Cell 18: Create Datasets & DataLoaders
# ============================================================
cells.append(make_code_cell("""\
# Cell 18: Create Datasets & DataLoaders

print("Creating datasets...")
train_dataset = CLIP4CADDataset(
    train_df_tri, config, brep_uid_to_idx, text_uid_to_idx,
    text_feat_key, text_mask_key, cache_ply=True
)
val_dataset = CLIP4CADDataset(
    val_df_tri, config, brep_uid_to_idx, text_uid_to_idx,
    text_feat_key, text_mask_key, cache_ply=True
)

train_loader = DataLoader(
    train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
    num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)

print(f"\\nDataLoaders created:")
print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
print(f"  Batch size: {config.BATCH_SIZE}")"""))

# ============================================================
# Cell 19: Device Setup + Model Creation
# ============================================================
cells.append(make_code_cell("""\
# Cell 19: Device Setup + Model Creation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Create model
model = CLIP4CADModel(config).to(device)

# Use DataParallel if multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

params = count_parameters(model)
print(f"\\nModel parameters:")
print(f"  Total: {params['total']:,}")
print(f"  Trainable: {params['trainable']:,}")

# Optimizer with different LR for text
core_model = model.module if hasattr(model, 'module') else model
text_params = list(core_model.text_encoder.parameters()) + \\
              list(core_model.text_pool.parameters()) + \\
              list(core_model.text_proj.parameters())
text_param_ids = set(id(p) for p in text_params)
other_params = [p for p in core_model.parameters() if id(p) not in text_param_ids and p.requires_grad]

optimizer = optim.AdamW([
    {'params': other_params, 'lr': config.LEARNING_RATE, 'weight_decay': config.WEIGHT_DECAY},
    {'params': text_params, 'lr': config.LEARNING_RATE * config.TEXT_LR_MULT, 'weight_decay': config.WEIGHT_DECAY},
])

print(f"\\nOptimizer: AdamW")
print(f"  Base LR: {config.LEARNING_RATE}")
print(f"  Text LR: {config.LEARNING_RATE * config.TEXT_LR_MULT}")"""))

# ============================================================
# Cell 20: Training header (markdown)
# ============================================================
cells.append(make_md_cell("""\
---
# Training

**Stage 0** (epochs 1-8): BRep <-> PC anchoring only
- Goal: margin > 0.3, BRep-PC cosine > 0.7

**Stage 1** (epochs 9-30): Full 3-way contrastive (Text <-> BRep <-> PC)
- Goal: margin > 0.4, text2pc R@1 > 20% by epoch 15, > 50% by epoch 30"""))

# ============================================================
# Cell 21: Training Loop
# ============================================================
cells.append(make_code_cell("""\
# Cell 21: Training Loop

model_name = 'CLIP4CAD-v4.9'
save_dir = os.path.join(config.OUTPUT_DIR, model_name.replace(' ', '_'))
os.makedirs(save_dir, exist_ok=True)

writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs')) if config.ENABLE_TENSORBOARD else None

# LR Scheduler (cosine with warmup)
total_steps = config.NUM_EPOCHS * len(train_loader)
warmup_steps = config.WARMUP_EPOCHS * len(train_loader)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.01 + 0.5 * 0.99 * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

metrics_history = []
best_recall = 0.0
all_training_results = []

print(f"\\n{'='*80}")
print(f"Training: {model_name}")
print(f"{'='*80}")

for epoch in range(config.NUM_EPOCHS):
    stage = 0 if epoch < config.STAGE0_EPOCHS else 1
    print(f"\\nEpoch {epoch+1}/{config.NUM_EPOCHS} [Stage {stage}]")

    # Train
    train_loss, train_margin = train_epoch(model, train_loader, optimizer, scheduler, device, config, stage=stage)

    # Validate
    val_loss, val_margin = validate_epoch(model, val_loader, device, config, stage=stage)

    # Evaluate retrieval (only in Stage 1)
    retrieval_metrics = {}
    if stage >= 1:
        retrieval_metrics = evaluate_clip4cad(model, val_loader, device, config.K_VALUES)

    # Print
    print(f"  Train Loss: {train_loss:.4f}, Margin: {train_margin:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}, Margin: {val_margin:.4f}")
    if retrieval_metrics:
        print(f"  Text->PC  R@1: {retrieval_metrics.get('text2pc_recall@1', 0):.2f}%")
        print(f"  Text->BRep R@1: {retrieval_metrics.get('text2brep_recall@1', 0):.2f}%")
        print(f"  BRep->PC  R@1: {retrieval_metrics.get('brep2pc_recall@1', 0):.2f}%")

    # Log
    epoch_metrics = {
        'epoch': epoch + 1, 'stage': stage,
        'train_loss': train_loss, 'val_loss': val_loss,
        'train_margin': train_margin, 'val_margin': val_margin,
        **{f'val_{k}': v for k, v in retrieval_metrics.items()}
    }
    metrics_history.append(epoch_metrics)

    if writer:
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Margin/train', train_margin, epoch)
        writer.add_scalar('Margin/val', val_margin, epoch)
        for k, v in retrieval_metrics.items():
            writer.add_scalar(f'Val/{k}', v, epoch)

    # Checkpoint
    r1 = retrieval_metrics.get('text2pc_recall@1', 0)
    if r1 > best_recall:
        best_recall = r1
        save_checkpoint(model, optimizer, scheduler, epoch, retrieval_metrics, save_dir, 'best.pth')
        print(f"  ** New best Text->PC R@1: {best_recall:.2f}% **")

    save_checkpoint(model, optimizer, scheduler, epoch, retrieval_metrics, save_dir, 'latest.pth')
    if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
        save_checkpoint(model, optimizer, scheduler, epoch, retrieval_metrics, save_dir, f'epoch_{epoch+1}.pth')

    # Collapse detection
    if stage >= 1 and epoch > config.STAGE0_EPOCHS + 3 and val_margin < 0.01:
        print("  !! WARNING: Margin near 0 - possible collapse!")

# Save metrics
metrics_df = pd.DataFrame(metrics_history)
metrics_df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
plot_training_curves(metrics_df, save_dir)

if writer: writer.close()

print(f"\\n{'='*80}")
print(f"{model_name} training complete!")
print(f"Best Text->PC R@1: {best_recall:.2f}%")
print(f"{'='*80}")"""))

# ============================================================
# Cell 22: Final Evaluation
# ============================================================
cells.append(make_code_cell("""\
# Cell 22: Final Evaluation on Validation Set

print(f"\\nFinal validation for: CLIP4CAD-v4.9")
print(f"{'='*80}")

# Load best checkpoint
best_path = os.path.join(save_dir, 'best.pth')
if os.path.exists(best_path):
    load_checkpoint(model, None, None, best_path)
    print(f"Loaded best checkpoint from {best_path}")

# Full evaluation
final_metrics = evaluate_clip4cad(model, val_loader, device, config.K_VALUES)

print(f"\\nFinal Retrieval Results:")
print(f"{'='*50}")
for direction in ['text2pc', 'text2brep', 'brep2pc']:
    print(f"\\n  {direction}:")
    for k in config.K_VALUES:
        r = final_metrics.get(f'{direction}_recall@{k}', 0)
        m = final_metrics.get(f'{direction}_mAP@{k}', 0)
        print(f"    R@{k}: {r:.2f}%  mAP@{k}: {m:.2f}%")

# Store for comparison
all_training_results.append({
    'model': 'CLIP4CAD-v4.9',
    'text2pc_recall@1': final_metrics.get('text2pc_recall@1', 0),
    'text2brep_recall@1': final_metrics.get('text2brep_recall@1', 0),
    'brep2pc_recall@1': final_metrics.get('brep2pc_recall@1', 0),
    'text2pc_recall@5': final_metrics.get('text2pc_recall@5', 0),
    'text2pc_recall@10': final_metrics.get('text2pc_recall@10', 0),
})"""))

# ============================================================
# Cell 23: Copy to Drive
# ============================================================
cells.append(make_code_cell("""\
# Cell 23: Copy Checkpoints to Google Drive
import shutil

DRIVE_SAVE_DIR = "/content/drive/MyDrive/MMCAD/clip4cad_checkpoints"
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

print("Copying checkpoints to Drive...")
for fname in ['best.pth', 'latest.pth', 'metrics.csv', 'training_curves.png']:
    src = os.path.join(save_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(DRIVE_SAVE_DIR, fname))
        print(f"  Copied {fname}")

print(f"\\nCheckpoints saved to: {DRIVE_SAVE_DIR}")"""))

# ============================================================
# Cell 24: Cleanup
# ============================================================
cells.append(make_code_cell("""\
# Cell 24: Cleanup
cleanup_memory()"""))

# ============================================================
# Cell 25: Comparison Table
# ============================================================
cells.append(make_code_cell("""\
# Cell 25: Comparison Table
print("\\n" + "="*80)
print("COMPARISON: CLIP4CAD vs Baselines")
print("="*80)

# Add baseline results (from existing notebook runs)
baselines = [
    {'model': 'BERT-DGCNN', 'text2pc_recall@1': 0, 'notes': 'Fill from mmcad_training_colab_text.ipynb'},
    {'model': 'CLIP-DGCNN', 'text2pc_recall@1': 0, 'notes': 'Fill from mmcad_training_colab_text.ipynb'},
]

print(f"\\n{'Model':<25} {'Text->PC R@1':>12} {'Text->BRep R@1':>14} {'BRep->PC R@1':>12}")
print("-" * 70)

for result in all_training_results:
    print(f"{result['model']:<25} {result.get('text2pc_recall@1', 'N/A'):>12.2f}% {result.get('text2brep_recall@1', 'N/A'):>13.2f}% {result.get('brep2pc_recall@1', 'N/A'):>11.2f}%")

print("\\nNote: Fill in baseline results from your existing notebook runs for comparison.")"""))

# ============================================================
# Cell 26: Save Summary
# ============================================================
cells.append(make_code_cell("""\
# Cell 26: Save Summary CSV
if all_training_results:
    summary_df = pd.DataFrame(all_training_results)
    summary_path = os.path.join(save_dir, 'comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    display(summary_df)"""))

# ============================================================
# Assemble notebook
# ============================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "A100"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

# Fix: remove trailing newline from last line of each cell
for cell in notebook['cells']:
    if cell['source'] and cell['source'][-1].endswith('\n'):
        cell['source'][-1] = cell['source'][-1].rstrip('\n')

output_path = "mmcad_training_clip4cad.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Generated {output_path} with {len(cells)} cells")
