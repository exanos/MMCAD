#!/usr/bin/env python3
"""
MM-CAD cross-modal retrieval inference.

Loads the released MM-CAD checkpoints and retrieves CAD models from a text,
sketch, and/or photorealistic-image query against a B-Rep or point-cloud gallery.

    Query modalities   text (EmbeddingGemma-300M)
                       sketch (ViT-Base, B-Rep-anchored)
                       photoreal image (SigLIP-Base, B-Rep-anchored)
    Galleries          B-Rep (BRepFormer)  ·  point cloud (DGCNN)
    Shared space       d=768, Matryoshka-nested over {128, 256, 512, 768}

Multi-modal queries are fused by summing the per-modality unit vectors and
renormalizing -- there is no learned fusion head. Trimodal text+sketch+image
reaches 45.91 R@1 against the B-Rep gallery on the MM-CAD:B validation split.

Checkpoints and precomputed gallery embeddings:
    https://huggingface.co/datasets/exanos/MMCAD  (checkpoints/)

Usage
-----
    python inference.py --query "herringbone gear with ten lightening holes"
    python inference.py --query "servo mount" --sketch my_sketch.png --dim 128
    python inference.py --evaluate --split val

Paper: https://doi.org/10.1111/cgf.70523
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

class Config:
    D_SHARED = 768
    MATRYOSHKA_DIMS = [128, 256, 512, 768]

    TEXT_MODEL = "google/embeddinggemma-300m"
    TEXT_MAX_LENGTH = 512
    VIT_MODEL = "google/vit-base-patch16-224"
    SIGLIP_MODEL = "google/siglip-base-patch16-224"

    DGCNN_K = 20
    DGCNN_LATENT = 1024
    NUM_POINTS = 2048

    # BRepFormer tower (needed so the joint checkpoint loads strictly)
    FACE_DIM = 16
    D_BREP_MODEL = 512
    N_BREP_LAYERS = 6
    N_HEADS = 8
    MAX_FACES = 192

    K_VALUES = [1, 5, 10, 25]
    SEED = 42


CONFIG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Text encoder
# --------------------------------------------------------------------------

class EmbeddingGemmaEncoder(nn.Module):
    """EmbeddingGemma-300M wrapped so its SentenceTransformer pipeline is a
    plain nn.ModuleList, which keeps checkpoint keys stable."""

    def __init__(self, model_id: str = Config.TEXT_MODEL):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(model_id, trust_remote_code=True)
        self._pipeline = nn.ModuleList(list(st))
        self.tokenizer = st.tokenizer

    def forward(self, input_ids, attention_mask):
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
        for module in self._pipeline:
            features = module(features)
        return features["sentence_embedding"]


# --------------------------------------------------------------------------
# Sketch encoder (Stage 2, anchored on frozen B-Rep embeddings)
# --------------------------------------------------------------------------

class ViTSketchEncoder(nn.Module):
    def __init__(self, model_name: str = Config.VIT_MODEL, d_out: int = Config.D_SHARED):
        super().__init__()
        from transformers import ViTModel

        self.vit = ViTModel.from_pretrained(model_name)
        hidden = self.vit.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_out, d_out),
        )

    def forward(self, pixel_values):
        cls = self.vit(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.projection(cls)


# --------------------------------------------------------------------------
# Photorealistic-image encoder (Stage 2, anchored on frozen B-Rep embeddings)
# --------------------------------------------------------------------------

class SigLIPRenderEncoder(nn.Module):
    def __init__(self, model_name: str = Config.SIGLIP_MODEL, d_out: int = Config.D_SHARED):
        super().__init__()
        from transformers import SiglipVisionModel

        self.vision = SiglipVisionModel.from_pretrained(model_name)
        hidden = self.vision.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_out, d_out),
        )

    def forward(self, pixel_values):
        pooled = self.vision(pixel_values=pixel_values).pooler_output
        return self.projection(pooled)


# --------------------------------------------------------------------------
# Point-cloud encoder (DGCNN)
# --------------------------------------------------------------------------

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
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
    def __init__(self, latent_size=Config.DGCNN_LATENT, k=Config.DGCNN_K):
        super().__init__()
        self.k = k
        self.bn1, self.bn2 = nn.BatchNorm2d(64), nn.BatchNorm2d(64)
        self.bn3, self.bn4 = nn.BatchNorm2d(128), nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(latent_size)
        self.bn6, self.bn7 = nn.BatchNorm1d(512), nn.BatchNorm1d(latent_size)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), self.bn1, nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), self.bn2, nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), self.bn3, nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), self.bn4, nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, latent_size, 1, bias=False), self.bn5, nn.LeakyReLU(0.2))
        self.linear1 = nn.Linear(latent_size * 2, 512, bias=False)
        self.linear2 = nn.Linear(512, latent_size)
        self.dp = nn.Dropout(0.3)

    def forward(self, x):
        x1 = self.conv1(get_graph_feature(x, self.k)).max(dim=-1)[0]
        x2 = self.conv2(get_graph_feature(x1, self.k)).max(dim=-1)[0]
        x3 = self.conv3(get_graph_feature(x2, self.k)).max(dim=-1)[0]
        x4 = self.conv4(get_graph_feature(x3, self.k)).max(dim=-1)[0]
        x = self.conv5(torch.cat((x1, x2, x3, x4), dim=1))
        x = torch.cat((x.max(2)[0], x.mean(2)), 1)
        x = self.dp(F.leaky_relu(self.bn6(self.linear1(x)), 0.2))
        return self.dp(self.bn7(self.linear2(x)))


class DGCNNWithProjection(nn.Module):
    def __init__(self, d_out=Config.D_SHARED, k=Config.DGCNN_K, latent_size=Config.DGCNN_LATENT):
        super().__init__()
        self.dgcnn = DGCNNEncoder(latent_size=latent_size, k=k)
        self.projection = nn.Sequential(
            nn.Linear(latent_size, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.projection(self.dgcnn(x))


# --------------------------------------------------------------------------
# B-Rep encoder (BRepFormer)
# --------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.scale * x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=True)
        self.w2 = nn.Linear(d_hidden, d_in, bias=True)
        self.w3 = nn.Linear(d_in, d_hidden, bias=True)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BRepFormerLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = SwiGLUFFN(d_model, d_model * ffn_mult)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None, mask=None):
        B, N, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).reshape(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(h).reshape(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_bias is not None:
            attn = attn + attn_bias
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.out_proj(out)
        return x + self.ffn(self.norm2(x))


class BRepFormerEncoder(nn.Module):
    def __init__(self, d_face_in=16, d_out=768, d_model=512, n_heads=8, n_layers=6, max_faces=192):
        super().__init__()
        self.d_model, self.max_faces = d_model, max_faces
        self.face_proj = nn.Linear(d_face_in, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.topo_bias_proj = nn.Sequential(
            nn.Linear(3, d_model), RMSNorm(d_model), nn.ReLU(), nn.Linear(d_model, n_heads)
        )
        self.layers = nn.ModuleList([BRepFormerLayer(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_out), nn.GELU(), nn.Linear(d_out, d_out)
        )

    def forward(self, face_features, face_centroids, edge_to_faces, face_mask, face_normals):
        B, N, _ = face_features.shape
        h = self.face_proj(face_features)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        mask = torch.cat([torch.ones(B, 1, device=face_mask.device), face_mask], dim=1)
        bias = self.topo_bias_proj(face_centroids)
        bias = F.pad(bias, (0, 0, 1, 0))
        attn_bias = (bias.unsqueeze(2) + bias.unsqueeze(1)).permute(0, 3, 1, 2)
        for layer in self.layers:
            h = layer(h, attn_bias=attn_bias, mask=mask)
        return self.output_proj(self.final_norm(h)[:, 0])


class Baseline(nn.Module):
    """The Stage-1 joint model: text + B-Rep + point cloud trained together."""

    def __init__(self, text_encoder, config=CONFIG):
        super().__init__()
        self.text_encoder = text_encoder
        self.brep_encoder = BRepFormerEncoder(
            d_face_in=config.FACE_DIM,
            d_out=config.D_SHARED,
            d_model=config.D_BREP_MODEL,
            n_heads=config.N_HEADS,
            n_layers=config.N_BREP_LAYERS,
            max_faces=config.MAX_FACES,
        )
        self.pc_encoder = DGCNNWithProjection(
            d_out=config.D_SHARED, k=config.DGCNN_K, latent_size=config.DGCNN_LATENT
        )
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))
        self.matryoshka_dims = config.MATRYOSHKA_DIMS


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _state_dict(ckpt_path: str | Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt
    return ckpt, {}


def load_joint_model(ckpt_path: str | Path, device=DEVICE):
    """Load the Stage-1 tri-modal model (text + B-Rep + point cloud)."""
    model = Baseline(EmbeddingGemmaEncoder(), CONFIG)
    sd, meta = _state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[joint] {len(missing)} missing keys (first: {missing[:2]})", file=sys.stderr)
    if unexpected:
        print(f"[joint] {len(unexpected)} unexpected keys (first: {unexpected[:2]})", file=sys.stderr)
    model.eval().to(device)
    if "epoch" in meta:
        print(f"[joint] loaded epoch {meta['epoch']}")
    return model


def load_sketch_encoder(ckpt_path: str | Path, device=DEVICE):
    model = ViTSketchEncoder()
    sd, meta = _state_dict(ckpt_path)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    if "epoch" in meta:
        print(f"[sketch] loaded epoch {meta['epoch']}")
    return model


def load_render_encoder(ckpt_path: str | Path, device=DEVICE):
    model = SigLIPRenderEncoder()
    sd, meta = _state_dict(ckpt_path)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    if "epoch" in meta:
        print(f"[render] loaded epoch {meta['epoch']}")
    return model


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

def truncate(z: torch.Tensor, dim: int) -> torch.Tensor:
    """Matryoshka truncation followed by renormalization."""
    return F.normalize(z[..., :dim], dim=-1)


@torch.no_grad()
def embed_text(model, query: str, dim: int = 768, device=DEVICE) -> torch.Tensor:
    prompted = f"title: none | text: {query}"
    tok = model.text_encoder.tokenizer(
        [prompted],
        padding=True,
        truncation=True,
        max_length=CONFIG.TEXT_MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    z = model.text_encoder(tok["input_ids"], tok["attention_mask"])
    return truncate(z, dim)


@torch.no_grad()
def embed_image(encoder, image_path: str | Path, processor_name: str,
                dim: int = 768, device=DEVICE) -> torch.Tensor:
    from PIL import Image
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(processor_name)
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
    return truncate(encoder(pixel_values), dim)


@torch.no_grad()
def embed_point_cloud(model, points: np.ndarray, dim: int = 768, device=DEVICE) -> torch.Tensor:
    """points: (N, 3) or (N, 6) with normals; sampled to CONFIG.NUM_POINTS."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] != CONFIG.NUM_POINTS:
        idx = np.random.choice(pts.shape[0], CONFIG.NUM_POINTS,
                               replace=pts.shape[0] < CONFIG.NUM_POINTS)
        pts = pts[idx]
    pts = pts[:, :3]
    pts = pts - pts.mean(0)
    pts = pts / (np.linalg.norm(pts, axis=1).max() + 1e-9)
    x = torch.from_numpy(pts).T.unsqueeze(0).to(device)
    return truncate(model.pc_encoder(x), dim)


def fuse(*embeddings: torch.Tensor) -> torch.Tensor:
    """Mean fusion: sum the unit vectors, renormalize. No learned head --
    replacing this with an MLP did not improve recall."""
    valid = [e for e in embeddings if e is not None]
    if not valid:
        raise ValueError("fuse() needs at least one embedding")
    return F.normalize(torch.stack(valid, dim=0).sum(dim=0), dim=-1)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def load_gallery(path: str | Path):
    """Load precomputed gallery embeddings.

    Expects an .npz with 'embeddings' (N, 768) float32 and 'uids' (N,).
    Released alongside the checkpoints so you do not have to re-encode 192K
    B-Reps yourself.
    """
    data = np.load(path, allow_pickle=True)
    embs = torch.from_numpy(data["embeddings"].astype(np.float32))
    uids = [str(u) for u in data["uids"]]
    return embs, uids


@torch.no_grad()
def retrieve(query_vec: torch.Tensor, gallery: torch.Tensor, uids: list[str],
             top_k: int = 10, dim: int = 768, chunk: int = 8192):
    """Cosine retrieval. For interactive use over the full 192K gallery, build
    a FAISS index at d=128 instead -- accuracy loss is negligible (trimodal
    R@1 45.91 -> 45.50)."""
    q = truncate(query_vec.float().cpu(), dim)
    scores = []
    for start in range(0, gallery.shape[0], chunk):
        g = truncate(gallery[start:start + chunk].float(), dim)
        scores.append(q @ g.T)
    scores = torch.cat(scores, dim=-1).squeeze(0)
    top = torch.topk(scores, min(top_k, scores.numel()))
    return [(uids[i], float(s)) for s, i in zip(top.values, top.indices)]


def recall_at_k(query_embs: torch.Tensor, gallery_embs: torch.Tensor,
                k_values=CONFIG.K_VALUES, dim: int = 768, chunk: int = 2048):
    """Diagonal-truth recall: query i matches gallery i."""
    q = truncate(query_embs.float(), dim)
    g = truncate(gallery_embs.float(), dim)
    n = q.shape[0]
    hits = {k: 0 for k in k_values}
    max_k = max(k_values)
    for start in range(0, n, chunk):
        block = q[start:start + chunk]
        scores = block @ g.T
        top = torch.topk(scores, max_k, dim=-1).indices
        truth = torch.arange(start, min(start + chunk, n)).unsqueeze(1)
        for k in k_values:
            hits[k] += (top[:, :k] == truth).any(dim=1).sum().item()
    return {f"R@{k}": 100.0 * hits[k] / n for k in k_values}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MM-CAD cross-modal retrieval")
    ap.add_argument("--ckpt-dir", default="checkpoints",
                    help="directory holding the released checkpoints")
    ap.add_argument("--joint", default=None, help="joint tri-modal checkpoint (.pth)")
    ap.add_argument("--sketch-ckpt", default=None, help="sketch encoder checkpoint (.pth)")
    ap.add_argument("--render-ckpt", default=None, help="render encoder checkpoint (.pth)")
    ap.add_argument("--gallery", default=None, help="precomputed gallery embeddings (.npz)")
    ap.add_argument("--query", default=None, help="text query")
    ap.add_argument("--sketch", default=None, help="path to a sketch image")
    ap.add_argument("--image", default=None, help="path to a photorealistic image")
    ap.add_argument("--dim", type=int, default=768, choices=CONFIG.MATRYOSHKA_DIMS)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="emit results as JSON")
    args = ap.parse_args()

    if not (args.query or args.sketch or args.image):
        ap.error("give at least one of --query / --sketch / --image")

    ckpt_dir = Path(args.ckpt_dir)
    joint_path = args.joint or ckpt_dir / "baseline_trimodal_v4.pth"
    gallery_path = args.gallery or ckpt_dir / "gallery_brep.npz"

    model = load_joint_model(joint_path)

    parts = []
    if args.query:
        parts.append(embed_text(model, args.query, args.dim))
    if args.sketch:
        sk_path = args.sketch_ckpt or ckpt_dir / "sketch_encoder_v1.pth"
        parts.append(embed_image(load_sketch_encoder(sk_path), args.sketch,
                                 CONFIG.VIT_MODEL, args.dim))
    if args.image:
        rd_path = args.render_ckpt or ckpt_dir / "render_encoder_v1.pth"
        parts.append(embed_image(load_render_encoder(rd_path), args.image,
                                 CONFIG.SIGLIP_MODEL, args.dim))

    z = fuse(*parts)

    gallery, uids = load_gallery(gallery_path)
    results = retrieve(z, gallery, uids, top_k=args.top_k, dim=args.dim)

    if args.json:
        print(json.dumps([{"uid": u, "score": s} for u, s in results], indent=2))
    else:
        modalities = "+".join(m for m, on in
                              [("text", args.query), ("sketch", args.sketch), ("image", args.image)] if on)
        print(f"\nquery [{modalities}] @ d={args.dim}, top {len(results)} of {len(uids)}")
        for rank, (uid, score) in enumerate(results, 1):
            print(f"  {rank:>3}. uid {uid:<12} {score:.4f}")


if __name__ == "__main__":
    main()
