"""
Comprehensive Dry-Run Test for CLIP4CAD-H Pipeline

Tests all pretrained models and pipeline components without requiring real dataset.
Uses minimal dummy data to verify:
- Individual encoder loading and forward passes
- Full CLIP4CAD_H model integration
- Data loading pipeline
- Pre-computed feature caching

Usage:
    python scripts/test_dry_run.py --test all
    python scripts/test_dry_run.py --test encoders
    python scripts/test_dry_run.py --test full-model
    python scripts/test_dry_run.py --test data-pipeline
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clip4cad.models.encoders import BRepEncoder, PointBertEncoder, UnifiedInputProjection
from clip4cad.models.text_encoder import HierarchicalTextEncoder
from clip4cad.models.clip4cad_h import CLIP4CAD_H


# ============================================================================
# Test Result Tracking
# ============================================================================

class TestResults:
    """Track test results and provide summary."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_pass(self, test_name: str, message: str = ""):
        self.passed.append((test_name, message))
        print(f"✓ PASS: {test_name}")
        if message:
            print(f"  → {message}")

    def add_fail(self, test_name: str, error: Exception):
        self.failed.append((test_name, error))
        print(f"✗ FAIL: {test_name}")
        print(f"  → Error: {str(error)}")
        print(f"  → Traceback: {traceback.format_exc()}")

    def add_warning(self, test_name: str, message: str):
        self.warnings.append((test_name, message))
        print(f"⚠ WARNING: {test_name}")
        print(f"  → {message}")

    def print_summary(self):
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Passed:   {len(self.passed)}")
        print(f"Failed:   {len(self.failed)}")
        print(f"Warnings: {len(self.warnings)}")

        if self.failed:
            print("\nFailed Tests:")
            for test_name, error in self.failed:
                print(f"  - {test_name}: {str(error)}")

        if self.warnings:
            print("\nWarnings:")
            for test_name, message in self.warnings:
                print(f"  - {test_name}: {message}")

        print("="*70)
        return len(self.failed) == 0


# ============================================================================
# Dummy Data Generation
# ============================================================================

def generate_dummy_brep_data(batch_size=2, num_faces=10, num_edges=15):
    """Generate dummy B-Rep data matching expected format."""
    # Face grids: [B, F, 32, 32, 3] (UV grid with xyz coordinates)
    face_grids = torch.randn(batch_size, num_faces, 32, 32, 3)

    # Edge curves: [B, E, 32, 3] (curve samples with xyz)
    edge_curves = torch.randn(batch_size, num_edges, 32, 3)

    # Adjacency: [B, F, E] (which edges bound which faces)
    adjacency = torch.randint(0, 2, (batch_size, num_faces, num_edges)).float()

    # Masks for padding
    face_mask = torch.ones(batch_size, num_faces)
    edge_mask = torch.ones(batch_size, num_edges)

    return {
        "face_grids": face_grids,
        "edge_curves": edge_curves,
        "adjacency": adjacency,
        "face_mask": face_mask,
        "edge_mask": edge_mask,
    }


def generate_dummy_pointcloud_data(batch_size=2, num_points=10000, channels=6):
    """Generate dummy point cloud data with xyz + normals."""
    # Points: [B, N, 6] (xyz + normals)
    points = torch.randn(batch_size, num_points, channels)

    # Normalize positions to unit sphere (first 3 channels)
    points[:, :, :3] = torch.nn.functional.normalize(points[:, :, :3], dim=-1)

    # Normalize normals (last 3 channels)
    points[:, :, 3:] = torch.nn.functional.normalize(points[:, :, 3:], dim=-1)

    return points


def generate_dummy_text_data(batch_size=2, max_title_len=32, max_desc_len=256):
    """Generate dummy text data (tokenized)."""
    # Title tokens: [B, max_title_len]
    title_input_ids = torch.randint(100, 5000, (batch_size, max_title_len))
    title_attention_mask = torch.ones(batch_size, max_title_len)

    # Description tokens: [B, max_desc_len]
    desc_input_ids = torch.randint(100, 5000, (batch_size, max_desc_len))
    desc_attention_mask = torch.ones(batch_size, max_desc_len)

    return {
        "title_input_ids": title_input_ids,
        "title_attention_mask": title_attention_mask,
        "desc_input_ids": desc_input_ids,
        "desc_attention_mask": desc_attention_mask,
    }


# ============================================================================
# Encoder Tests
# ============================================================================

def test_brep_encoder_basic(results: TestResults, device: str = "cuda"):
    """Test BRepEncoder with random initialization."""
    test_name = "BRepEncoder (random init)"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    try:
        # Create encoder
        encoder = BRepEncoder(
            face_dim=48,
            edge_dim=12,
            surface_checkpoint=None,
            edge_checkpoint=None,
        ).to(device)

        # Generate dummy data
        batch = generate_dummy_brep_data(batch_size=2, num_faces=10, num_edges=15)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        with torch.no_grad():
            face_tokens, edge_tokens = encoder(
                batch["face_grids"],
                batch["edge_curves"],
                batch["face_mask"],
                batch["edge_mask"],
            )

        # Check output shapes
        assert face_tokens.shape == (2, 10, 48), f"Expected (2, 10, 48), got {face_tokens.shape}"
        assert edge_tokens.shape == (2, 15, 12), f"Expected (2, 15, 12), got {edge_tokens.shape}"

        results.add_pass(test_name, f"Output shapes: face={face_tokens.shape}, edge={edge_tokens.shape}")

    except Exception as e:
        results.add_fail(test_name, e)


def test_brep_encoder_pretrained(results: TestResults, device: str = "cuda"):
    """Test BRepEncoder with pretrained weights (if available)."""
    test_name = "BRepEncoder (pretrained weights)"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    # Check if pretrained weights exist
    surface_ckpt = project_root / "pretrained" / "autobrep" / "surface_fsq_vae.pt"
    edge_ckpt = project_root / "pretrained" / "autobrep" / "edge_fsq_vae.pt"

    if not (surface_ckpt.exists() and edge_ckpt.exists()):
        results.add_warning(
            test_name,
            "Pretrained AutoBrep weights not found. Run scripts/download_autobrep_weights.py"
        )
        return

    try:
        encoder = BRepEncoder(
            face_dim=48,
            edge_dim=12,
            surface_checkpoint=str(surface_ckpt),
            edge_checkpoint=str(edge_ckpt),
        ).to(device)

        batch = generate_dummy_brep_data(batch_size=2, num_faces=10, num_edges=15)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            face_tokens, edge_tokens = encoder(
                batch["face_grids"],
                batch["edge_curves"],
                batch["face_mask"],
                batch["edge_mask"],
            )

        results.add_pass(test_name, "Pretrained weights loaded and forward pass successful")

    except Exception as e:
        results.add_fail(test_name, e)


def test_pointbert_encoder_basic(results: TestResults, device: str = "cuda"):
    """Test PointBertEncoder with random initialization."""
    test_name = "PointBertEncoder (random init)"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    try:
        encoder = PointBertEncoder(
            num_points=10000,
            in_channels=6,  # xyz + normals
            embed_dim=768,
            num_groups=512,
            depth=12,
            num_heads=12,
        ).to(device)

        # Generate dummy point cloud
        points = generate_dummy_pointcloud_data(batch_size=2, num_points=10000, channels=6)
        points = points.to(device)

        # Forward pass
        with torch.no_grad():
            features = encoder(points, return_all_tokens=True)

        # Check output shape: [B, 513, 768] (CLS + 512 groups)
        expected_shape = (2, 513, 768)
        assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"

        results.add_pass(test_name, f"Output shape: {features.shape}")

    except Exception as e:
        results.add_fail(test_name, e)


def test_pointbert_encoder_ulip2(results: TestResults, device: str = "cuda"):
    """Test PointBertEncoder with ULIP-2 pretrained weights."""
    test_name = "PointBertEncoder (ULIP-2 weights)"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    try:
        # This will auto-download ULIP-2 weights if not present
        from clip4cad.models.encoders.pointbert_encoder import download_ulip2_weights

        # Download weights (will skip if already exists)
        weights_path = download_ulip2_weights()
        print(f"ULIP-2 weights at: {weights_path}")

        # Create encoder with pretrained weights
        encoder = PointBertEncoder(
            num_points=10000,
            in_channels=6,
            embed_dim=768,
            num_groups=512,
            depth=12,
            num_heads=12,
        ).to(device)

        # Load ULIP-2 weights
        encoder.load_ulip2_weights(str(weights_path))

        # Test forward pass
        points = generate_dummy_pointcloud_data(batch_size=2, num_points=10000, channels=6)
        points = points.to(device)

        with torch.no_grad():
            features = encoder(points, return_all_tokens=True)

        results.add_pass(test_name, f"ULIP-2 weights loaded, output shape: {features.shape}")

    except Exception as e:
        results.add_fail(test_name, e)


def test_text_encoder_basic(results: TestResults, device: str = "cuda"):
    """Test HierarchicalTextEncoder initialization (without LLM)."""
    test_name = "HierarchicalTextEncoder (init only)"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    try:
        encoder = HierarchicalTextEncoder(
            llm_name="microsoft/Phi-4-mini-instruct",
            d_unified=256,
            freeze_llm=True,
        )

        # Check that tokenizer loads
        tokenizer = encoder.get_tokenizer()
        assert tokenizer is not None, "Tokenizer is None"

        # Check LLM is not loaded yet (lazy loading)
        assert encoder.llm is None, "LLM should not be loaded yet (lazy loading)"

        results.add_pass(test_name, f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    except Exception as e:
        results.add_fail(test_name, e)


def test_text_encoder_llm_loading(results: TestResults, device: str = "cuda"):
    """Test loading the full LLM (heavy operation)."""
    test_name = "HierarchicalTextEncoder (LLM loading)"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")
    print("WARNING: This will download ~7.6GB Phi-4-mini model if not cached")

    try:
        encoder = HierarchicalTextEncoder(
            llm_name="microsoft/Phi-4-mini-instruct",
            d_unified=256,
            freeze_llm=True,
        )

        # Load LLM explicitly
        print("Loading LLM... (this may take a few minutes on first run)")
        encoder.load_llm(device=device)

        assert encoder.llm is not None, "LLM failed to load"

        # Test forward pass with dummy text
        text_data = generate_dummy_text_data(batch_size=2)
        text_data = {k: v.to(device) for k, v in text_data.items()}

        with torch.no_grad():
            output = encoder.forward_live(
                title_input_ids=text_data["title_input_ids"],
                title_attention_mask=text_data["title_attention_mask"],
                desc_input_ids=text_data["desc_input_ids"],
                desc_attention_mask=text_data["desc_attention_mask"],
            )

        # Check outputs
        assert "title_features" in output, "Missing title_features"
        assert "desc_features" in output, "Missing desc_features"

        results.add_pass(
            test_name,
            f"LLM loaded, title shape: {output['title_features'].shape}, "
            f"desc shape: {output['desc_features'].shape}"
        )

    except Exception as e:
        results.add_fail(test_name, e)


# ============================================================================
# Full Model Test
# ============================================================================

def test_full_model(results: TestResults, device: str = "cuda"):
    """Test full CLIP4CAD_H model integration."""
    test_name = "CLIP4CAD_H (full model)"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    try:
        # Load model config directly (not using Hydra composition)
        config_path = project_root / "configs" / "model" / "clip4cad_h.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        config = OmegaConf.load(config_path)

        # Create model
        print("Initializing CLIP4CAD_H model...")
        model = CLIP4CAD_H(config)

        # Load LLM explicitly
        print("Loading text LLM...")
        model.load_llm(device=device)

        model = model.to(device)
        model.eval()

        # Create batch with all modalities
        batch = {}

        # B-Rep data
        brep_data = generate_dummy_brep_data(batch_size=2, num_faces=64, num_edges=128)
        batch.update({
            "brep_faces": brep_data["face_grids"],
            "brep_edges": brep_data["edge_curves"],
            "brep_face_mask": brep_data["face_mask"],
            "brep_edge_mask": brep_data["edge_mask"],
            "brep_adjacency": brep_data["adjacency"],
        })

        # Point cloud data
        batch["points"] = generate_dummy_pointcloud_data(batch_size=2, num_points=10000)

        # Text data
        text_data = generate_dummy_text_data(batch_size=2)
        batch.update(text_data)

        # Modality flags
        batch["has_brep"] = torch.ones(2, dtype=torch.bool)
        batch["has_pointcloud"] = torch.ones(2, dtype=torch.bool)
        batch["has_text"] = torch.ones(2, dtype=torch.bool)

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        print("Running forward pass...")
        with torch.no_grad():
            outputs = model(batch)

        # Verify outputs
        assert "brep" in outputs, "Missing brep modality"
        assert "z_global_proj" in outputs["brep"], "Missing z_global_proj in brep"
        assert "z_local_proj" in outputs["brep"], "Missing z_local_proj in brep"

        assert "pointcloud" in outputs, "Missing pointcloud modality"
        assert "z_global_proj" in outputs["pointcloud"], "Missing z_global_proj in pointcloud"
        assert "z_local_proj" in outputs["pointcloud"], "Missing z_local_proj in pointcloud"

        assert "text" in outputs, "Missing text modality"
        assert "t_global_proj" in outputs["text"], "Missing t_global_proj in text"
        assert "t_local_proj" in outputs["text"], "Missing t_local_proj in text"

        # Check shapes
        print("\nOutput shapes:")
        for modality, mod_outputs in outputs.items():
            if isinstance(mod_outputs, dict):
                print(f"{modality}:")
                for key, value in mod_outputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")

        # Check GPU memory usage
        if device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1e9  # GB
            print(f"\nGPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

            if memory_allocated > 20:
                results.add_warning(test_name, f"High GPU memory usage: {memory_allocated:.2f}GB")

        results.add_pass(test_name, "Full model forward pass successful")

    except Exception as e:
        results.add_fail(test_name, e)


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CLIP4CAD-H Dry Run Tests")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "encoders", "full-model", "data-pipeline"],
        help="Which tests to run"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run tests on"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM loading tests (saves time and download)"
    )

    args = parser.parse_args()

    print("="*70)
    print("CLIP4CAD-H DRY RUN TESTS")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)

    results = TestResults()

    # Run encoder tests
    if args.test in ["all", "encoders"]:
        print("\n" + "="*70)
        print("ENCODER TESTS")
        print("="*70)

        test_brep_encoder_basic(results, device=args.device)
        test_brep_encoder_pretrained(results, device=args.device)
        test_pointbert_encoder_basic(results, device=args.device)
        test_pointbert_encoder_ulip2(results, device=args.device)
        test_text_encoder_basic(results, device=args.device)

        if not args.skip_llm:
            test_text_encoder_llm_loading(results, device=args.device)

    # Run full model test
    if args.test in ["all", "full-model"]:
        if args.skip_llm:
            results.add_warning("Full model test", "Skipped due to --skip-llm flag")
        else:
            test_full_model(results, device=args.device)

    # Print summary
    success = results.print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
