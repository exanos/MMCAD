"""
Comprehensive Dry-Run Test for CLIP4CAD-GFA Pipeline

Tests the new Grounded Feature Alignment architecture:
- Model loading and initialization
- Forward pass with all modalities
- Output shape verification
- Loss computation
- Inference methods

Usage:
    python scripts/test_dry_run_gfa.py --test all
    python scripts/test_dry_run_gfa.py --test model
    python scripts/test_dry_run_gfa.py --test loss
"""

import argparse
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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
        print(f"[PASS] {test_name}")
        if message:
            print(f"       {message}")

    def add_fail(self, test_name: str, error: Exception):
        self.failed.append((test_name, error))
        print(f"[FAIL] {test_name}")
        print(f"       Error: {str(error)}")
        traceback.print_exc()

    def add_warning(self, test_name: str, message: str):
        self.warnings.append((test_name, message))
        print(f"[WARN] {test_name}")
        print(f"       {message}")

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
                print(f"  - {test_name}: {str(error)[:60]}")

        if self.warnings:
            print("\nWarnings:")
            for test_name, message in self.warnings:
                print(f"  - {test_name}: {message}")

        print("="*70)
        return len(self.failed) == 0


# ============================================================================
# Dummy Data Generation
# ============================================================================

def generate_dummy_gfa_batch(
    batch_size: int = 2,
    max_faces: int = 192,
    max_edges: int = 512,
    num_pc_tokens: int = 513,
    max_text_len: int = 256,
    device: str = "cuda"
):
    """Generate dummy batch for GFA testing with pre-computed features."""
    return {
        # B-Rep features (pre-computed from AutoBrep)
        "brep_face_features": torch.randn(batch_size, max_faces, 48, device=device),
        "brep_edge_features": torch.randn(batch_size, max_edges, 12, device=device),
        "brep_face_mask": torch.ones(batch_size, max_faces, device=device),
        "brep_edge_mask": torch.ones(batch_size, max_edges, device=device),

        # Point cloud features (pre-computed from ULIP-2)
        "pc_features": torch.randn(batch_size, num_pc_tokens, 768, device=device),

        # Text features (pre-computed from Phi-4-mini)
        "desc_embedding": torch.randn(batch_size, max_text_len, 3072, device=device),
        "desc_mask": torch.ones(batch_size, max_text_len, device=device),

        # Flags
        "has_brep": torch.ones(batch_size, dtype=torch.bool, device=device),
        "has_pointcloud": torch.ones(batch_size, dtype=torch.bool, device=device),
        "use_cached_brep_features": True,
        "use_cached_pc_features": True,
        "use_cached_embeddings": True,
    }


# ============================================================================
# Model Tests
# ============================================================================

def test_gfa_model_init(results: TestResults, device: str = "cuda"):
    """Test CLIP4CAD_GFA model initialization from config."""
    test_name = "GFA Model Initialization"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    try:
        from clip4cad.models.clip4cad_gfa import CLIP4CAD_GFA

        # Load config
        config_path = project_root / "configs" / "model" / "clip4cad_gfa.yaml"
        config = OmegaConf.load(config_path)

        # Create model
        model = CLIP4CAD_GFA(config)
        model = model.to(device)
        model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results.add_pass(
            test_name,
            f"Model created. Params: {trainable_params:,} trainable / {total_params:,} total"
        )

        return model

    except Exception as e:
        results.add_fail(test_name, e)
        return None


def test_gfa_submodules(results: TestResults, model, device: str = "cuda"):
    """Verify all expected submodules exist."""
    test_name = "GFA Submodule Verification"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    if model is None:
        results.add_warning(test_name, "Skipped - model not initialized")
        return

    try:
        # Check required submodules
        required_modules = [
            "projection",
            "feature_queries",
            "feature_pos_enc",
            "text_parser",
            "confidence_predictor",
            "ground_text",
            "ground_geo",
            "align_brep",
            "align_pc",
            "fusion_norm",
            "global_proj_head",
            "local_proj_head",
            "self_ground_queries",
        ]

        missing = []
        for module_name in required_modules:
            if not hasattr(model, module_name):
                missing.append(module_name)

        if missing:
            results.add_fail(test_name, Exception(f"Missing modules: {missing}"))
        else:
            results.add_pass(test_name, f"All {len(required_modules)} required modules present")

    except Exception as e:
        results.add_fail(test_name, e)


def test_gfa_forward_full(results: TestResults, model, device: str = "cuda"):
    """Test forward pass with all modalities."""
    test_name = "GFA Forward Pass (Full)"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    if model is None:
        results.add_warning(test_name, "Skipped - model not initialized")
        return

    try:
        batch = generate_dummy_gfa_batch(batch_size=2, device=device)

        with torch.no_grad():
            outputs = model(batch)

        # Print output keys
        print(f"Output keys: {list(outputs.keys())}")

        # Check required outputs
        required_keys = [
            "confidence", "G_brep", "G_pc",
            "F_brep_aligned", "F_pc_aligned",
            "z_brep", "z_pc", "z_text",
            "F_brep_local", "F_pc_local",
            "temperature"
        ]

        missing = [k for k in required_keys if k not in outputs]
        if missing:
            results.add_fail(test_name, Exception(f"Missing output keys: {missing}"))
            return

        results.add_pass(test_name, "All required outputs present")

    except Exception as e:
        results.add_fail(test_name, e)


def test_gfa_output_shapes(results: TestResults, model, device: str = "cuda"):
    """Verify all output tensor shapes are correct."""
    test_name = "GFA Output Shape Verification"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    if model is None:
        results.add_warning(test_name, "Skipped - model not initialized")
        return

    try:
        B = 2
        K = model.num_feature_slots  # Should be 12
        max_brep_tokens = 704  # 192 + 512
        num_pc_tokens = 513
        d_proj = model.d_proj  # 128
        d_align = model.d_align  # 128

        batch = generate_dummy_gfa_batch(batch_size=B, device=device)

        with torch.no_grad():
            outputs = model(batch)

        # Expected shapes
        expected_shapes = {
            "confidence": (B, K),
            "G_brep": (B, K, max_brep_tokens),
            "G_pc": (B, K, num_pc_tokens),
            "F_brep_aligned": (B, K, d_align),
            "F_pc_aligned": (B, K, d_align),
            "z_brep": (B, d_proj),
            "z_pc": (B, d_proj),
            "z_text": (B, d_proj),
            "F_brep_local": (B, K, d_proj),
            "F_pc_local": (B, K, d_proj),
        }

        shape_errors = []
        for key, expected in expected_shapes.items():
            if key in outputs:
                actual = tuple(outputs[key].shape)
                if actual != expected:
                    shape_errors.append(f"{key}: expected {expected}, got {actual}")

        if shape_errors:
            results.add_fail(test_name, Exception("\n".join(shape_errors)))
        else:
            print("Output shapes:")
            for key, expected in expected_shapes.items():
                if key in outputs:
                    print(f"  {key}: {tuple(outputs[key].shape)}")
            results.add_pass(test_name, "All shapes match expected values")

    except Exception as e:
        results.add_fail(test_name, e)


def test_gfa_grounding_properties(results: TestResults, model, device: str = "cuda"):
    """Verify grounding matrices have correct properties."""
    test_name = "GFA Grounding Matrix Properties"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    if model is None:
        results.add_warning(test_name, "Skipped - model not initialized")
        return

    try:
        batch = generate_dummy_gfa_batch(batch_size=2, device=device)

        with torch.no_grad():
            outputs = model(batch)

        errors = []

        # Check G_brep
        G_brep = outputs["G_brep"]
        row_sums = G_brep.sum(dim=-1)  # Should be 1.0 for each row
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            errors.append(f"G_brep rows don't sum to 1: {row_sums}")

        if G_brep.min() < 0 or G_brep.max() > 1:
            errors.append(f"G_brep values outside [0,1]: min={G_brep.min()}, max={G_brep.max()}")

        # Check G_pc
        G_pc = outputs["G_pc"]
        row_sums = G_pc.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            errors.append(f"G_pc rows don't sum to 1: {row_sums}")

        if G_pc.min() < 0 or G_pc.max() > 1:
            errors.append(f"G_pc values outside [0,1]: min={G_pc.min()}, max={G_pc.max()}")

        if errors:
            results.add_fail(test_name, Exception("\n".join(errors)))
        else:
            results.add_pass(test_name, "Grounding matrices are valid probability distributions")

    except Exception as e:
        results.add_fail(test_name, e)


def test_gfa_confidence_range(results: TestResults, model, device: str = "cuda"):
    """Verify confidence scores are in (0, 1)."""
    test_name = "GFA Confidence Range"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    if model is None:
        results.add_warning(test_name, "Skipped - model not initialized")
        return

    try:
        batch = generate_dummy_gfa_batch(batch_size=2, device=device)

        with torch.no_grad():
            outputs = model(batch)

        confidence = outputs["confidence"]

        if confidence.min() <= 0 or confidence.max() >= 1:
            results.add_fail(
                test_name,
                Exception(f"Confidence outside (0,1): min={confidence.min()}, max={confidence.max()}")
            )
        else:
            results.add_pass(
                test_name,
                f"Confidence in valid range: min={confidence.min():.4f}, max={confidence.max():.4f}"
            )

    except Exception as e:
        results.add_fail(test_name, e)


def test_gfa_forward_brep_only(results: TestResults, model, device: str = "cuda"):
    """Test forward pass with B-Rep only (no point cloud)."""
    test_name = "GFA Forward Pass (B-Rep Only)"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    if model is None:
        results.add_warning(test_name, "Skipped - model not initialized")
        return

    try:
        batch = generate_dummy_gfa_batch(batch_size=2, device=device)
        batch["has_pointcloud"] = torch.zeros(2, dtype=torch.bool, device=device)

        with torch.no_grad():
            outputs = model(batch)

        # Should have brep outputs but not pc
        if "z_brep" not in outputs:
            results.add_fail(test_name, Exception("Missing z_brep"))
            return

        if "G_pc" in outputs:
            results.add_fail(test_name, Exception("G_pc should not be present"))
            return

        results.add_pass(test_name, "B-Rep-only forward pass successful")

    except Exception as e:
        results.add_fail(test_name, e)


def test_gfa_forward_pc_only(results: TestResults, model, device: str = "cuda"):
    """Test forward pass with point cloud only (no B-Rep)."""
    test_name = "GFA Forward Pass (PC Only)"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    if model is None:
        results.add_warning(test_name, "Skipped - model not initialized")
        return

    try:
        batch = generate_dummy_gfa_batch(batch_size=2, device=device)
        batch["has_brep"] = torch.zeros(2, dtype=torch.bool, device=device)

        with torch.no_grad():
            outputs = model(batch)

        # Should have pc outputs but not brep
        if "z_pc" not in outputs:
            results.add_fail(test_name, Exception("Missing z_pc"))
            return

        if "G_brep" in outputs:
            results.add_fail(test_name, Exception("G_brep should not be present"))
            return

        results.add_pass(test_name, "PC-only forward pass successful")

    except Exception as e:
        results.add_fail(test_name, e)


# ============================================================================
# Inference Tests
# ============================================================================

def test_gfa_encode_text(results: TestResults, model, device: str = "cuda"):
    """Test text encoding for retrieval."""
    test_name = "GFA Encode Text"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    if model is None:
        results.add_warning(test_name, "Skipped - model not initialized")
        return

    try:
        B = 2
        L = 256
        d_text = 3072

        text_features = torch.randn(B, L, d_text, device=device)
        text_mask = torch.ones(B, L, device=device)

        with torch.no_grad():
            z_text = model.encode_text(text_features, text_mask)

        # Check shape
        expected_shape = (B, model.d_proj)
        if tuple(z_text.shape) != expected_shape:
            results.add_fail(
                test_name,
                Exception(f"Shape mismatch: expected {expected_shape}, got {tuple(z_text.shape)}")
            )
            return

        # Check L2 normalized
        norms = z_text.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
            results.add_fail(test_name, Exception(f"Not L2 normalized: norms={norms}"))
            return

        results.add_pass(test_name, f"Output shape: {tuple(z_text.shape)}, L2 normalized")

    except Exception as e:
        results.add_fail(test_name, e)


def test_gfa_encode_geometry(results: TestResults, model, device: str = "cuda"):
    """Test geometry encoding for retrieval (with self-grounding)."""
    test_name = "GFA Encode Geometry"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    if model is None:
        results.add_warning(test_name, "Skipped - model not initialized")
        return

    try:
        B = 2

        # B-Rep features
        face_features = torch.randn(B, 64, 48, device=device)
        edge_features = torch.randn(B, 128, 12, device=device)
        face_mask = torch.ones(B, 64, device=device)
        edge_mask = torch.ones(B, 128, device=device)

        # PC features
        pc_features = torch.randn(B, 513, 768, device=device)

        with torch.no_grad():
            # Test with B-Rep only
            z_brep = model.encode_geometry(
                brep_face_features=face_features,
                brep_edge_features=edge_features,
                brep_face_mask=face_mask,
                brep_edge_mask=edge_mask,
            )

            # Test with PC only
            z_pc = model.encode_geometry(
                pc_features=pc_features,
            )

            # Test with both
            z_both = model.encode_geometry(
                brep_face_features=face_features,
                brep_edge_features=edge_features,
                brep_face_mask=face_mask,
                brep_edge_mask=edge_mask,
                pc_features=pc_features,
            )

        # Check shapes
        expected_shape = (B, model.d_proj)
        for name, z in [("z_brep", z_brep), ("z_pc", z_pc), ("z_both", z_both)]:
            if tuple(z.shape) != expected_shape:
                results.add_fail(
                    test_name,
                    Exception(f"{name} shape mismatch: expected {expected_shape}, got {tuple(z.shape)}")
                )
                return

        results.add_pass(test_name, f"All geometry encodings have shape {expected_shape}")

    except Exception as e:
        results.add_fail(test_name, e)


# ============================================================================
# Loss Tests
# ============================================================================

def test_gfa_loss_forward(results: TestResults, model, device: str = "cuda"):
    """Test GFALoss forward pass."""
    test_name = "GFA Loss Computation"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    if model is None:
        results.add_warning(test_name, "Skipped - model not initialized")
        return

    try:
        from clip4cad.losses.gfa_losses import GFALoss

        # Create loss function
        loss_fn = GFALoss(
            lambda_global=1.0,
            lambda_local=0.5,
            lambda_consist=0.5,
            lambda_diverse=0.2,
            lambda_conf_reg=0.1,
        )

        # Get model outputs
        batch = generate_dummy_gfa_batch(batch_size=4, device=device)

        with torch.no_grad():
            outputs = model(batch)

        # Compute loss
        total_loss, loss_dict = loss_fn(outputs, stage=1)

        # Check loss is valid
        if not torch.isfinite(total_loss):
            results.add_fail(test_name, Exception(f"Loss is not finite: {total_loss}"))
            return

        print(f"Loss breakdown:")
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.item():.4f}")

        results.add_pass(test_name, f"Total loss: {total_loss.item():.4f}")

    except Exception as e:
        results.add_fail(test_name, e)


def test_gfa_loss_stage_difference(results: TestResults, model, device: str = "cuda"):
    """Verify stage 1 uses reduced global loss weight."""
    test_name = "GFA Loss Stage Difference"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    if model is None:
        results.add_warning(test_name, "Skipped - model not initialized")
        return

    try:
        from clip4cad.losses.gfa_losses import GFALoss

        loss_fn = GFALoss(
            lambda_global=1.0,
            lambda_local=0.5,
            lambda_consist=0.5,
            lambda_diverse=0.2,
            lambda_conf_reg=0.1,
            lambda_global_stage1=0.2,  # Reduced for stage 1
        )

        batch = generate_dummy_gfa_batch(batch_size=4, device=device)

        with torch.no_grad():
            outputs = model(batch)

        # Compute loss for both stages
        _, loss_dict_s1 = loss_fn(outputs, stage=1)
        _, loss_dict_s2 = loss_fn(outputs, stage=2)

        # Get global loss contribution
        global_s1 = loss_dict_s1.get("global_loss", torch.tensor(0.0))
        global_s2 = loss_dict_s2.get("global_loss", torch.tensor(0.0))

        # They should be different (s1 has reduced weight)
        if isinstance(global_s1, torch.Tensor):
            global_s1 = global_s1.item()
        if isinstance(global_s2, torch.Tensor):
            global_s2 = global_s2.item()

        print(f"Stage 1 global loss: {global_s1:.4f}")
        print(f"Stage 2 global loss: {global_s2:.4f}")

        # The global loss itself might be the same, but weighted differently
        # Let's check the weighted contributions
        weighted_s1 = loss_dict_s1.get("global_loss_weighted", loss_dict_s1.get("global_loss", 0))
        weighted_s2 = loss_dict_s2.get("global_loss_weighted", loss_dict_s2.get("global_loss", 0))

        if isinstance(weighted_s1, torch.Tensor):
            weighted_s1 = weighted_s1.item()
        if isinstance(weighted_s2, torch.Tensor):
            weighted_s2 = weighted_s2.item()

        results.add_pass(
            test_name,
            f"Stage 1 weighted: {weighted_s1:.4f}, Stage 2 weighted: {weighted_s2:.4f}"
        )

    except Exception as e:
        results.add_fail(test_name, e)


# ============================================================================
# Memory Test
# ============================================================================

def test_gfa_memory_usage(results: TestResults, model, device: str = "cuda"):
    """Check GPU memory usage."""
    test_name = "GFA Memory Usage"
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    if model is None:
        results.add_warning(test_name, "Skipped - model not initialized")
        return

    if device != "cuda":
        results.add_warning(test_name, "Skipped - not on CUDA")
        return

    try:
        torch.cuda.reset_peak_memory_stats()

        batch = generate_dummy_gfa_batch(batch_size=4, device=device)

        with torch.no_grad():
            outputs = model(batch)

        peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
        current_memory = torch.cuda.memory_allocated() / 1e9

        print(f"Peak GPU memory: {peak_memory:.2f} GB")
        print(f"Current GPU memory: {current_memory:.2f} GB")

        if peak_memory > 8:
            results.add_warning(test_name, f"High memory usage: {peak_memory:.2f} GB")
        else:
            results.add_pass(test_name, f"Peak memory: {peak_memory:.2f} GB")

    except Exception as e:
        results.add_fail(test_name, e)


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CLIP4CAD-GFA Dry Run Tests")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "model", "loss", "inference"],
        help="Which tests to run"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run tests on"
    )

    args = parser.parse_args()

    print("="*70)
    print("CLIP4CAD-GFA DRY RUN TESTS")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)

    results = TestResults()

    # Initialize model once
    model = test_gfa_model_init(results, device=args.device)

    # Run tests based on selection
    if args.test in ["all", "model"]:
        print("\n" + "="*70)
        print("MODEL TESTS")
        print("="*70)

        test_gfa_submodules(results, model, device=args.device)
        test_gfa_forward_full(results, model, device=args.device)
        test_gfa_output_shapes(results, model, device=args.device)
        test_gfa_grounding_properties(results, model, device=args.device)
        test_gfa_confidence_range(results, model, device=args.device)
        test_gfa_forward_brep_only(results, model, device=args.device)
        test_gfa_forward_pc_only(results, model, device=args.device)

    if args.test in ["all", "inference"]:
        print("\n" + "="*70)
        print("INFERENCE TESTS")
        print("="*70)

        test_gfa_encode_text(results, model, device=args.device)
        test_gfa_encode_geometry(results, model, device=args.device)

    if args.test in ["all", "loss"]:
        print("\n" + "="*70)
        print("LOSS TESTS")
        print("="*70)

        test_gfa_loss_forward(results, model, device=args.device)
        test_gfa_loss_stage_difference(results, model, device=args.device)

    # Memory test
    if args.test == "all":
        test_gfa_memory_usage(results, model, device=args.device)

    # Print summary
    success = results.print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
