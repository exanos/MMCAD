#!/usr/bin/env python3
"""
Quick test script to verify data loading works with pre-split files.
Run this BEFORE full training to catch any issues.

Usage:
    python scripts/test_data_loading.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clip4cad.data.gfa_dataset import GFAMappedDataset


def test_loading():
    """Test loading train and val datasets."""
    print("="*60)
    print("Testing Data Loading with Pre-Split Files")
    print("="*60)

    # Paths (same as training)
    data_root = Path("d:/Defect_Det/MMCAD/data")
    pc_file = Path("c:/Users/User/Desktop/pc_embeddings_full.h5")
    brep_file = Path("c:/Users/User/Desktop/brep_features.h5")
    text_file = Path("c:/Users/User/Desktop/text_embeddings.h5")

    print("\n1. Testing TRAIN dataset loading...")
    print("-" * 60)
    try:
        train_dataset = GFAMappedDataset(
            data_root=data_root,
            split='train',
            brep_file=brep_file,
            text_file=text_file,
            pc_file=pc_file,
            use_live_text=False,
            load_to_memory=True,
            num_rotations=1,
        )
        print(f"✓ Train dataset loaded: {len(train_dataset)} samples")

        # Try loading one sample
        print("\n  Testing sample access...")
        sample = train_dataset[0]
        print(f"  ✓ Sample 0 loaded successfully")
        print(f"    Keys: {list(sample.keys())}")
        print(f"    desc_embedding shape: {sample['desc_embedding'].shape}")
        print(f"    brep_face_features shape: {sample['brep_face_features'].shape}")
        print(f"    pc_features shape: {sample['pc_features'].shape}")

    except Exception as e:
        print(f"✗ Train dataset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n2. Testing VAL dataset loading (from disk, not RAM)...")
    print("-" * 60)
    try:
        val_dataset = GFAMappedDataset(
            data_root=data_root,
            split='val',
            brep_file=brep_file,
            text_file=text_file,
            pc_file=pc_file,
            use_live_text=False,
            load_to_memory=False,  # Val stays on disk to save RAM
            num_rotations=1,
        )
        print(f"✓ Val dataset loaded: {len(val_dataset)} samples")

        # Try loading one sample
        print("\n  Testing sample access...")
        sample = val_dataset[0]
        print(f"  ✓ Sample 0 loaded successfully")

    except Exception as e:
        print(f"✗ Val dataset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nYou can now:")
    print("1. Delete the original 390GB file:")
    print("   del c:/Users/User/Desktop/text_embeddings.h5")
    print("\n2. Run full training:")
    print("   python scripts/train_gfa.py \\")
    print("       --data-root d:/Defect_Det/MMCAD/data \\")
    print("       --pc-file c:/Users/User/Desktop/pc_embeddings_full.h5 \\")
    print("       --brep-file c:/Users/User/Desktop/brep_features.h5 \\")
    print("       --text-file c:/Users/User/Desktop/text_embeddings.h5 \\")
    print("       --output-dir outputs/gfa_111k \\")
    print("       --batch-size 512 \\")
    print("       --load-to-memory \\")
    print("       --num-workers 2")

    return True


if __name__ == '__main__':
    success = test_loading()
    sys.exit(0 if success else 1)
