"""
Loss Functions for CLIP4CAD Training

Three loss configurations available:
1. CLIP4CADLoss: For CLIP4CAD-H (hierarchical compression)
2. GFALoss: For CLIP4CAD-GFA (grounded feature alignment)
3. HUSLoss: For CLIP4CAD-HUS (hierarchical unified space)

Components:
- Global contrastive loss (InfoNCE across modalities)
- Local contrastive loss (Hungarian matching with confidence)
- Reconstruction loss (auxiliary regularization)
- Grounding consistency loss (for GFA)
- Grounding diversity loss (for GFA)
"""

from .combined import CLIP4CADLoss
from .infonce import InfoNCELoss
from .local_matching import LocalMatchingLoss
from .reconstruction import ReconstructionLoss
from .gfa_losses import (
    GFALoss,
    GroundingConsistencyLoss,
    GroundingDiversityLoss,
    SlotContrastiveLoss,
)
from .gfa_v2_losses import GFAv2Loss, compute_self_grounding_quality
from .gfa_v2_4_losses import GFAv2_4Loss, compute_self_grounding_quality as compute_self_grounding_quality_v2_4
from .gfa_v4_losses import GFAv4Loss, compute_self_grounding_quality as compute_self_grounding_quality_v4, compute_query_alignment
from .gfa_v4_2_losses import GFAv4_2Loss, compute_self_grounding_quality as compute_self_grounding_quality_v4_2, compute_query_alignment as compute_query_alignment_v4_2
from .gfa_v4_4_losses import (
    GFAv44Loss,
    compute_self_grounding_quality as compute_self_grounding_quality_v44,
    compute_query_alignment as compute_query_alignment_v44,
    compute_retrieval_metrics,
)
from .hus_losses import HUSLoss, HUSLossConfig
from .gfa_v4_8_losses import (
    GFAv48Loss,
    compute_modality_gap,
    compute_true_pair_cosine,
    compute_code_diversity,
    mine_hard_negatives_by_code,
)
from .gfa_v4_8_1_losses import (
    GFAv481Loss,
    compute_modality_gap as compute_modality_gap_v481,
    compute_true_pair_cosine as compute_true_pair_cosine_v481,
    compute_brep_pc_metrics,
    compute_code_diversity as compute_code_diversity_v481,
    compute_active_codes,
    mine_hard_negatives_by_code as mine_hard_negatives_by_code_v481,
)
from .gfa_v4_8_2_losses import (
    GFAv482LossSmooth,
    compute_modality_gap as compute_modality_gap_v482,
    compute_true_pair_cosine as compute_true_pair_cosine_v482,
    compute_brep_pc_metrics as compute_brep_pc_metrics_v482,
    compute_code_diversity as compute_code_diversity_v482,
    compute_active_codes as compute_active_codes_v482,
    mine_hard_negatives_by_code as mine_hard_negatives_by_code_v482,
    get_warmup_cosine_scheduler,
)
from .gfa_v4_8_3_losses import (
    GFAv483Loss,
    compute_cross_modal_metrics,
    compute_retrieval_metrics as compute_retrieval_metrics_v483,
)
from .gfa_v4_9_losses import (
    GFAv49Loss,
    compute_retrieval_metrics as compute_retrieval_metrics_v49,
    compute_contrastive_quality,
    compute_batch_r1,
    diagnose_embeddings,
    mine_hard_negatives_simple,
    get_warmup_cosine_scheduler as get_warmup_cosine_scheduler_v49,
    get_param_groups_with_lr,
)

__all__ = [
    # Main loss functions
    "CLIP4CADLoss",
    "GFALoss",
    "GFAv2Loss",
    "GFAv2_4Loss",
    "GFAv4Loss",
    "GFAv4_2Loss",
    "GFAv44Loss",
    "GFAv48Loss",
    "HUSLoss",
    "HUSLossConfig",
    # Component losses
    "InfoNCELoss",
    "LocalMatchingLoss",
    "ReconstructionLoss",
    "GroundingConsistencyLoss",
    "GroundingDiversityLoss",
    "SlotContrastiveLoss",
    # Utilities
    "compute_self_grounding_quality",
    "compute_query_alignment",
    "compute_retrieval_metrics",
    # v4.8 utilities
    "compute_modality_gap",
    "compute_true_pair_cosine",
    "compute_code_diversity",
    "mine_hard_negatives_by_code",
    # v4.8.1 model and utilities
    "GFAv481Loss",
    "compute_brep_pc_metrics",
    "compute_active_codes",
    # v4.8.2 model and utilities
    "GFAv482LossSmooth",
    "get_warmup_cosine_scheduler",
    # v4.8.3 model and utilities (fixed cross-modal)
    "GFAv483Loss",
    "compute_cross_modal_metrics",
    # v4.9 model and utilities (no codebook)
    "GFAv49Loss",
    "compute_contrastive_quality",
    "compute_batch_r1",
    "diagnose_embeddings",
    "mine_hard_negatives_simple",
]
