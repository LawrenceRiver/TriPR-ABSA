"""Model-agnostic pragmatic residual modules for ABSA logits."""

from .apply import apply_batch, apply_pragmatic_residual

__all__ = ["apply_batch", "apply_pragmatic_residual"]
