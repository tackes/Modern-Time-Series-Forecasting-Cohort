"""
src/__init__.py
===============
Exposes the primary public interfaces of the src package.
Notebooks should import from here rather than from submodules directly
when using the convenience aliases below.

Direct submodule imports remain valid for advanced use:
    from src.evaluation import pooled_wmape
    from src.schemas import validate_forecast
"""

from src.checkpointing import load_checkpoint, list_checkpoints
from src.evaluation import score_forecasts, build_leaderboard
from src.schemas import validate_forecast, validate_score, validate_panel

__all__ = [
    # Checkpointing
    "load_checkpoint",
    "list_checkpoints",
    # Evaluation
    "score_forecasts",
    "build_leaderboard",
    # Schema validation
    "validate_forecast",
    "validate_score",
    "validate_panel",
]
