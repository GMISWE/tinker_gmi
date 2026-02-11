from typing import Dict, List, Optional

from .._models import BaseModel
from .loss_fn_output import LossFnOutput

__all__ = ["OptimStepResponse"]


class OptimStepResponse(BaseModel):
    metrics: Optional[Dict[str, float]] = None
    """Optimization step metrics as key-value pairs"""

    loss_fn_outputs: Optional[List[LossFnOutput]] = None
    """Per-sample loss function outputs (e.g. training logprobs).
    Populated by backends that compute logprobs during optimizer step
    rather than during forward_backward (e.g. NeMo RL deferred mode)."""
