from typing import Dict, List, Optional

from ..._models import BaseModel
from ..loss_fn_output import LossFnOutput


class OptimStepResponse(BaseModel):
    metrics: Optional[Dict[str, float]] = None
    """Optimization step metrics as key-value pairs"""

    loss_fn_outputs: Optional[List[LossFnOutput]] = None
    """Per-datum loss function outputs when the backend defers them from
    forward_backward to the optimizer step (forward_backward then returns an
    empty loss_fn_outputs list)."""
