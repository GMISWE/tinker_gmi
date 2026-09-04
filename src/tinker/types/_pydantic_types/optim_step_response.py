from typing import Dict, List, Optional

from ..._models import BaseModel
from .tensor_data import TensorData


class OptimStepResponse(BaseModel):
    metrics: Optional[Dict[str, float]] = None
    """Optimization step metrics as key-value pairs"""

    # The pydantic TensorData (this response is parsed from JSON); it carries
    # to_torch / to_numpy like the public dataclass.
    loss_fn_outputs: Optional[List[Dict[str, TensorData]]] = None
    """Per-datum loss function outputs when the backend defers them from
    forward_backward to the optimizer step (forward_backward then returns an
    empty loss_fn_outputs list)."""
