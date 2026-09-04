from typing import Any, Optional

from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .lora_config import LoraConfig
from .rlve_config import RLVEConfig

__all__ = ["CreateModelRequest"]


class CreateModelRequest(StrictBase):
    session_id: str

    model_seq_id: int

    base_model: str
    """The name of the base model to fine-tune (e.g., 'Qwen/Qwen3-8B')."""

    user_metadata: Optional[dict[str, Any]] = None
    """Optional metadata about this model/training run, set by the end-user."""

    lora_config: Optional[LoraConfig] = None
    """LoRA configuration"""

    # --- TinkerCloud (GMI) additive fields: every one has a server default ---
    debug_train_only: bool = False
    """Skip inference-engine weight updates (debug/testing)."""

    checkpoint_path: Optional[str] = None
    """Load this checkpoint while creating the model (Megatron backends load at init)."""

    max_batch_size: int = 4096
    """Max datums per forward_backward without gradient accumulation."""

    max_seq_len: int = 2048
    """Max sequence length, used for parallelism decisions (context parallel above 2048)."""

    learning_rate: float = 1e-6
    """Optimizer learning rate fixed at model creation (Megatron backends)."""

    rlve_config: Optional[RLVEConfig] = None
    """Server-side RLVE / GRPO native options."""

    staleness_k: int = 0
    """Max sampler weight-version staleness the tenant accepts; 0 = strict on-policy."""

    objective: str = "language_modeling"
    """"language_modeling" | "sequence_classification" | "token_classification"."""

    num_labels: Optional[int] = None
    """Number of classes (classification objectives only)."""

    head_config: Optional[dict[str, Any]] = None
    """Classification head config, e.g. torch_dtype, target_modules."""

    type: Literal["create_model"] = "create_model"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
