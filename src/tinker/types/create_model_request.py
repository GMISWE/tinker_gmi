from typing import Any, Optional

from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .lora_config import LoraConfig

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

    # GMI-specific: enable debug mode for testing
    debug_train_only: bool = False

    # GMI-specific: load checkpoint during model creation (for Slime/Megatron)
    checkpoint_path: Optional[str] = None

    type: Literal["create_model"] = "create_model"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
