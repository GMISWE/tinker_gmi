from typing import List, Optional

from .._models import StrictBase

__all__ = ["RLVEConfig"]


class RLVEConfig(StrictBase):
    """RLVE configuration for server-side GRPO native options.

    When passed to create_model, enables RLVE-specific training parameters:
    - use-tis: Truncated Importance Sampling
    - balance-data: Balance data across environments
    - partial-rollout: Enable partial rollout support
    - over-sampling-batch-size: Over-sampling batch size for dynamic sampling
    - num-rollout: Number of rollouts
    - dynamic-sampling-filter: Filter for non-zero reward std

    For client-side RLVE (tinker-cookbook), environment_list can be empty
    since problem generation happens on the client.
    """

    enabled: bool = True

    # Environment settings (can be empty for client-side RLVE)
    environment_list: List[str] = []

    # Batch configuration
    rollout_batch_size: int = 128
    n_samples_per_prompt: int = 16

    # Response length and temperature
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0

    # GRPO native options
    num_rollout: int = 500
    over_sampling_batch_size: int = 384
    balance_data: bool = True
    partial_rollout: bool = True
    use_dynamic_sampling_filter: bool = True
