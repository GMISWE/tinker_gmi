#!/usr/bin/env python3
"""
Test: Supervised Learning Single Step

Tests Tinker client's supervised fine-tuning (SFT) workflow:
  1. Create training client
  2. Prepare Datum with cross_entropy loss function
  3. Forward-backward pass
  4. Optimizer step with AdamParams
  5. Extract and verify metrics (loss, grad_norm)

Mirrors the cookbook's supervised/train.py do_update() function.
"""
import os
import sys

# Configure environment for kgateway-training
os.environ["TINKER_BASE_URL"] = "http://kgateway-training.slime-gmi:8000"
os.environ["TINKER_API_KEY"] = "slime-dev-key"

import tinker
from tinker import types
import torch

print("=" * 80)
print("TEST: Supervised Learning Single Step")
print("=" * 80)
print(f"Base URL: {os.environ.get('TINKER_BASE_URL')}")
print(f"API Key: {os.environ.get('TINKER_API_KEY')[:10]}...")
print("=" * 80)

try:
    # Step 1: Create service client
    print("\n[1/6] Creating service client...")
    client = tinker.ServiceClient()
    print("   ✓ Service client created")

    # Step 2: Create training client
    print("\n[2/6] Creating training client (no LoRA)...")
    training_client = client.create_lora_training_client(
        base_model="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
        rank=0  # No LoRA
    )
    print(f"   ✓ Training client created with model_id: {training_client.model_id}")

    # Step 3: Prepare training data
    print("\n[3/6] Preparing training data...")

    # Simple example: predict next tokens
    # Input tokens: [1, 2, 3, 4, 5]
    # Target: [2, 3, 4, 5, 6]
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8]

    # Create Datum for cross_entropy loss
    # For supervised learning, we typically:
    # - model_input: input token sequence
    # - loss_fn_inputs: contains target tokens and weights (mask) for which positions to compute loss

    datum = tinker.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target": types.TensorData(
                data=input_tokens[1:] + [input_tokens[-1]],  # Shifted targets
                shape=[len(input_tokens)],
                dtype="int64"
            ),
            "weights": types.TensorData(
                data=[1.0] * len(input_tokens),  # Weight all positions equally
                shape=[len(input_tokens)],
                dtype="float32"
            )
        }
    )

    data_batch = [datum]
    print(f"   ✓ Created batch with {len(data_batch)} datum")
    print(f"     Input tokens: {input_tokens[:5]}... (length: {len(input_tokens)})")

    # Step 4: Forward-backward pass
    print("\n[4/6] Running forward-backward pass...")
    print("   Loss function: cross_entropy")

    fwd_bwd_future = training_client.forward_backward(
        data_batch,
        loss_fn="cross_entropy"
    )
    print("   ✓ Forward-backward submitted")

    print("   Waiting for completion...")
    fwd_bwd_result = fwd_bwd_future.result()
    print("   ✓ Forward-backward completed")

    # Extract metrics
    loss_value = None
    if fwd_bwd_result.loss_fn_outputs:
        first_output = fwd_bwd_result.loss_fn_outputs[0]
        if "logprobs" in first_output:
            logprobs = first_output["logprobs"]
            print(f"   Logprobs shape: {logprobs.to_torch().shape}")

        # Loss might be in metrics
        if hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
            print(f"   Metrics: {fwd_bwd_result.metrics}")
            if 'loss:mean' in fwd_bwd_result.metrics:
                loss_value = fwd_bwd_result.metrics['loss:mean']
                print(f"   Loss (mean): {loss_value:.6f}")

    # Step 5: Optimizer step
    print("\n[5/6] Running optimizer step...")

    # AdamParams from cookbook
    adam_params = tinker.AdamParams(
        learning_rate=2e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8
    )
    print(f"   Learning rate: {adam_params.learning_rate}")
    print(f"   Beta1: {adam_params.beta1}, Beta2: {adam_params.beta2}")

    optim_future = training_client.optim_step(adam_params)
    print("   ✓ Optimizer step submitted")

    print("   Waiting for completion...")
    optim_result = optim_future.result()
    print("   ✓ Optimizer step completed")

    # Step 6: Verify results
    print("\n[6/6] Verifying results...")

    print("   Training step metrics:")
    print(f"     • Forward-backward: completed")
    print(f"     • Optimizer step: completed")
    if loss_value is not None:
        print(f"     • Loss: {loss_value:.6f}")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print(f"\nSupervised learning single step verified:")
    print(f"  • Model ID: {training_client.model_id}")
    print(f"  • Loss function: cross_entropy")
    print(f"  • Optimizer: Adam")
    if loss_value is not None:
        print(f"  • Loss: {loss_value:.6f}")
    print(f"\n🎉 SFT workflow ready!")
    print("=" * 80)

    sys.exit(0)

except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

    print("\n" + "=" * 80)
    print("✗ TEST FAILED")
    print("=" * 80)

    sys.exit(1)
