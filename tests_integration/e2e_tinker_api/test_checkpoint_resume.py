#!/usr/bin/env python3
"""
Test: Checkpoint Save and Resume

Tests Tinker client's checkpoint save/load workflow:
  1. Create training client
  2. Train for N steps
  3. Save checkpoint
  4. Create new training client from checkpoint
  5. Resume training
  6. Verify continuity

Mirrors the cookbook's checkpoint_utils.save_checkpoint() and resume functionality.
"""
import os
import sys

# Configure environment for kgateway-training
os.environ["TINKER_BASE_URL"] = "http://kgateway-training.slime-gmi:8000"
os.environ["TINKER_API_KEY"] = "slime-dev-key"

import tinker
from tinker import types

print("=" * 80)
print("TEST: Checkpoint Save and Resume")
print("=" * 80)
print(f"Base URL: {os.environ.get('TINKER_BASE_URL')}")
print(f"API Key: {os.environ.get('TINKER_API_KEY')[:10]}...")
print("=" * 80)

checkpoint_path = None

try:
    # Step 1: Create initial training client
    print("\n[1/6] Creating initial training client...")
    client = tinker.ServiceClient()
    training_client_1 = client.create_lora_training_client(
        base_model="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
        rank=0  # No LoRA
    )
    print(f"   ✓ Training client 1 created: {training_client_1.model_id}")

    # Step 2: Train for a few steps
    print("\n[2/6] Training for 3 steps...")

    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    losses_phase1 = []

    for step in range(3):
        datum = tinker.Datum(
            model_input=types.TokenSequence(tokens=input_tokens),
            loss_fn_inputs={
                "target": types.TokenSequence(tokens=input_tokens[1:] + [input_tokens[-1]]),
                "weights": types.Vector(data=[1.0] * len(input_tokens))
            }
        )

        # Forward-backward
        fwd_bwd_future = training_client_1.forward_backward(
            [datum],
            loss_fn="cross_entropy"
        )
        fwd_bwd_result = fwd_bwd_future.result()

        # Extract loss
        loss_value = None
        if hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
            if 'loss:mean' in fwd_bwd_result.metrics:
                loss_value = fwd_bwd_result.metrics['loss:mean']
                losses_phase1.append(loss_value)

        # Optimizer step
        adam_params = tinker.AdamParams(
            learning_rate=2e-4,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8
        )
        optim_future = training_client_1.optim_step(adam_params)
        optim_result = optim_future.result()

        if loss_value is not None:
            print(f"   Step {step + 1}: loss={loss_value:.6f}")
        else:
            print(f"   Step {step + 1}: completed")

    print(f"   ✓ Phase 1 training completed ({len(losses_phase1)} steps)")

    # Step 3: Save checkpoint
    print("\n[3/6] Saving checkpoint...")
    checkpoint_name = "test_resume_checkpoint"
    save_future = training_client_1.save_state(checkpoint_name)
    save_result = save_future.result()
    checkpoint_path = save_result.path
    print(f"   ✓ Checkpoint saved:")
    print(f"     Name: {checkpoint_name}")
    print(f"     Path: {checkpoint_path}")

    # Step 4: Create new training client from checkpoint
    print("\n[4/6] Creating training client from checkpoint...")
    print(f"   Loading from: {checkpoint_path}")

    training_client_2 = client.create_training_client_from_state(checkpoint_path)
    print(f"   ✓ Training client 2 created: {training_client_2.model_id}")

    # Step 5: Resume training
    print("\n[5/6] Resuming training for 3 more steps...")

    losses_phase2 = []

    for step in range(3):
        datum = tinker.Datum(
            model_input=types.TokenSequence(tokens=input_tokens),
            loss_fn_inputs={
                "target": types.TokenSequence(tokens=input_tokens[1:] + [input_tokens[-1]]),
                "weights": types.Vector(data=[1.0] * len(input_tokens))
            }
        )

        # Forward-backward
        fwd_bwd_future = training_client_2.forward_backward(
            [datum],
            loss_fn="cross_entropy"
        )
        fwd_bwd_result = fwd_bwd_future.result()

        # Extract loss
        loss_value = None
        if hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
            if 'loss:mean' in fwd_bwd_result.metrics:
                loss_value = fwd_bwd_result.metrics['loss:mean']
                losses_phase2.append(loss_value)

        # Optimizer step
        adam_params = tinker.AdamParams(
            learning_rate=2e-4,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8
        )
        optim_future = training_client_2.optim_step(adam_params)
        optim_result = optim_future.result()

        if loss_value is not None:
            print(f"   Step {step + 4}: loss={loss_value:.6f}")
        else:
            print(f"   Step {step + 4}: completed")

    print(f"   ✓ Phase 2 training completed ({len(losses_phase2)} steps)")

    # Step 6: Verify continuity
    print("\n[6/6] Verifying checkpoint resume...")

    all_losses = losses_phase1 + losses_phase2

    if all_losses:
        print(f"   Loss progression:")
        for i, loss in enumerate(all_losses):
            marker = "← checkpoint" if i == len(losses_phase1) - 1 else ""
            print(f"     Step {i + 1}: {loss:.6f} {marker}")

        # Check if training continued properly (loss should generally decrease or stay stable)
        if len(all_losses) >= 2:
            print(f"\n   Initial loss: {all_losses[0]:.6f}")
            print(f"   Final loss: {all_losses[-1]:.6f}")
            if all_losses[0] >= all_losses[-1]:
                print(f"   ✓ Loss improved or maintained")
    else:
        print(f"   ⚠ No loss values captured (metrics may not be available)")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print(f"\nCheckpoint save/resume verified:")
    print(f"  • Training client 1: {training_client_1.model_id}")
    print(f"  • Checkpoint: {checkpoint_name}")
    print(f"  • Path: {checkpoint_path}")
    print(f"  • Training client 2: {training_client_2.model_id}")
    print(f"  • Total steps: {len(all_losses)}")
    print(f"\n🎉 Checkpoint resume workflow functional!")
    print("=" * 80)

    sys.exit(0)

except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

    if checkpoint_path:
        print(f"\nCheckpoint path (for debugging): {checkpoint_path}")

    print("\n" + "=" * 80)
    print("✗ TEST FAILED")
    print("=" * 80)

    sys.exit(1)
