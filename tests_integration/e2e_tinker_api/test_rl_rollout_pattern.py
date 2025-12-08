#!/usr/bin/env python3
"""
Test: RL Rollout Pattern

Tests Tinker client's RL training workflow:
  1. Create training client
  2. Create sampling client for rollouts
  3. Generate rollouts (sampling)
  4. Prepare training data with advantages
  5. Forward-backward with importance_sampling
  6. Optimizer step
  7. Update sampling client for next iteration

Mirrors the cookbook's rl/train.py train_step() and do_group_rollout() functions.
"""
import os
import sys

# Configure environment for kgateway-training
os.environ["TINKER_BASE_URL"] = "http://kgateway-training.slime-gmi:8000"
os.environ["TINKER_API_KEY"] = "slime-dev-key"

import tinker
from tinker import types
import time

print("=" * 80)
print("TEST: RL Rollout Pattern")
print("=" * 80)
print(f"Base URL: {os.environ.get('TINKER_BASE_URL')}")
print(f"API Key: {os.environ.get('TINKER_API_KEY')[:10]}...")
print("=" * 80)

try:
    # Step 1: Create service client
    print("\n[1/7] Creating service client...")
    client = tinker.ServiceClient()
    print("   ✓ Service client created")

    # Step 2: Create training client
    print("\n[2/7] Creating training client (no LoRA)...")
    training_client = client.create_lora_training_client(
        base_model="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
        rank=0  # No LoRA
    )
    print(f"   ✓ Training client created: {training_client.model_id}")

    # Step 3: Create initial sampling client
    print("\n[3/7] Creating sampling client for rollouts...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="rl_iteration_0"
    )
    print("   ✓ Sampling client created")

    # Wait for SGLang to be ready
    print("   Waiting for SGLang router (15 seconds)...")
    time.sleep(15)

    # Step 4: Generate rollouts (samples)
    print("\n[4/7] Generating rollouts...")

    prompt_tokens = [1, 2, 3, 4, 5]
    prompt = types.ModelInput.from_ints(prompt_tokens)

    sampling_params = types.SamplingParams(
        max_tokens=20,
        temperature=0.7,
        top_p=0.9
    )

    print(f"   Prompt: {prompt_tokens}")
    print(f"   Num samples: 2")

    sample_future = sampling_client.sample(
        prompt=prompt,
        num_samples=2,
        sampling_params=sampling_params
    )

    sample_response = sample_future.result()
    samples = sample_response.sequences
    print(f"   ✓ Generated {len(samples)} rollout samples")

    for i, sample in enumerate(samples):
        print(f"     Sample {i + 1}: {len(sample.tokens)} tokens, stop={sample.stop_reason}")

    # Step 5: Prepare training data with advantages
    print("\n[5/7] Preparing training data with advantages...")

    # In RL, we typically:
    # - Use the sampled trajectories
    # - Compute advantages (rewards - baseline)
    # - Train with importance_sampling loss

    # For this test, we'll use simple mock advantages
    training_data = []

    for i, sample in enumerate(samples):
        # Mock advantage (in real RL, this comes from reward function)
        advantage = 1.0 if i == 0 else -0.5  # Prefer first sample

        # Sample tokens become training input
        sample_tokens = sample.tokens[:10]  # Limit length for test

        datum = tinker.Datum(
            model_input=types.ModelInput.from_ints(sample_tokens),
            loss_fn_inputs={
                "target": types.TensorData(
                    data=sample_tokens[1:] + [sample_tokens[-1]],
                    shape=[len(sample_tokens)],
                    dtype="int64"
                ),
                "weights": types.TensorData(
                    data=[1.0] * len(sample_tokens),
                    shape=[len(sample_tokens)],
                    dtype="float32"
                ),
                "advantages": types.TensorData(
                    data=[advantage] * len(sample_tokens),
                    shape=[len(sample_tokens)],
                    dtype="float32"
                ),
                # For importance sampling, we also need log_probs from sampling
                # In real RL, these come from the rollout
                "log_probs": types.TensorData(
                    data=[0.0] * len(sample_tokens),
                    shape=[len(sample_tokens)],
                    dtype="float32"
                )  # Mock values
            }
        )
        training_data.append(datum)

    print(f"   ✓ Prepared {len(training_data)} training datums")
    print(f"     Advantages: [1.0, -0.5] (mock values)")

    # Step 6: Forward-backward with importance_sampling
    print("\n[6/7] Running forward-backward with importance_sampling...")

    fwd_bwd_future = training_client.forward_backward(
        training_data,
        loss_fn="importance_sampling"
    )
    print("   ✓ Forward-backward submitted")

    fwd_bwd_result = fwd_bwd_future.result()
    print("   ✓ Forward-backward completed")

    # Extract metrics
    if hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
        print(f"   Metrics:")
        for key, value in fwd_bwd_result.metrics.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.6f}")
            else:
                print(f"     {key}: {value}")

    # Optimizer step
    print("\n   Running optimizer step...")
    adam_params = tinker.AdamParams(
        learning_rate=4e-5,  # RL typically uses lower LR
        beta1=0.9,
        beta2=0.95,
        eps=1e-8
    )
    optim_future = training_client.optim_step(adam_params)
    optim_result = optim_future.result()
    print("   ✓ Optimizer step completed")

    # Step 7: Update sampling client for next iteration
    print("\n[7/7] Updating sampling client for next iteration...")
    new_sampling_client = training_client.save_weights_and_get_sampling_client(
        name="rl_iteration_1"
    )
    print("   ✓ New sampling client created")
    print("   (In real RL, this would be used for the next rollout)")

    # Success!
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print(f"\nRL rollout pattern verified:")
    print(f"  • Training client: {training_client.model_id}")
    print(f"  • Sampling client: created")
    print(f"  • Rollouts generated: {len(samples)}")
    print(f"  • Training data: {len(training_data)} datums")
    print(f"  • Loss function: importance_sampling")
    print(f"  • Updated sampling client: ready for iteration 2")
    print(f"\n🎉 RL workflow functional!")
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
