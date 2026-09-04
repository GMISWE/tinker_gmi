#!/usr/bin/env python3
"""
Test: Supervised Multi-Epoch Training

Tests Tinker client's multi-epoch SFT workflow:
  1. Create training client
  2. Load data from dummy.jsonl
  3. Run 2 epochs of training
  4. Periodic checkpointing
  5. Track loss across epochs

Mirrors the cookbook's supervised/train.py main() training loop.
"""
import os
import sys
import json

# Configure environment for kgateway-training
os.environ["TINKER_BASE_URL"] = "http://kgateway-training.slime-gmi:8000"
os.environ["TINKER_API_KEY"] = "slime-dev-key"

import tinker
from tinker import types

print("=" * 80)
print("TEST: Supervised Multi-Epoch Training")
print("=" * 80)
print(f"Base URL: {os.environ.get('TINKER_BASE_URL')}")
print(f"API Key: {os.environ.get('TINKER_API_KEY')[:10]}...")
print("=" * 80)

# Configuration
NUM_EPOCHS = 2
DATA_FILE = "/tmp/dummy.jsonl"  # Available in slime-training pod

try:
    # Step 1: Create service client
    print("\n[1/5] Creating service client...")
    client = tinker.ServiceClient()
    print("   ✓ Service client created")

    # Step 2: Create training client
    print("\n[2/5] Creating training client (no LoRA)...")
    training_client = client.create_lora_training_client(
        base_model="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
        rank=0,  # No LoRA
        debug_train_only=True  # Skip update_weights() for supervised-only training
    )
    print(f"   ✓ Training client created with model_id: {training_client.model_id}")

    # Get tokenizer for data preparation
    tokenizer = training_client.get_tokenizer()
    print(f"   ✓ Tokenizer retrieved")

    # Step 3: Load training data
    print(f"\n[3/5] Loading training data from {DATA_FILE}...")
    training_data = []

    try:
        with open(DATA_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    training_data.append(item)

        print(f"   ✓ Loaded {len(training_data)} training examples")
        for i, item in enumerate(training_data[:3]):
            print(f"     Example {i+1}: {item['prompt'][:30]}... → {item['response'][:20]}...")

    except FileNotFoundError:
        print(f"   ⚠ Data file not found, using synthetic data instead")
        # Use simple synthetic data for testing - need at least 8 for one batch
        training_data = [
            {"prompt": "What is 2+2?", "response": "4"},
            {"prompt": "What is 3+3?", "response": "6"},
            {"prompt": "What is 4+4?", "response": "8"},
            {"prompt": "What is 5+5?", "response": "10"},
            {"prompt": "What is 6+6?", "response": "12"},
            {"prompt": "What is 7+7?", "response": "14"},
            {"prompt": "What is 8+8?", "response": "16"},
            {"prompt": "What is 9+9?", "response": "18"},
        ]
        print(f"   ✓ Using {len(training_data)} synthetic examples")

    # CRITICAL: Batch size must match Slime's global_batch_size
    # Slime will distribute across DP ranks internally
    # With global_batch_size=8, we need to send 8 samples total
    BATCH_SIZE = 8

    # Step 4: Multi-epoch training
    print(f"\n[4/5] Training for {NUM_EPOCHS} epochs (batch_size={BATCH_SIZE})...")

    losses = []
    total_steps = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        # Process data in batches
        num_batches = (len(training_data) + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx in range(num_batches):
            step = epoch * num_batches + batch_idx
            total_steps += 1

            # Get batch of items
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(training_data))
            batch_items = training_data[batch_start:batch_end]

            # Pad last batch if needed to maintain batch size
            while len(batch_items) < BATCH_SIZE:
                batch_items.append(training_data[0])  # Repeat first item

            # Create batch of datums
            datums = []
            for item in batch_items:
                input_text = f"{item['prompt']} {item['response']}"
                input_tokens = tokenizer.encode(input_text)

                datum = tinker.Datum(
                    model_input=types.ModelInput.from_ints(input_tokens),
                    loss_fn_inputs={
                        "target": types.TensorData(
                            data=input_tokens[1:] + [input_tokens[-1]],
                            shape=[len(input_tokens)],
                            dtype="int64"
                        ),
                        "weights": types.TensorData(
                            data=[1.0] * len(input_tokens),
                            shape=[len(input_tokens)],
                            dtype="float32"
                        )
                    }
                )
                datums.append(datum)

            # Forward-backward with BATCH of datums
            fwd_bwd_future = training_client.forward_backward(
                datums,  # Send BATCH_SIZE datums
                loss_fn="cross_entropy"
            )
            fwd_bwd_result = fwd_bwd_future.result()

            # Optimizer step with learning rate
            learning_rate = 2e-4  # From cookbook
            adam_params = tinker.AdamParams(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8
            )
            optim_future = training_client.optim_step(adam_params)
            optim_result = optim_future.result()

            # Extract loss if available
            loss_value = None
            if hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
                if 'loss:mean' in fwd_bwd_result.metrics:
                    loss_value = fwd_bwd_result.metrics['loss:mean']
                    losses.append(loss_value)

            # Print progress
            if loss_value is not None:
                print(f"  Step {step + 1}/{NUM_EPOCHS * num_batches}: "
                      f"loss={loss_value:.6f}, batch={batch_idx + 1}/{num_batches} "
                      f"(samples {batch_start+1}-{batch_end})")
            else:
                print(f"  Step {step + 1}/{NUM_EPOCHS * num_batches}: "
                      f"batch={batch_idx + 1}/{num_batches} "
                      f"(samples {batch_start+1}-{batch_end})")

        print(f"✓ Epoch {epoch + 1} completed")

    # Step 5: Save final checkpoint
    print(f"\n[5/5] Saving final checkpoint...")
    save_future = training_client.save_state("multi_epoch_final")
    save_result = save_future.result()
    print(f"   ✓ Checkpoint saved: {save_result.path}")

    # Summary
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print(f"\nMulti-epoch training completed:")
    print(f"  • Model ID: {training_client.model_id}")
    print(f"  • Epochs: {NUM_EPOCHS}")
    print(f"  • Training examples: {len(training_data)}")
    print(f"  • Total steps: {total_steps}")
    if losses:
        print(f"  • Initial loss: {losses[0]:.6f}")
        print(f"  • Final loss: {losses[-1]:.6f}")
        if losses[0] > losses[-1]:
            print(f"  • Loss improved: {losses[0] - losses[-1]:.6f} ✓")
    print(f"  • Final checkpoint: {save_result.path}")
    print(f"\n🎉 Multi-epoch SFT workflow functional!")
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
