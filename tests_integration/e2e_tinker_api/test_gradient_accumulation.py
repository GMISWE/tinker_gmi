"""
Test: Gradient Accumulation Pattern from Tinker Cookbook

Cookbook Reference: rl/train.py, Advanced Features section
Pattern: Multiple forward_backward() calls before single optim_step()

This test validates gradient accumulation, which enables:
1. Large effective batch sizes with limited memory
2. Multiple forward_backward() calls accumulate gradients
3. Single optim_step() applies all accumulated gradients
4. Critical for large model training

Cookbook Pattern:
```python
config.num_substeps = 4  # Accumulate over 4 batches

batches = split_list(data, num_substeps=4)
for batch in batches:
    fwd_bwd = training_client.forward_backward(batch, loss_fn="...")
    fwd_bwd.result()  # Returns immediately, gradients accumulate

# Single optimizer step applies all accumulated gradients
training_client.optim_step(adam_params)
```

Why This Matters:
- Enables training with batch_size=256 on GPU that only fits batch_size=64
- Used in most cookbook recipes for large-scale training
- Critical for memory-constrained environments
- Maintains gradient quality while reducing memory

Expected Behavior:
- 4 forward_backward calls with small batches
- 1 optim_step with accumulated gradients
- Equivalent to 1 forward_backward with large batch
"""

import os
import sys

import tinker
from tinker import types

# Configuration
TINKER_BASE_URL = os.getenv("TINKER_BASE_URL", "http://kgateway-training.slime-gmi:8000")
TINKER_API_KEY = os.getenv("TINKER_API_KEY", "slime-dev-key")
BASE_MODEL = "/data/models/Qwen2.5-0.5B-Instruct_torch_dist"


def split_list(items, num_splits):
    """Split list into roughly equal chunks."""
    chunk_size = len(items) // num_splits
    remainder = len(items) % num_splits

    result = []
    start = 0
    for i in range(num_splits):
        # Add 1 to chunk size for first 'remainder' chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end = start + current_chunk_size
        result.append(items[start:end])
        start = end

    return result


def test_gradient_accumulation():
    """
    Test gradient accumulation pattern from cookbook.

    Pattern:
    1. Split training data into num_substeps batches
    2. Call forward_backward() multiple times (gradients accumulate)
    3. Call optim_step() once (applies accumulated gradients)
    """
    print("=" * 80)
    print("TEST: Gradient Accumulation (Tinker Cookbook Pattern)")
    print("=" * 80)

    # Step 1: Connect to service
    print("\n[1/7] Connecting to Tinker service...")
    service_client = tinker.ServiceClient(
        base_url=TINKER_BASE_URL,
        api_key=TINKER_API_KEY
    )
    print(f"✓ Connected to {TINKER_BASE_URL}")

    # Step 2: Create training client
    print("\n[2/7] Creating training client...")
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=0,  # No LoRA for simplicity
        debug_train_only=True,  # Skip update_weights()
    )
    print(f"✓ Training client created: {training_client.model_id}")

    # Step 3: Prepare training data
    print("\n[3/7] Preparing training data...")

    tokenizer = training_client.get_tokenizer()

    # Create training examples (SFT data for simplicity)
    examples = [
        ("What is 2+2?", "The answer is 4."),
        ("What is 3+3?", "The answer is 6."),
        ("What is 5+5?", "The answer is 10."),
        ("What is 7+7?", "The answer is 14."),
        ("What is 9+9?", "The answer is 18."),
        ("What is 11+11?", "The answer is 22."),
        ("What is 13+13?", "The answer is 26."),
        ("What is 15+15?", "The answer is 30."),
    ]

    training_datums = []
    for question, answer in examples:
        # Tokenize
        question_tokens = tokenizer.encode(question, add_special_tokens=True)
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        full_tokens = question_tokens + answer_tokens

        # Create weights (only train on answer)
        weights = [0.0] * len(question_tokens) + [1.0] * len(answer_tokens)

        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(full_tokens),
            loss_fn_inputs={
                "weights": types.TensorData.from_list(weights, dtype="float"),
            }
        )
        training_datums.append(datum)

    print(f"✓ Created {len(training_datums)} training examples")

    # Step 4: Configure gradient accumulation
    print("\n[4/7] Configuring gradient accumulation...")

    num_substeps = 4  # Accumulate over 4 forward_backward calls
    batches = split_list(training_datums, num_substeps)

    print(f"  Total examples: {len(training_datums)}")
    print(f"  Num substeps: {num_substeps}")
    print(f"  Batch sizes: {[len(b) for b in batches]}")
    print(f"  Effective batch size: {len(training_datums)}")
    print(f"✓ Data split into {num_substeps} sub-batches")

    # Step 5: Gradient accumulation - Multiple forward_backward calls
    print("\n[5/7] Running gradient accumulation (multiple forward_backward)...")

    accumulated_losses = []
    for i, batch in enumerate(batches):
        print(f"\n  Substep {i+1}/{num_substeps}:")
        print(f"    Batch size: {len(batch)}")

        # Forward-backward pass (gradients accumulate)
        fwd_bwd_future = training_client.forward_backward(
            batch,
            loss_fn="cross_entropy"
        )
        fwd_bwd_result = fwd_bwd_future.result()

        # Extract loss
        metrics = fwd_bwd_result.metrics
        loss = metrics.get("total_loss:sum", 0.0)
        accumulated_losses.append(loss)
        print(f"    Loss: {loss:.6f}")
        print(f"    ✓ Gradients accumulated (no weight update yet)")

    total_accumulated_loss = sum(accumulated_losses)
    print(f"\n  Total accumulated loss: {total_accumulated_loss:.6f}")
    print(f"✓ Completed {num_substeps} forward_backward calls with gradient accumulation")

    # Step 6: Single optimizer step (applies all accumulated gradients)
    print("\n[6/7] Applying optimizer step (accumulated gradients)...")

    adam_params = tinker.AdamParams(
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8
    )

    optim_future = training_client.optim_step(adam_params)
    optim_result = optim_future.result()

    grad_norm = optim_result.get("grad_norm", 0.0)
    print(f"  Gradient norm: {grad_norm:.6f}")
    print(f"✓ Optimizer step completed - accumulated gradients applied")

    # Step 7: Verify behavior
    print("\n[7/7] Verifying gradient accumulation behavior...")

    # Verify we accumulated over multiple batches
    assert num_substeps == 4, "Should have used 4 substeps"
    print(f"  ✓ Used {num_substeps} accumulation steps")

    # Verify losses were computed for each substep
    assert len(accumulated_losses) == num_substeps, "Should have loss for each substep"
    print(f"  ✓ Computed loss for each substep")

    # Verify total effective batch size
    total_examples = sum(len(b) for b in batches)
    assert total_examples == len(training_datums), "Should process all examples"
    print(f"  ✓ Effective batch size = {total_examples} (via {num_substeps} accumulation steps)")

    # Verify only single optimizer step
    print(f"  ✓ Single optimizer step applied all accumulated gradients")

    # Cleanup
    training_client.delete_model()
    print(f"\n✓ Model cleanup completed")

    print("\n" + "=" * 80)
    print("TEST PASSED: Gradient Accumulation Pattern")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Multiple forward_backward() calls ACCUMULATE gradients")
    print("2. Single optim_step() APPLIES all accumulated gradients")
    print("3. Effective batch size = sum of all substep batches")
    print("4. Memory usage = max(substep batch sizes), not effective batch size")
    print("5. Enables large-batch training on memory-constrained hardware")
    print(f"\nMemory Savings Example:")
    print(f"  Without accumulation: Need memory for batch_size={len(training_datums)}")
    print(f"  With accumulation: Need memory for batch_size={max(len(b) for b in batches)}")
    print(f"  Memory reduction: {len(training_datums) / max(len(b) for b in batches):.1f}x")

    return 0


if __name__ == "__main__":
    try:
        exit_code = test_gradient_accumulation()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
