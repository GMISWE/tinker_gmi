#!/usr/bin/env python3
"""
Test: Evaluator Integration (Tinker Cookbook Pattern)

Tests periodic evaluation during training from eval/evaluators.py:
  1. Define evaluation callback (like NLLEvaluator)
  2. Run training loop
  3. Periodically call evaluator every N steps
  4. Log evaluation metrics alongside training metrics

Cookbook Reference:
  - eval/evaluators.py: Evaluator interfaces
  - supervised/train.py: Uses evaluators in training loop
  - Pattern: if step % eval_every == 0: metrics = await evaluator(sampling_client)

Key Pattern:
  for step in range(n_steps):
      # Training
      fwd_bwd = training.forward_backward(batch, ...)
      optim = training.optim_step(adam)

      # Periodic evaluation
      if step % eval_every == 0:
          eval_metrics = evaluator(sampling_client)
          logger.log(eval_metrics, step=step)
"""
import os
import sys

# Configure environment for kgateway-training
os.environ["TINKER_BASE_URL"] = "http://kgateway-training.slime-gmi:8000"
os.environ["TINKER_API_KEY"] = "slime-dev-key"

import tinker
from tinker import types

print("=" * 80)
print("TEST: Evaluator Integration (Tinker Cookbook Pattern)")
print("=" * 80)

def test_evaluator_integration():
    """Test periodic evaluation during training"""

    # Step 1: Create training client
    print("\n[1/5] Creating training client...")
    client = tinker.ServiceClient()

    training_client = client.create_lora_training_client(
        base_model="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
        rank=0,
        debug_train_only=True  # SFT only
    )
    print(f"✓ Training client created: {training_client.model_id}")

    # Step 2: Define simple evaluator (cookbook pattern)
    print("\n[2/5] Defining evaluation function...")

    def simple_evaluator():
        """
        Simple evaluator that computes a mock metric.
        In real cookbook: NLLEvaluator computes test set NLL
        """
        # Mock evaluation metric (in real use: compute NLL on test set)
        test_loss = 0.123  # Placeholder
        return {
            "eval/test_loss": test_loss,
            "eval/status": "completed"
        }

    eval_every = 3  # Evaluate every 3 steps
    print(f"   Evaluator defined: simple_evaluator()")
    print(f"   Eval frequency: every {eval_every} steps")

    # Step 3: Training loop with periodic evaluation
    print("\n[3/5] Running training loop with periodic evaluation...")

    num_steps = 8
    batch_size = 2
    base_lr = 1e-4

    training_metrics = []
    evaluation_metrics = []

    for step in range(num_steps):
        # Prepare training batch
        batch = []
        for _ in range(batch_size):
            input_tokens = [1, 2, 3, 4, 5]
            target_tokens = [2, 3, 4, 5, 6]
            weights = [1.0] * len(target_tokens)

            datum = tinker.Datum(
                model_input=types.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    "target": types.TensorData(
                        data=target_tokens,
                        shape=[len(target_tokens)],
                        dtype="int64"
                    ),
                    "weights": types.TensorData(
                        data=weights,
                        shape=[len(weights)],
                        dtype="float32"
                    )
                }
            )
            batch.append(datum)

        # Forward-backward
        fwd_bwd_future = training_client.forward_backward(
            batch,
            loss_fn="cross_entropy"
        )
        fwd_bwd_result = fwd_bwd_future.result()

        # Extract training metrics
        train_loss = fwd_bwd_result.metrics.get("loss:mean", 0.0)

        # Optimizer step
        adam_params = tinker.AdamParams(
            learning_rate=base_lr,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8
        )
        optim_future = training_client.optim_step(adam_params)
        optim_result = optim_future.result()

        # Log training metrics
        training_metrics.append({
            "step": step,
            "train/loss": train_loss
        })

        print(f"   Step {step}/{num_steps}: train_loss={train_loss:.6f}", end="")

        # Periodic evaluation (cookbook pattern)
        if step % eval_every == 0:
            eval_result = simple_evaluator()
            evaluation_metrics.append({
                "step": step,
                **eval_result
            })
            print(f" | eval_loss={eval_result['eval/test_loss']:.6f} ✓")
        else:
            print()

    print(f"\n   ✓ Training completed: {num_steps} steps")
    print(f"   ✓ Evaluations run: {len(evaluation_metrics)} times")

    # Step 4: Verify evaluation pattern
    print("\n[4/5] Verifying evaluation pattern...")

    expected_eval_steps = [s for s in range(num_steps) if s % eval_every == 0]
    actual_eval_steps = [m["step"] for m in evaluation_metrics]

    print(f"   Expected eval steps: {expected_eval_steps}")
    print(f"   Actual eval steps:   {actual_eval_steps}")

    assert actual_eval_steps == expected_eval_steps, "Eval steps mismatch"
    print("   ✓ Evaluation triggered at correct steps")

    # Step 5: Display metrics timeline
    print("\n[5/5] Metrics timeline...")

    for step in range(num_steps):
        train_metric = next(m for m in training_metrics if m["step"] == step)
        eval_metric = next((m for m in evaluation_metrics if m["step"] == step), None)

        if eval_metric:
            print(f"   Step {step}: train_loss={train_metric['train/loss']:.6f}, "
                  f"eval_loss={eval_metric['eval/test_loss']:.6f} [EVAL]")
        else:
            print(f"   Step {step}: train_loss={train_metric['train/loss']:.6f}")

    print("\n" + "=" * 80)
    print("TEST PASSED: Evaluator Integration")
    print("=" * 80)

    print("\nKey Takeaways:")
    print("1. Evaluator is a callback function called periodically during training")
    print("2. Pattern: if step % eval_every == 0: eval_metrics = evaluator(...)")
    print("3. Evaluation metrics logged alongside training metrics")
    print("4. Used in ALL cookbook recipes (SFT, RL, preference learning)")
    print("5. Common evaluators: NLLEvaluator, InspectAI benchmarks")

    print("\nCookbook Usage:")
    print("- eval/evaluators.py: Evaluator interfaces (SamplingClientEvaluator, etc.)")
    print("- supervised/train.py: Uses NLLEvaluator for test set loss")
    print("- rl/train.py: Uses evaluators for periodic benchmarking")
    print("- eval/inspect_evaluators.py: Integration with InspectAI benchmarks")
    print("=" * 80)

    return 0

try:
    exit_code = test_evaluator_integration()
    sys.exit(exit_code)

except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    print("=" * 80)
    sys.exit(1)
