#!/usr/bin/env python3
"""Test script for the mock Tinker API server.

Before running this script:
1. Start the mock server: python mock_server.py
2. Set environment variables:
   export TINKER_BASE_URL=http://localhost:8000
   export TINKER_API_KEY=test-key
3. Run this script: python test_mock_server.py
"""

import os
import sys

# Ensure environment is configured
if "TINKER_BASE_URL" not in os.environ:
    os.environ["TINKER_BASE_URL"] = "http://localhost:8000"

if "TINKER_API_KEY" not in os.environ:
    os.environ["TINKER_API_KEY"] = "test-key"

import tinker
from tinker import types


def test_service_client():
    """Test basic service client functionality."""
    print("\n" + "="*80)
    print("TEST 1: Service Client - Health Check")
    print("="*80)

    try:
        # Create service client
        client = tinker.ServiceClient()

        # Get server capabilities
        print("Getting server capabilities...")
        capabilities = client.get_server_capabilities()
        print(f"✓ Server capabilities retrieved:")
        print(f"  Supported models: {len(capabilities.supported_models)}")
        for model in capabilities.supported_models:
            print(f"    - {model.model_name}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_flow():
    """Test training client workflow."""
    print("\n" + "="*80)
    print("TEST 2: Training Client - Create Model and Training")
    print("="*80)

    try:
        # Create service client
        client = tinker.ServiceClient()

        # Create a LoRA training client
        print("Creating LoRA training client...")
        training_client = client.create_lora_training_client(
            base_model="Qwen/Qwen2.5-7B",
            rank=32
        )
        print(f"✓ Training client created with model_id: {training_client.model_id}")

        # Prepare some mock training data
        print("\nPreparing training data...")
        training_data = [
            types.Datum(
                model_input=types.ModelInput.from_ints([1, 2, 3]),
                loss_fn_inputs={
                    "target_tokens": types.TensorData(
                        data=[1, 2, 3],
                        shape=[3],
                        dtype="int64"
                    )
                }
            )
        ]
        print(f"✓ Prepared {len(training_data)} training examples")

        # Forward-backward pass
        print("\nPerforming forward-backward pass...")
        fwdbwd_future = training_client.forward_backward(training_data, "cross_entropy")
        print("  Waiting for result...")
        fwdbwd_result = fwdbwd_future.result(timeout=10)
        print(f"✓ Forward-backward completed:")
        print(f"  Total loss: {fwdbwd_result.metrics.get('total_loss', 'N/A')}")
        print(f"  Num tokens: {fwdbwd_result.metrics.get('num_tokens', 'N/A')}")

        # Optimizer step
        print("\nPerforming optimizer step...")
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=1e-4)
        )
        print("  Waiting for result...")
        optim_result = optim_future.result(timeout=10)
        print(f"✓ Optimizer step completed")

        # Save weights
        print("\nSaving weights...")
        save_future = training_client.save_weights_for_sampler("test-checkpoint")
        save_result = save_future.result(timeout=10)
        print(f"✓ Weights saved to: {save_result.path}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sampling_flow():
    """Test sampling client workflow."""
    print("\n" + "="*80)
    print("TEST 3: Sampling Client - Text Generation")
    print("="*80)

    try:
        # Create service client
        client = tinker.ServiceClient()

        # Create sampling client
        print("Creating sampling client...")
        sampling_client = client.create_sampling_client(
            base_model="Qwen/Qwen2.5-7B"
        )
        print("✓ Sampling client created")

        # Prepare prompt
        print("\nPreparing prompt...")
        prompt = types.ModelInput.from_ints([1, 2, 3, 4])  # "Once upon a time" as token IDs
        print("✓ Prompt prepared")

        # Sample
        print("\nGenerating samples...")
        sampling_params = types.SamplingParams(
            max_tokens=20,
            temperature=0.7,
            top_p=0.9
        )

        sample_future = sampling_client.sample(
            prompt=prompt,
            num_samples=2,
            sampling_params=sampling_params
        )
        print("  Waiting for result...")
        result = sample_future.result(timeout=10)

        print(f"✓ Generated {len(result.sequences)} samples:")
        for i, seq in enumerate(result.sequences):
            print(f"  Sample {i+1}: tokens={seq.tokens[:10]}... stop_reason={seq.stop_reason}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rest_client_training_runs():
    """Test REST API for training run metadata."""
    print("\n" + "="*80)
    print("TEST 4: REST Client - Training Runs API")
    print("="*80)

    try:
        client = tinker.ServiceClient()

        # Create a training client
        print("Creating LoRA training client...")
        training_client = client.create_lora_training_client(
            base_model="Qwen/Qwen2.5-7B",
            rank=32
        )
        model_id = training_client.model_id
        print(f"✓ Training client created: {model_id}")

        # Get training run by ID
        print("\nGetting training run by ID...")
        rest_client = client.create_rest_client()
        training_run_future = rest_client.get_training_run(model_id)
        training_run = training_run_future.result(timeout=10)
        print(f"✓ Training run retrieved:")
        print(f"  Base model: {training_run.base_model}")
        print(f"  Is LoRA: {training_run.is_lora}")
        print(f"  LoRA rank: {training_run.lora_rank}")

        # List all training runs
        print("\nListing all training runs...")
        runs_future = rest_client.list_training_runs(limit=10)
        runs_response = runs_future.result(timeout=10)
        print(f"✓ Found {len(runs_response.training_runs)} training runs")
        print(f"  Total count: {runs_response.cursor.total_count}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_resume_training_from_state():
    """Test create_training_client_from_state() - critical for RLHF."""
    print("\n" + "="*80)
    print("TEST 5: Resume Training from State")
    print("="*80)

    try:
        client = tinker.ServiceClient()

        # Create and train original model
        print("Creating initial training client...")
        training_client_1 = client.create_lora_training_client(
            base_model="meta-llama/Llama-3.1-8B",
            rank=64
        )
        print(f"✓ Created: {training_client_1.model_id}")

        # Save state
        print("\nSaving state checkpoint...")
        save_future = training_client_1.save_state("test-state")
        save_result = save_future.result(timeout=10)
        state_path = save_result.path
        print(f"✓ State saved to: {state_path}")

        # Resume from state using create_training_client_from_state
        print("\nResuming training from state...")
        training_client_2 = client.create_training_client_from_state(state_path)
        print(f"✓ Resumed: {training_client_2.model_id}")

        # Verify lora_rank matches (critical check)
        info = training_client_2.get_info()
        if info.model_data.lora_config and info.model_data.lora_config.rank == 64:
            print(f"✓ LoRA rank verified: {info.model_data.lora_config.rank}")
        else:
            raise ValueError("LoRA rank mismatch!")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_operations():
    """Test checkpoint listing, download, and deletion."""
    print("\n" + "="*80)
    print("TEST 6: Checkpoint Operations")
    print("="*80)

    try:
        client = tinker.ServiceClient()

        # Create training client
        print("Creating training client...")
        training_client = client.create_lora_training_client(
            base_model="Qwen/Qwen2.5-7B",
            rank=32
        )
        model_id = training_client.model_id
        print(f"✓ Created: {model_id}")

        # Save multiple checkpoints
        print("\nSaving checkpoints...")
        training_client.save_state("checkpoint-001").result(timeout=10)
        training_client.save_weights_for_sampler("sampler-001").result(timeout=10)
        print("✓ Saved 2 checkpoints")

        # List checkpoints
        print("\nListing checkpoints...")
        rest_client = client.create_rest_client()
        checkpoints_future = rest_client.list_checkpoints(model_id)
        checkpoints_response = checkpoints_future.result(timeout=10)
        print(f"✓ Found {len(checkpoints_response.checkpoints)} checkpoints:")
        for ckpt in checkpoints_response.checkpoints:
            print(f"  - {ckpt.checkpoint_type}: {ckpt.checkpoint_id}")

        # Download checkpoint archive
        if checkpoints_response.checkpoints:
            checkpoint = checkpoints_response.checkpoints[0]
            print(f"\nDownloading checkpoint: {checkpoint.checkpoint_id}")
            archive_future = rest_client.download_checkpoint_archive(
                model_id, checkpoint.checkpoint_id
            )
            archive_data = archive_future.result(timeout=10)
            print(f"✓ Downloaded {len(archive_data)} bytes")

            # Delete checkpoint
            print(f"\nDeleting checkpoint: {checkpoint.checkpoint_id}")
            delete_future = rest_client.delete_checkpoint(model_id, checkpoint.checkpoint_id)
            delete_future.result(timeout=10)
            print("✓ Checkpoint deleted")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MOCK TINKER API SERVER TEST SUITE")
    print("="*80)
    print(f"Base URL: {os.environ.get('TINKER_BASE_URL')}")
    print(f"API Key: {os.environ.get('TINKER_API_KEY')}")
    print("="*80)

    results = []

    # Run tests
    results.append(("Service Client", test_service_client()))
    results.append(("Training Flow", test_training_flow()))
    results.append(("Sampling Flow", test_sampling_flow()))
    results.append(("REST Client - Training Runs", test_rest_client_training_runs()))
    results.append(("Resume Training from State", test_resume_training_from_state()))
    results.append(("Checkpoint Operations", test_checkpoint_operations()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("="*80)
    print(f"Results: {passed}/{total} tests passed")
    print("="*80 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
