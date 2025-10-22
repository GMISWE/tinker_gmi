#!/usr/bin/env python3
"""Test Tinker client against GMI wrapper.

This test verifies that the GMI wrapper correctly implements the Tinker API.
"""
import os
import sys

# Configure environment for GMI wrapper
os.environ["TINKER_BASE_URL"] = "http://gmi-wrapper.slime-gmi:8000"
os.environ["TINKER_API_KEY"] = "slime-dev-key"

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
    print("TEST 2: Training Client - Create Model and Single Training Step")
    print("="*80)

    try:
        # Create service client
        client = tinker.ServiceClient()

        # Create a training client (no LoRA)
        print("Creating training client...")
        training_client = client.create_lora_training_client(
            base_model="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
            rank=0  # No LoRA
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
        fwdbwd_result = fwdbwd_future.result(timeout=120)
        print(f"✓ Forward-backward completed:")
        print(f"  Total loss: {fwdbwd_result.metrics.get('total_loss:sum', 'N/A')}")
        print(f"  PG loss: {fwdbwd_result.metrics.get('pg_loss:sum', 'N/A')}")
        print(f"  Grad norm: {fwdbwd_result.metrics.get('grad_norm:mean', 'N/A')}")
        print(f"  Num tokens: {fwdbwd_result.metrics.get('num_tokens:sum', 'N/A')}")

        # Optimizer step
        print("\nPerforming optimizer step...")
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=1e-4)
        )
        print("  Waiting for result...")
        optim_result = optim_future.result(timeout=60)
        print(f"✓ Optimizer step completed")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("TINKER CLIENT TEST - GMI WRAPPER")
    print("="*80)
    print(f"Base URL: {os.environ.get('TINKER_BASE_URL')}")
    print(f"API Key: {os.environ.get('TINKER_API_KEY')[:10]}...")
    print("="*80)

    results = []

    # Run tests
    results.append(("Service Client", test_service_client()))
    results.append(("Training Flow", test_training_flow()))

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
