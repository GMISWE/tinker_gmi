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
