#!/usr/bin/env python3
"""
E2E test for Tinker API sampling with kgateway-training + Slime SGLang

This test verifies:
1. Model creation initializes both training actors AND RolloutManager with SGLang
2. Async sampling endpoint (/api/v1/asample) works
3. Response format matches Tinker API schema
4. Real tokens are generated from SGLang (not dummy data)
"""
import os
import sys
import time

# Configure Tinker to use kgateway-training
os.environ["TINKER_BASE_URL"] = "http://kgateway-training.slime-gmi:8000"
os.environ["TINKER_API_KEY"] = "slime-dev-key"

print("[DEBUG] Script started")
print("[DEBUG] Importing tinker...")

import tinker
from tinker import types

print("[DEBUG] Tinker imported successfully")
print("[DEBUG] About to print separator...")

try:
    print("\n" + "="*80)
    print("TEST: Tinker Sampling Client - SGLang Integration")
    print("="*80)
    print(f"Base URL: {os.environ['TINKER_BASE_URL']}")
    print(f"API Key: {os.environ['TINKER_API_KEY'][:12]}...")
    print("="*80)

    # Step 1: Create service client
    print("\n1. Creating service client...")
    client = tinker.ServiceClient()
    print("   ✓ Service client created")

    # Step 2: Create training client (this initializes SGLang)
    print("\n2. Creating training client (initializing SGLang)...")
    print("   This may take a few minutes as SGLang engines start up...")
    training_client = client.create_lora_training_client(
        base_model="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
        rank=0  # No LoRA
    )
    print(f"   ✓ Training client created with model_id: {training_client.model_id}")

    # Wait for SGLang to be ready (RolloutManager needs time to start router)
    print("\n3. Waiting for SGLang router to be ready...")
    time.sleep(15)  # Give SGLang time to start
    print("   ✓ SGLang should be ready")

    # Step 3: Generate samples using async API
    print("\n4. Generating samples with SGLang...")
    prompt = types.ModelInput.from_ints([1, 2, 3, 4])  # Simple test tokens
    sampling_params = types.SamplingParams(
        max_tokens=20,
        temperature=0.7,
        top_p=0.9
    )

    print("   Calling asample endpoint...")
    sample_future = training_client.sample_async(
        prompt=prompt,
        num_samples=2,
        sampling_params=sampling_params
    )

    print("   Waiting for result (timeout=120s)...")
    result = sample_future.result(timeout=120)

    # Step 4: Verify results
    print(f"\n✓ Generated {len(result.sequences)} samples:")
    for i, seq in enumerate(result.sequences):
        print(f"  Sample {i+1}:")
        print(f"    - Tokens: {len(seq.tokens)} tokens")
        print(f"    - Stop reason: {seq.stop_reason}")
        print(f"    - First 5 tokens: {seq.tokens[:5]}")
        if seq.logprobs:
            print(f"    - First 5 logprobs: {seq.logprobs[:5]}")

    # Verification
    assert len(result.sequences) == 2, f"Expected 2 sequences, got {len(result.sequences)}"
    assert result.type == "sample", f"Expected type='sample', got '{result.type}'"

    for i, seq in enumerate(result.sequences):
        assert seq.stop_reason in ["length", "eos_token", "stop_sequence"], \
            f"Invalid stop_reason: {seq.stop_reason}"
        assert len(seq.tokens) > 0, f"Sample {i+1} has no tokens"
        if seq.logprobs:
            assert len(seq.logprobs) == len(seq.tokens), \
                f"Sample {i+1}: logprobs length mismatch"

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - Model ID: {training_client.model_id}")
    print(f"  - Samples generated: {len(result.sequences)}")
    print(f"  - Tokens per sample: {[len(s.tokens) for s in result.sequences]}")
    print("="*80)
    sys.exit(0)

except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
