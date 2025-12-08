#!/usr/bin/env python3
"""
Test: Sampling Client Creation

Tests Tinker client's sampling client creation workflow:
  1. Create training client
  2. Save weights and create sampling client (async)
  3. Verify sampling client is functional
  4. Generate sample using sampling client

This is a prerequisite for RL workflows where we need to:
- Use training_client for gradient updates
- Use sampling_client for generating rollouts
"""
import os
import sys
import asyncio

# Configure environment for kgateway-training
os.environ["TINKER_BASE_URL"] = "http://kgateway-training.slime-gmi:8000"
os.environ["TINKER_API_KEY"] = "slime-dev-key"

import tinker
from tinker import types

print("=" * 80)
print("TEST: Sampling Client Creation")
print("=" * 80)
print(f"Base URL: {os.environ.get('TINKER_BASE_URL')}")
print(f"API Key: {os.environ.get('TINKER_API_KEY')[:10]}...")
print("=" * 80)

async def run_test():
    """Async test function to properly await async methods"""
    # Step 1: Create service client
    print("\n[1/5] Creating service client...")
    client = tinker.ServiceClient()
    print("   ✓ Service client created")

    # Step 2: Create training client (using async version in async context)
    print("\n[2/5] Creating training client (no LoRA)...")
    training_client = await client.create_lora_training_client_async(
        base_model="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
        rank=0  # No LoRA
    )
    print(f"   ✓ Training client created with model_id: {training_client.model_id}")

    # Step 3: Save weights and create sampling client
    print("\n[3/5] Creating sampling client from training weights...")
    print("   Using: save_weights_and_get_sampling_client_async() [AWAITING]")

    # This is the async version used in cookbook's RL training
    # Properly await the async method
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(
        name="test_sampling_checkpoint"
    )
    print("   ✓ Sampling client created successfully")
    print(f"   Checkpoint name: test_sampling_checkpoint")

    # Wait a moment for SGLang to be ready
    print("   Waiting for SGLang router to initialize (15 seconds)...")
    await asyncio.sleep(15)
    print("   ✓ SGLang should be ready")

    # Step 4: Verify sampling client
    print("\n[4/5] Verifying sampling client...")
    print(f"   Sampling client type: {type(sampling_client)}")
    print(f"   ✓ Sampling client created successfully")

    # Step 5: Generate a sample
    print("\n[5/5] Generating sample with sampling client...")

    # Create a simple prompt
    prompt_tokens = [1, 2, 3, 4, 5]  # Simple token sequence
    prompt = types.ModelInput.from_ints(prompt_tokens)

    sampling_params = types.SamplingParams(
        max_tokens=20,
        temperature=0.7,
        top_p=0.9
    )

    print(f"   Prompt: {prompt_tokens}")
    print(f"   Max tokens: {sampling_params.max_tokens}")
    print(f"   Temperature: {sampling_params.temperature}")

    # Generate samples using async method and await properly
    print("   Calling sample_async() [AWAITING]...")
    result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=2,
        sampling_params=sampling_params
    )

    print(f"   ✓ Generated {len(result.sequences)} samples")

    for i, sample in enumerate(result.sequences):
        print(f"\n   Sample {i + 1}:")
        print(f"     Tokens: {sample.tokens[:10]}{'...' if len(sample.tokens) > 10 else ''}")
        print(f"     Total tokens: {len(sample.tokens)}")
        print(f"     Stop reason: {sample.stop_reason}")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print(f"\nSampling client functionality verified:")
    print(f"  • Training client: {training_client.model_id}")
    print(f"  • Sampling client: created successfully")
    print(f"  • Generated samples: {len(result.sequences)}")
    print(f"\n🎉 Sampling client ready for RL workflows!")
    print("=" * 80)

try:
    asyncio.run(run_test())
    sys.exit(0)

except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

    print("\n" + "=" * 80)
    print("✗ TEST FAILED")
    print("=" * 80)

    sys.exit(1)
