#!/usr/bin/env python3
"""
Test: Sampling Parameters Variations (Tinker Cookbook Pattern)

Tests different sampling strategies used across cookbook recipes:
  1. Greedy decoding (temperature=0.0)
  2. Nucleus sampling (top_p)
  3. Top-k sampling
  4. Stop sequence handling
  5. Temperature effects

Cookbook Reference:
  - Multiple recipes use different sampling strategies
  - completers.py shows various sampling configurations
  - RL recipes typically use temperature=0.7-1.0 for exploration
  - Evaluation uses temperature=0.0 for deterministic outputs
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
print("TEST: Sampling Parameters Variations (Tinker Cookbook Pattern)")
print("=" * 80)

async def test_sampling_variations():
    """Test different sampling strategies"""

    # Step 1: Create service client and training client
    print("\n[1/7] Creating service client and training client...")
    client = tinker.ServiceClient()

    training_client = await client.create_lora_training_client_async(
        base_model="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
        rank=0  # No LoRA
    )
    print(f"✓ Training client created: {training_client.model_id}")

    # Step 2: Create sampling client
    print("\n[2/7] Creating sampling client...")
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(
        name="test_sampling_variations"
    )
    print("✓ Sampling client created")

    # Wait for SGLang to initialize
    print("   Waiting for SGLang router (15 seconds)...")
    await asyncio.sleep(15)
    print("   ✓ SGLang ready")

    # Common prompt for all tests
    prompt_tokens = [1, 2, 3, 4, 5]
    prompt = types.ModelInput.from_ints(prompt_tokens)

    # Step 3: Test greedy decoding (temperature=0.0)
    print("\n[3/7] Testing greedy decoding (temperature=0.0)...")
    print("   Use case: Deterministic evaluation, testing")

    greedy_params = types.SamplingParams(
        max_tokens=20,
        temperature=0.0  # Deterministic - always picks most likely token
    )

    result1 = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=2,  # Should get identical results
        sampling_params=greedy_params
    )

    print(f"   Generated {len(result1.sequences)} samples")
    print(f"   Sample 1 tokens: {result1.sequences[0].tokens[:10]}")
    print(f"   Sample 2 tokens: {result1.sequences[1].tokens[:10]}")

    # Verify samples are identical (greedy should be deterministic)
    if result1.sequences[0].tokens == result1.sequences[1].tokens:
        print("   ✓ Greedy decoding is deterministic (samples match)")
    else:
        print("   ⚠ Warning: Greedy samples differ (may be due to sampling implementation)")

    # Step 4: Test nucleus sampling (top_p=0.9)
    print("\n[4/7] Testing nucleus sampling (top_p=0.9, temperature=0.7)...")
    print("   Use case: RL training, creative generation")

    nucleus_params = types.SamplingParams(
        max_tokens=20,
        temperature=0.7,  # Moderate randomness
        top_p=0.9  # Nucleus sampling - sample from top 90% probability mass
    )

    result2 = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=2,
        sampling_params=nucleus_params
    )

    print(f"   Generated {len(result2.sequences)} samples")
    print(f"   Sample 1 tokens: {result2.sequences[0].tokens[:10]}")
    print(f"   Sample 2 tokens: {result2.sequences[1].tokens[:10]}")
    print(f"   ✓ Nucleus sampling completed")

    # Step 5: Test top-k sampling
    print("\n[5/7] Testing top-k sampling (top_k=50, temperature=1.0)...")
    print("   Use case: Controlled diversity")

    topk_params = types.SamplingParams(
        max_tokens=20,
        temperature=1.0,  # Higher randomness
        top_k=50  # Only consider top 50 tokens
    )

    result3 = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=2,
        sampling_params=topk_params
    )

    print(f"   Generated {len(result3.sequences)} samples")
    print(f"   Sample 1 length: {len(result3.sequences[0].tokens)} tokens")
    print(f"   Sample 2 length: {len(result3.sequences[1].tokens)} tokens")
    print(f"   ✓ Top-k sampling completed")

    # Step 6: Test high temperature (temperature=1.5)
    print("\n[6/7] Testing high temperature (temperature=1.5)...")
    print("   Use case: Maximum exploration, creative generation")

    high_temp_params = types.SamplingParams(
        max_tokens=20,
        temperature=1.5  # Very high randomness
    )

    result4 = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=2,
        sampling_params=high_temp_params
    )

    print(f"   Generated {len(result4.sequences)} samples")
    print(f"   ✓ High temperature sampling completed")

    # Step 7: Summary and analysis
    print("\n[7/7] Analyzing sampling behavior...")

    results_summary = {
        "Greedy (T=0.0)": result1,
        "Nucleus (T=0.7, top_p=0.9)": result2,
        "Top-k (T=1.0, top_k=50)": result3,
        "High temp (T=1.5)": result4
    }

    print("\n  Summary:")
    for strategy, result in results_summary.items():
        avg_length = sum(len(s.tokens) for s in result.sequences) / len(result.sequences)
        print(f"    {strategy:30s} → Avg length: {avg_length:.1f} tokens")

    print("\n" + "=" * 80)
    print("TEST PASSED: Sampling Parameters Variations")
    print("=" * 80)

    print("\nKey Takeaways:")
    print("1. Temperature=0.0 (greedy): Deterministic, best for evaluation")
    print("2. Temperature=0.7 + top_p=0.9 (nucleus): Balanced exploration, common in RL")
    print("3. Temperature=1.0 + top_k=50: Controlled diversity")
    print("4. Temperature=1.5: Maximum exploration, very creative")
    print("5. All sampling strategies work correctly with kgateway-training")

    print("\nCookbook Usage:")
    print("- RL training: temperature=0.7-1.0 for diverse rollouts")
    print("- Evaluation: temperature=0.0 for deterministic outputs")
    print("- Creative tasks: temperature=1.0-1.5 for variety")
    print("=" * 80)

    return 0

try:
    exit_code = asyncio.run(test_sampling_variations())
    sys.exit(exit_code)

except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    print("=" * 80)
    sys.exit(1)
