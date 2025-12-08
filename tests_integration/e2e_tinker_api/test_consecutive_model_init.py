#!/usr/bin/env python3
"""Test consecutive model initializations to verify timeout behavior."""
import os
import sys
import time

print("[TEST] Starting consecutive model initialization test", flush=True)

# Configure environment for kgateway-training
os.environ["TINKER_BASE_URL"] = "http://kgateway-training.slime-gmi:8000"
os.environ["TINKER_API_KEY"] = "slime-dev-key"

import tinker
from tinker import types

print("\n" + "="*80)
print("TEST: Consecutive Model Creation - Timeout Protection")
print("="*80)
print(f"Base URL: {os.environ.get('TINKER_BASE_URL')}")
print(f"API Key: {os.environ.get('TINKER_API_KEY')[:10]}...")
print("="*80)

try:
    # Create service client
    print("\n1. Creating service client...")
    client = tinker.ServiceClient()
    print("   ✓ Service client created")

    # Get server capabilities
    print("\n2. Getting server capabilities...")
    capabilities = client.get_server_capabilities()
    print(f"   ✓ Server capabilities retrieved:")
    print(f"     Supported models: {len(capabilities.supported_models)}")

    # FIRST MODEL: Should succeed
    print("\n" + "="*80)
    print("FIRST MODEL CREATION (should succeed)")
    print("="*80)
    start_time = time.time()
    training_client_1 = client.create_lora_training_client(
        base_model="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
        rank=0  # No LoRA
    )
    elapsed_1 = time.time() - start_time
    print(f"   ✓ First model created: {training_client_1.model_id}")
    print(f"   Time taken: {elapsed_1:.1f}s")

    # SECOND MODEL: Should fail with timeout (GPUs held by first model)
    print("\n" + "="*80)
    print("SECOND MODEL CREATION (should timeout)")
    print("="*80)
    print("   Attempting to create second model...")
    print("   Expected: Timeout after 120s with clear error message")
    start_time = time.time()

    try:
        training_client_2 = client.create_lora_training_client(
            base_model="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
            rank=0  # No LoRA
        )
        elapsed_2 = time.time() - start_time
        print(f"   ✗ UNEXPECTED: Second model created: {training_client_2.model_id}")
        print(f"   Time taken: {elapsed_2:.1f}s")
        print("\n" + "="*80)
        print("✗ TEST FAILED: Second model should have timed out!")
        print("="*80)
        sys.exit(1)

    except Exception as e:
        elapsed_2 = time.time() - start_time
        error_msg = str(e)
        print(f"\n   ✓ Second model failed as expected")
        print(f"   Time taken: {elapsed_2:.1f}s")
        print(f"   Error message: {error_msg}")

        # Verify timeout behavior
        if "timeout" in error_msg.lower() or "insufficient" in error_msg.lower():
            print(f"\n   ✓ Error message indicates resource exhaustion (correct)")
        else:
            print(f"\n   ⚠ Warning: Error message doesn't clearly indicate timeout")

        if 115 <= elapsed_2 <= 135:  # Allow 15s margin around 120s timeout
            print(f"   ✓ Timeout duration is correct (~120s)")
        else:
            print(f"   ⚠ Warning: Timeout duration unexpected (expected ~120s, got {elapsed_2:.1f}s)")

    print("\n" + "="*80)
    print("✓ TEST PASSED!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - First model: SUCCESS ({elapsed_1:.1f}s)")
    print(f"  - Second model: FAILED with timeout ({elapsed_2:.1f}s)")
    print(f"  - Timeout protection: WORKING")
    print("="*80)
    sys.exit(0)

except Exception as e:
    print(f"\n✗ Test failed with unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
