#!/usr/bin/env python3
"""
Test 3: Checkpoint Save Workflow

Tests GMI wrapper's checkpoint save functionality:
  1. Create model
  2. Save model state/checkpoint
  3. Verify checkpoint creation
"""
import httpx
import json
import time
import sys

# Configuration
GMI_URL = "http://kgateway-training.miles-gmi-tinker:8000"
API_KEY = "slime-dev-key"
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

print("=" * 80)
print("GMI Wrapper - Test 3: Checkpoint Save Workflow")
print("=" * 80)
print(f"URL: {GMI_URL}")
print(f"API Key: {API_KEY[:10]}...")
print("=" * 80)

model_id = None

try:
    # Step 0: Health check
    print("\n[0/3] Health Check")
    try:
        response = httpx.get(f"{GMI_URL}/health", timeout=10.0)
        if response.status_code == 200:
            health = response.json()
            print(f"✓ Status: {health.get('status')}")
            print(f"  Ray initialized: {health.get('ray_initialized')}")
        else:
            print(f"⚠ Health check returned {response.status_code}")
    except Exception as e:
        print(f"⚠ Health check failed: {e}")

    # Step 1: Create model
    print("\n[1/3] Creating Model")
    print("  Base model: /data/models/Qwen2.5-0.5B-Instruct_torch_dist")
    print("  LoRA: disabled (rank=0, alpha=0)")

    response = httpx.post(
        f"{GMI_URL}/api/v1/create_model",
        json={
            "base_model": "/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
            "lora_config": {"rank": 0, "alpha": 0},
            "debug_train_only": True  # Skip update_weights() for supervised-only training
        },
        headers=headers,
        timeout=30.0
    )

    if response.status_code != 200:
        print(f"✗ Create model failed: {response.status_code}")
        print(f"  {response.text}")
        sys.exit(1)

    result = response.json()
    req_id = result["request_id"]
    print(f"✓ Submitted: {req_id}")

    # Poll for model creation completion
    print("  Polling for model creation (max 180 seconds)...")

    for attempt in range(90):
        time.sleep(2)
        poll_response = httpx.post(
            f"{GMI_URL}/api/v1/retrieve_future",
            json={"request_id": req_id},
            headers=headers,
            timeout=120.0
        )

        if poll_response.status_code == 200:
            result = poll_response.json()
            model_id = result.get("model_id")
            print(f"✓ Model created: {model_id}")
            break
        elif poll_response.status_code == 408:
            if attempt % 10 == 0:
                print(f"  Still creating... ({attempt+1}/90)")
        else:
            print(f"✗ Error: {poll_response.status_code}")
            print(f"  {poll_response.text}")
            sys.exit(1)

    if not model_id:
        print("✗ Timed out waiting for model creation (180 seconds)")
        sys.exit(1)

    # Step 2: Save checkpoint
    print("\n[2/3] Saving Checkpoint")
    checkpoint_name = "test_checkpoint_003"
    print(f"  Checkpoint name: {checkpoint_name}")

    response = httpx.post(
        f"{GMI_URL}/api/v1/save_weights",
        json={
            "model_id": model_id,
            "path": checkpoint_name
        },
        headers=headers,
        timeout=60.0
    )

    if response.status_code != 200:
        print(f"✗ Save state failed: {response.status_code}")
        print(f"  {response.text}")
        sys.exit(1)

    result = response.json()
    req_id = result["request_id"]
    print(f"✓ Submitted: {req_id}")

    # Poll for save completion
    print("  Polling for save completion (max 120 seconds)...")
    checkpoint_path = None

    for attempt in range(60):
        time.sleep(2)
        poll_response = httpx.post(
            f"{GMI_URL}/api/v1/retrieve_future",
            json={"request_id": req_id},
            headers=headers,
            timeout=120.0
        )

        if poll_response.status_code == 200:
            result = poll_response.json()
            checkpoint_path = result.get("path")
            print(f"✓ Checkpoint saved:")
            print(f"    Path: {checkpoint_path}")
            break
        elif poll_response.status_code == 408:
            if attempt % 10 == 0:
                print(f"  Still saving... ({attempt+1}/60)")
        else:
            print(f"✗ Error: {poll_response.status_code}")
            print(f"  {poll_response.text}")
            sys.exit(1)

    if not checkpoint_path:
        print("✗ Timed out waiting for checkpoint save (120 seconds)")
        sys.exit(1)

    # Step 3: Verify final state
    print("\n[3/3] Verifying Final State")
    try:
        response = httpx.get(f"{GMI_URL}/health", timeout=10.0)
        if response.status_code == 200:
            health = response.json()
            active_clients = health.get('active_training_clients', 0)
            print(f"✓ Active training clients: {active_clients}")
        else:
            print(f"⚠ Health check returned {response.status_code}")
    except Exception as e:
        print(f"⚠ Final health check failed: {e}")

    # Success!
    print("\n" + "=" * 80)
    print("✓ TEST 3 PASSED - Checkpoint workflow functional!")
    print("=" * 80)
    print(f"\nSuccessfully completed:")
    print(f"  • Model: {model_id}")
    print(f"  • Checkpoint: {checkpoint_name}")
    print(f"  • Path: {checkpoint_path}")
    print("\n🎉 GMI Wrapper can save model checkpoints!")
    print("=" * 80)

    sys.exit(0)

except Exception as e:
    print(f"\n✗ Test failed with exception: {e}")
    import traceback
    traceback.print_exc()

    if model_id:
        print(f"\nModel ID (for debugging): {model_id}")

    print("\n" + "=" * 80)
    print("✗ TEST 3 FAILED")
    print("=" * 80)

    sys.exit(1)
