#!/usr/bin/env python3
"""
Test 2: Model Creation + Single Train Step

Based on test_complete_workflow.py
Tests GMI wrapper's complete training workflow:
  1. Create model (initialize Slime actors)
  2. Forward-backward pass (compute gradients)
  3. Optimizer step (update weights)
"""
import httpx
import json
import time
import sys

# Configuration (updated for kgateway-training)
GMI_URL = "http://kgateway-training.miles-gmi-tinker:8000"
API_KEY = "slime-dev-key"
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

print("=" * 80)
print("GMI Wrapper - Test 2: Model Creation + Single Train Step")
print("=" * 80)
print(f"URL: {GMI_URL}")
print(f"API Key: {API_KEY[:10]}...")
print("=" * 80)

model_id = None

try:
    # Step 0: Health check (optional)
    print("\n[0/4] Health Check")
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
    print("\n[1/4] Creating Model")
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

    # Poll for model creation completion (max 180 seconds)
    print("  Polling for model creation (max 180 seconds)...")

    for attempt in range(90):
        time.sleep(2)
        poll_response = httpx.post(
            f"{GMI_URL}/api/v1/retrieve_future",
            json={"request_id": req_id},
            headers=headers,
            timeout=120.0  # Increased for slow operations
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

    # Step 2: Forward-backward pass
    print("\n[2/4] Forward-Backward Pass")
    print("  Computing gradients...")

    # Prepare data in proper Tinker API format
    # Simple test tokens: [1, 2, 3, 4, 5]
    # NOTE: Must send at least DP=8 samples to match data parallel configuration
    input_tokens = [1, 2, 3, 4, 5]
    target_tokens = [2, 3, 4, 5, 6]  # Shifted for next-token prediction

    datum = {
        "model_input": {
            "tokens": input_tokens
        },
        "loss_fn_inputs": {
            "target": {
                "data": target_tokens,
                "shape": [len(target_tokens)],
                "dtype": "int64"
            },
            "weights": {
                "data": [1.0] * len(target_tokens),
                "shape": [len(target_tokens)],
                "dtype": "float32"
            }
        }
    }

    # Create 8 samples (matching DP=8 configuration)
    data = [datum for _ in range(8)]

    response = httpx.post(
        f"{GMI_URL}/api/v1/forward_backward",
        json={
            "model_id": model_id,
            "data": data,
            "loss_fn": "cross_entropy"
        },
        headers=headers,
        timeout=120.0
    )

    if response.status_code != 200:
        print(f"✗ Forward-backward failed: {response.status_code}")
        print(f"  {response.text}")
        sys.exit(1)

    result = response.json()
    req_id = result["request_id"]
    print(f"✓ Submitted: {req_id}")

    # Poll for forward-backward completion (max 120 seconds)
    print(f"  Polling for completion (max 120 seconds)...")
    fb_completed = False

    for attempt in range(60):
        time.sleep(2)
        poll_response = httpx.post(
            f"{GMI_URL}/api/v1/retrieve_future",
            json={"request_id": req_id},
            headers=headers,
            timeout=120.0  # Increased for slow operations
        )

        if poll_response.status_code == 200:
            result = poll_response.json()
            loss = result.get("loss", {})
            grad_norm = result.get("grad_norm", 0.0)
            valid_step = result.get("valid_step", False)
            print(f"✓ Completed:")
            print(f"    Loss: {json.dumps(loss)}")
            print(f"    Grad norm: {grad_norm}")
            print(f"    Valid step: {valid_step}")
            fb_completed = True
            break
        elif poll_response.status_code == 408:
            if attempt % 10 == 0:
                print(f"  Still pending... ({attempt+1}/60)")
        else:
            print(f"✗ Error: {poll_response.status_code}")
            print(f"  {poll_response.text}")
            sys.exit(1)

    if not fb_completed:
        print(f"✗ Timed out waiting for forward-backward (120 seconds)")
        sys.exit(1)

    # Step 3: Optimizer step
    print("\n[3/4] Optimizer Step")
    print("  Applying gradients to update weights...")

    response = httpx.post(
        f"{GMI_URL}/api/v1/optim_step",
        json={"model_id": model_id},
        headers=headers,
        timeout=60.0
    )

    if response.status_code != 200:
        print(f"✗ Optimizer step failed: {response.status_code}")
        print(f"  {response.text}")
        sys.exit(1)

    result = response.json()
    req_id = result["request_id"]
    print(f"✓ Submitted: {req_id}")

    # Poll for optimizer step completion (max 60 seconds)
    print(f"  Polling for completion (max 60 seconds)...")
    optim_completed = False

    for attempt in range(30):
        time.sleep(2)
        poll_response = httpx.post(
            f"{GMI_URL}/api/v1/retrieve_future",
            json={"request_id": req_id},
            headers=headers,
            timeout=120.0  # Increased for slow operations
        )

        if poll_response.status_code == 200:
            result = poll_response.json()
            success = result.get("success", False)
            grad_norm = result.get("grad_norm", 0.0)
            print(f"✓ Completed:")
            print(f"    Success: {success}")
            print(f"    Grad norm: {grad_norm}")
            optim_completed = True
            break
        elif poll_response.status_code == 408:
            if attempt % 5 == 0:
                print(f"  Still pending... ({attempt+1}/30)")
        else:
            print(f"✗ Error: {poll_response.status_code}")
            print(f"  {poll_response.text}")
            sys.exit(1)

    if not optim_completed:
        print(f"✗ Timed out waiting for optimizer step (60 seconds)")
        sys.exit(1)

    # Step 4: Verify final state
    print("\n[4/4] Verifying Final State")
    try:
        response = httpx.get(f"{GMI_URL}/health", timeout=10.0)
        if response.status_code == 200:
            health = response.json()
            active_clients = health.get('active_training_clients', 0)
            total_futures = health.get('futures_count', 0)
            print(f"✓ Active training clients: {active_clients}")
            print(f"  Total futures: {total_futures}")
        else:
            print(f"⚠ Health check returned {response.status_code}")
    except Exception as e:
        print(f"⚠ Final health check failed: {e}")

    # Success!
    print("\n" + "=" * 80)
    print("✓ TEST 2 PASSED - Training workflow functional!")
    print("=" * 80)
    print(f"\nSuccessfully completed:")
    print(f"  • Model: {model_id}")
    print(f"  • 1x forward_backward (gradient computation)")
    print(f"  • 1x optim_step (weight update)")
    print("\n🎉 GMI Wrapper can perform single training steps!")
    print("=" * 80)

    sys.exit(0)

except Exception as e:
    print(f"\n✗ Test failed with exception: {e}")
    import traceback
    traceback.print_exc()

    if model_id:
        print(f"\nModel ID (for debugging): {model_id}")

    print("\n" + "=" * 80)
    print("✗ TEST 2 FAILED")
    print("=" * 80)

    sys.exit(1)
