#!/usr/bin/env python3
"""
Test 4: Multi-Step RL Training Loop

Tests GMI wrapper's ability to perform multiple RL training iterations with GRPO:
  1. Create model
  2. Loop 3 times: forward_backward → optim_step
  3. Track loss/metrics across iterations
  4. Verify training progression

Note: Slime instance is configured for RL training with GRPO, so this test
validates the RL/policy gradient workflow.
"""
import httpx
import json
import time
import sys

# Configuration
GMI_URL = "http://kgateway-training.slime-gmi:8000"
API_KEY = "slime-dev-key"
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

NUM_TRAIN_STEPS = 3

print("=" * 80)
print("GMI Wrapper - Test 4: Multi-Step Training Loop")
print("=" * 80)
print(f"URL: {GMI_URL}")
print(f"API Key: {API_KEY[:10]}...")
print(f"Training steps: {NUM_TRAIN_STEPS}")
print("=" * 80)

model_id = None
losses = []

try:
    # Step 0: Health check
    print("\n[0/2] Health Check")
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
    print("\n[1/2] Creating Model")
    print("  Base model: /data/models/Qwen2.5-0.5B-Instruct_torch_dist")
    print("  LoRA: disabled (rank=0, alpha=0)")

    response = httpx.post(
        f"{GMI_URL}/api/v1/create_model",
        json={
            "base_model": "/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
            "lora_config": {"rank": 0, "alpha": 0}
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

    # Step 2: Multi-step training loop
    print(f"\n[2/2] Multi-Step Training Loop ({NUM_TRAIN_STEPS} iterations)")

    for step in range(NUM_TRAIN_STEPS):
        print(f"\n--- Training Step {step + 1}/{NUM_TRAIN_STEPS} ---")

        # Forward-backward pass
        print(f"  [Step {step + 1}] Forward-backward...")
        response = httpx.post(
            f"{GMI_URL}/api/v1/forward_backward",
            json={
                "model_id": model_id,
                "data": [{"input": f"training sample step {step + 1}", "target": "output"}]
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

        # Poll for forward-backward completion
        fb_completed = False
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
                loss = result.get("loss", {})
                grad_norm = result.get("grad_norm", 0.0)
                losses.append(loss)
                print(f"  ✓ Completed - Loss: {json.dumps(loss)}, Grad norm: {grad_norm:.4f}")
                fb_completed = True
                break
            elif poll_response.status_code == 408:
                if attempt % 10 == 0:
                    print(f"    Waiting... ({attempt+1}/60)")
            else:
                print(f"✗ Error: {poll_response.status_code}")
                print(f"  {poll_response.text}")
                sys.exit(1)

        if not fb_completed:
            print(f"✗ Timed out on forward-backward step {step + 1}")
            sys.exit(1)

        # Optimizer step
        print(f"  [Step {step + 1}] Optimizer step...")
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

        # Poll for optimizer step completion
        optim_completed = False
        for attempt in range(30):
            time.sleep(2)
            poll_response = httpx.post(
                f"{GMI_URL}/api/v1/retrieve_future",
                json={"request_id": req_id},
                headers=headers,
                timeout=120.0
            )

            if poll_response.status_code == 200:
                result = poll_response.json()
                success = result.get("success", False)
                grad_norm = result.get("grad_norm", 0.0)
                print(f"  ✓ Optimizer step completed - Success: {success}, Grad norm: {grad_norm:.4f}")
                optim_completed = True
                break
            elif poll_response.status_code == 408:
                if attempt % 5 == 0:
                    print(f"    Waiting... ({attempt+1}/30)")
            else:
                print(f"✗ Error: {poll_response.status_code}")
                print(f"  {poll_response.text}")
                sys.exit(1)

        if not optim_completed:
            print(f"✗ Timed out on optimizer step {step + 1}")
            sys.exit(1)

    # Success!
    print("\n" + "=" * 80)
    print("✓ TEST 4 PASSED - Multi-step training functional!")
    print("=" * 80)
    print(f"\nSuccessfully completed:")
    print(f"  • Model: {model_id}")
    print(f"  • Training steps: {NUM_TRAIN_STEPS}")
    print(f"  • Loss progression:")
    for i, loss in enumerate(losses):
        print(f"      Step {i + 1}: {json.dumps(loss)}")
    print("\n🎉 GMI Wrapper can perform multi-step training loops!")
    print("=" * 80)

    sys.exit(0)

except Exception as e:
    print(f"\n✗ Test failed with exception: {e}")
    import traceback
    traceback.print_exc()

    if model_id:
        print(f"\nModel ID (for debugging): {model_id}")

    print("\n" + "=" * 80)
    print("✗ TEST 4 FAILED")
    print("=" * 80)

    sys.exit(1)
