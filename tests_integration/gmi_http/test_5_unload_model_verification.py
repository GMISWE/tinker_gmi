#!/usr/bin/env python3
"""
Test unload_model endpoint - Verifies Ray actor cleanup and metadata persistence

This test validates the core resource management for checkpoint resume:
1. Creates a training client (allocates 4 Ray actors + GPU resources)
2. Calls unload_model (Tinker's native endpoint) to free resources
3. Verifies all Ray actors are killed
4. Confirms training run metadata persists after unload

Critical for: Checkpoint resume workflow (must free GPU resources between clients)
"""
import requests
import subprocess
import time
import json

BASE_URL = "http://kgateway-training.miles-gmi-tinker:8000"
API_KEY = "slime-dev-key"
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def count_megatron_actors():
    """Count MegatronTrainRayActor actors using ray list"""
    try:
        result = subprocess.run(
            ["ray", "list", "actors", "--filter", "state=ALIVE"],
            capture_output=True,
            text=True,
            check=True
        )
        # Count lines with MegatronTrainRayActor
        count = result.stdout.count("MegatronTrainRayActor")
        return count
    except subprocess.CalledProcessError:
        # No actors
        return 0


def poll_future(request_id: str, timeout: float = 30.0) -> dict:
    """Poll retrieve_future until completion or timeout.

    retrieve_future returns:
    - 408 (Request Timeout) if operation is still pending
    - 200 with result if completed successfully
    - 500 if failed
    """
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            json={"request_id": request_id},
            headers=HEADERS,
            timeout=10
        )
        if resp.status_code == 200:
            # Success - operation completed
            return resp.json()
        elif resp.status_code == 500:
            # Failed - operation errored
            error_detail = resp.json().get("detail", "Unknown error")
            raise Exception(f"Operation failed: {error_detail}")
        elif resp.status_code == 408:
            # Still pending - continue polling
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Timeout waiting for request {request_id}")


print("=" * 80)
print("TEST: unload_model - Ray Actor Cleanup & Metadata Persistence")
print("=" * 80)
print()

# Step 1: Check baseline
print("[1/6] Checking baseline (should be 0 actors)...")
baseline_count = count_megatron_actors()
print(f"  Baseline MegatronTrainRayActor count: {baseline_count}")
if baseline_count != 0:
    print(f"  Warning: Expected 0 actors, found {baseline_count}")
print()

# Step 2: Create training client
print("[2/6] Creating training client...")
payload = {
    "session_id": "test-session-unload",
    "model_seq_id": 0,
    "base_model": "/data/models/Qwen2.5-0.5B-Instruct",
    "lora_config": {"rank": 16},
    "debug_train_only": True
}
resp = requests.post(f"{BASE_URL}/api/v1/create_model", headers=HEADERS, json=payload)
resp.raise_for_status()
create_result = resp.json()
request_id = create_result["request_id"]
model_id = create_result.get("model_id")
print(f"  Request ID: {request_id}")
print(f"  Model ID: {model_id}")

# Poll for creation completion
print("  Waiting for model creation...")
for i in range(60):  # 2 min timeout
    time.sleep(2)
    resp = requests.post(f"{BASE_URL}/api/v1/retrieve_future", headers=HEADERS, json={"request_id": request_id})
    if resp.status_code != 200:
        continue

    future_data = resp.json()
    if future_data.get("status") == "pending":
        print(f"    Creating... ({i+1}/60)")
        continue

    if future_data.get("status") == "completed":
        # Completed - model_id might be in the result
        if not model_id:
            model_id = future_data.get("result", {}).get("model_id") or future_data.get("model_id")
        print(f"  Model created: {model_id}")
        break
else:
    print("  Timeout waiting for model creation")
    exit(1)

print()

# Step 3: Verify actors were created
print("[3/6] Verifying actors created...")
time.sleep(3)  # Wait for actors to initialize
after_create_count = count_megatron_actors()
print(f"  MegatronTrainRayActor count: {after_create_count}")
if after_create_count == 0:
    print("  No actors found! Model creation may have failed")
    exit(1)
print(f"  {after_create_count} actors allocated")
print()

# Step 4: Call unload_model (Tinker's native endpoint)
print(f"[4/6] Calling unload_model for {model_id}...")
unload_payload = {"model_id": model_id, "type": "unload_model"}
resp = requests.post(f"{BASE_URL}/api/v1/unload_model", headers=HEADERS, json=unload_payload)
resp.raise_for_status()
unload_result = resp.json()
unload_request_id = unload_result.get("request_id")
print(f"  Request ID: {unload_request_id}")

# Poll for unload completion
if unload_request_id:
    print("  Waiting for unload to complete...")
    final_result = poll_future(unload_request_id)
    print(f"  Unload completed: {final_result.get('result', {})}")
print()

# Step 5: Verify actors were killed
print("[5/6] Verifying actors killed...")
time.sleep(3)  # Wait for cleanup
after_unload_count = count_megatron_actors()
print(f"  MegatronTrainRayActor count: {after_unload_count}")

if after_unload_count == 0:
    print("  All actors successfully killed!")
else:
    print(f"  Expected 0 actors, found {after_unload_count}")
    print("  GPU resources may not be freed")
    exit(1)

print()

# Step 6: Verify metadata persists
print(f"[6/6] Verifying metadata persists for {model_id}...")
resp = requests.get(f"{BASE_URL}/api/v1/training_runs/{model_id}", headers=HEADERS)

if resp.status_code == 200:
    metadata = resp.json()
    print("  Metadata preserved after unload!")
    print(f"    training_run_id: {metadata['training_run_id']}")
    print(f"    base_model: {metadata['base_model']}")
    print(f"    lora_rank: {metadata['lora_rank']}")
    print(f"    is_lora: {metadata['is_lora']}")
elif resp.status_code == 404:
    print("  Metadata NOT found (404)")
    print("  This will break checkpoint resume!")
    exit(1)
else:
    print(f"  Unexpected status: {resp.status_code}")
    exit(1)

print()
print("=" * 80)
print("UNLOAD MODEL TEST PASSED")
print("=" * 80)
print()
print("Summary:")
print(f"  Actors before create: {baseline_count}")
print(f"  Actors after create:  {after_create_count}")
print(f"  Actors after unload:  {after_unload_count}")
print(f"  Metadata preserved:   Yes")
print()
print("Result: GPU resources freed, metadata persists for checkpoint resume")
print("=" * 80)
