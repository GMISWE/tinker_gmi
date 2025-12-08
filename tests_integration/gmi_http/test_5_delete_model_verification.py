#!/usr/bin/env python3
"""
Test delete_model endpoint - Verifies Ray actor cleanup and metadata persistence

This test validates the core resource management for checkpoint resume:
1. Creates a training client (allocates 4 Ray actors + GPU resources)
2. Calls delete_model to free resources
3. Verifies all Ray actors are killed
4. Confirms training run metadata persists after deletion

Critical for: Checkpoint resume workflow (must free GPU resources between clients)
"""
import requests
import subprocess
import time
import json

BASE_URL = "http://kgateway-training.slime-gmi:8000"
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

print("=" * 80)
print("TEST: delete_model - Ray Actor Cleanup & Metadata Persistence")
print("=" * 80)
print()

# Step 1: Check baseline
print("[1/6] Checking baseline (should be 0 actors)...")
baseline_count = count_megatron_actors()
print(f"  Baseline MegatronTrainRayActor count: {baseline_count}")
if baseline_count != 0:
    print(f"  ⚠ Warning: Expected 0 actors, found {baseline_count}")
print()

# Step 2: Create training client
print("[2/6] Creating training client...")
payload = {
    "model_id": None,  # Auto-generate
    "base_model": "/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
    "lora_config": {"rank": 0},
    "debug_train_only": True
}
resp = requests.post(f"{BASE_URL}/api/v1/create_model", headers=HEADERS, json=payload)
resp.raise_for_status()
request_id = resp.json()["request_id"]
print(f"  Request ID: {request_id}")

# Poll for creation completion
print("  Waiting for model creation...")
model_id = None
for i in range(60):  # 2 min timeout
    time.sleep(2)
    resp = requests.post(f"{BASE_URL}/api/v1/retrieve_future", headers=HEADERS, json={"request_id": request_id})
    if resp.status_code != 200:
        continue

    future_data = resp.json()
    if future_data.get("type") == "try_again":
        print(f"    Creating... ({i+1}/60)")
        continue

    # Completed
    model_id = future_data["model_id"]
    print(f"  ✓ Model created: {model_id}")
    break
else:
    print("  ✗ Timeout waiting for model creation")
    exit(1)

print()

# Step 3: Verify actors were created
print("[3/6] Verifying actors created...")
time.sleep(3)  # Wait for actors to initialize
after_create_count = count_megatron_actors()
print(f"  MegatronTrainRayActor count: {after_create_count}")
if after_create_count == 0:
    print("  ✗ No actors found! Model creation may have failed")
    exit(1)
print(f"  ✓ {after_create_count} actors allocated")
print()

# Step 4: Call delete_model
print(f"[4/6] Calling delete_model for {model_id}...")
delete_payload = {"model_id": model_id}
resp = requests.post(f"{BASE_URL}/api/v1/delete_model", headers=HEADERS, json=delete_payload)
resp.raise_for_status()
delete_result = resp.json()
print(f"  Status: {delete_result['status']}")
print(f"  Message: {delete_result['message']}")
print()

# Step 5: Verify actors were killed
print("[5/6] Verifying actors killed...")
time.sleep(3)  # Wait for cleanup
after_delete_count = count_megatron_actors()
print(f"  MegatronTrainRayActor count: {after_delete_count}")

if after_delete_count == 0:
    print("  ✓ All actors successfully killed!")
else:
    print(f"  ✗ Expected 0 actors, found {after_delete_count}")
    print("  GPU resources may not be freed")
    exit(1)

print()

# Step 6: Verify metadata persists
print(f"[6/6] Verifying metadata persists for {model_id}...")
resp = requests.get(f"{BASE_URL}/api/v1/training_runs/{model_id}", headers=HEADERS)

if resp.status_code == 200:
    metadata = resp.json()
    print("  ✓ Metadata preserved after deletion!")
    print(f"    training_run_id: {metadata['training_run_id']}")
    print(f"    base_model: {metadata['base_model']}")
    print(f"    lora_rank: {metadata['lora_rank']}")
    print(f"    is_lora: {metadata['is_lora']}")
elif resp.status_code == 404:
    print("  ✗ Metadata NOT found (404)")
    print("  This will break checkpoint resume!")
    exit(1)
else:
    print(f"  ✗ Unexpected status: {resp.status_code}")
    exit(1)

print()
print("=" * 80)
print("✓ DELETE MODEL TEST PASSED")
print("=" * 80)
print()
print("Summary:")
print(f"  • Actors before create: {baseline_count}")
print(f"  • Actors after create:  {after_create_count}")
print(f"  • Actors after delete:  {after_delete_count}")
print(f"  • Metadata preserved:   ✓")
print()
print("Result: GPU resources freed, metadata persists for checkpoint resume")
print("=" * 80)
