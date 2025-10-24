#!/usr/bin/env python3
"""
Test unload_model endpoint with fresh model
"""
import requests
import time
import json

BASE_URL = "http://kgateway-training.slime-gmi:8000"
API_KEY = "slime-dev-key"
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

MODEL_ID = "model_db454ea312cb4a64"

print("=" * 80)
print("Testing /api/v1/unload_model endpoint")
print("=" * 80)
print(f"Model ID: {MODEL_ID}")
print()

# Step 1: Check current state
print("[1/4] Checking current state...")
resp = requests.get(f"{BASE_URL}/health", headers=HEADERS)
resp.raise_for_status()
health = resp.json()
print(f"  Active training clients: {health['active_training_clients']}")
print(f"  Futures count: {health['futures_count']}")
print()

# Step 2: Submit unload request
print("[2/4] Submitting unload request...")
payload = {"model_id": MODEL_ID}
resp = requests.post(f"{BASE_URL}/api/v1/unload_model", headers=HEADERS, json=payload)
resp.raise_for_status()
result = resp.json()
request_id = result["request_id"]
print(f"  Request ID: {request_id}")
print()

# Step 3: Poll for completion (use POST, not GET)
print("[3/4] Polling for unload completion...")
max_polls = 30
for i in range(max_polls):
    time.sleep(2)
    resp = requests.post(f"{BASE_URL}/api/v1/retrieve_future", headers=HEADERS, json={"request_id": request_id})
    resp.raise_for_status()
    future_data = resp.json()

    if future_data.get("type") == "try_again":
        print(f"  Still unloading... ({i+1}/{max_polls})")
        continue

    # Completed
    print(f"  ✓ Unload completed!")
    print(f"    Result: {json.dumps(future_data, indent=2)}")
    break
else:
    print(f"  ✗ Timeout waiting for unload")
    exit(1)

print()

# Step 4: Verify cleanup
print("[4/4] Verifying cleanup...")
resp = requests.get(f"{BASE_URL}/health", headers=HEADERS)
resp.raise_for_status()
health = resp.json()
print(f"  Active training clients: {health['active_training_clients']}")
print(f"  Futures count: {health['futures_count']}")

if health['active_training_clients'] == 0:
    print("  ✓ Model successfully unloaded!")
else:
    print(f"  ✗ Expected 0 active clients, got {health['active_training_clients']}")
    exit(1)

print()
print("=" * 80)
print("✓ UNLOAD TEST PASSED")
print("=" * 80)
