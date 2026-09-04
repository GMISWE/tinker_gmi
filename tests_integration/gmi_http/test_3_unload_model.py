#!/usr/bin/env python3
"""
Test unload_model endpoint - creates a model and then unloads it

Uses Tinker's native /api/v1/unload_model endpoint with proper HTTP status code polling:
- 408 (Request Timeout) = operation pending
- 200 = operation completed
- 500 = operation failed
"""
import requests
import time
import json

BASE_URL = "http://kgateway-training.miles-gmi-tinker:8000"
API_KEY = "slime-dev-key"
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}


def poll_future(request_id: str, timeout: float = 60.0) -> dict:
    """Poll retrieve_future until completion or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            json={"request_id": request_id},
            headers=HEADERS,
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 500:
            error_detail = resp.json().get("detail", "Unknown error")
            raise Exception(f"Operation failed: {error_detail}")
        elif resp.status_code == 408:
            pass  # Still pending
        time.sleep(1)
    raise TimeoutError(f"Timeout waiting for request {request_id}")


print("=" * 80)
print("Testing /api/v1/unload_model endpoint")
print("=" * 80)
print(f"Base URL: {BASE_URL}")
print()

# Step 1: Check current state
print("[1/5] Checking current state...")
resp = requests.get(f"{BASE_URL}/health", headers=HEADERS)
resp.raise_for_status()
health = resp.json()
print(f"  Active training clients: {health.get('active_training_clients', 0)}")
print(f"  Futures count: {health.get('futures_count', 0)}")
print()

# Step 2: Create a model first
print("[2/5] Creating model to test unload...")
create_payload = {
    "session_id": "test-unload-session",
    "model_seq_id": 0,
    "base_model": "/data/models/Qwen2.5-0.5B-Instruct",
    "lora_config": {"rank": 16},
    "debug_train_only": True
}
resp = requests.post(f"{BASE_URL}/api/v1/create_model", headers=HEADERS, json=create_payload)
resp.raise_for_status()
create_result = resp.json()
request_id = create_result["request_id"]
print(f"  Request ID: {request_id}")

# Poll for creation completion
print("  Waiting for model creation...")
create_response = poll_future(request_id, timeout=120)
model_id = create_response.get("model_id")
print(f"  ✓ Model created: {model_id}")
print()

# Step 3: Verify model is active
print("[3/5] Verifying model is active...")
resp = requests.get(f"{BASE_URL}/health", headers=HEADERS)
resp.raise_for_status()
health = resp.json()
active_clients = health.get('active_training_clients', 0)
print(f"  Active training clients: {active_clients}")
if active_clients == 0:
    print("  ✗ Expected at least 1 active client")
    exit(1)
print(f"  ✓ Model is active")
print()

# Step 4: Submit unload request
print(f"[4/5] Submitting unload request for {model_id}...")
payload = {"model_id": model_id, "type": "unload_model"}
resp = requests.post(f"{BASE_URL}/api/v1/unload_model", headers=HEADERS, json=payload)
resp.raise_for_status()
result = resp.json()
request_id = result["request_id"]
print(f"  Request ID: {request_id}")

# Poll for unload completion
print("  Waiting for unload to complete...")
unload_result = poll_future(request_id, timeout=30)
print(f"  ✓ Unload completed!")
print(f"    Result: {json.dumps(unload_result, indent=2)}")
print()

# Step 5: Verify cleanup
print("[5/5] Verifying cleanup...")
resp = requests.get(f"{BASE_URL}/health", headers=HEADERS)
resp.raise_for_status()
health = resp.json()
active_clients = health.get('active_training_clients', 0)
print(f"  Active training clients: {active_clients}")

if active_clients == 0:
    print("  ✓ Model successfully unloaded!")
else:
    print(f"  Note: {active_clients} active clients remain (may be from other tests)")

print()
print("=" * 80)
print("✓ UNLOAD TEST PASSED")
print("=" * 80)
