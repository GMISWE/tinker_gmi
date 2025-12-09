#!/usr/bin/env python3
"""
Direct HTTP API test for kgateway-training sampling endpoint
Tests model creation and SGLang sampling via raw HTTP requests
"""
import requests
import time
import json

BASE_URL = "http://kgateway-training.miles-gmi-tinker:8000"
API_KEY = "slime-dev-key"

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

print("=" * 80)
print("Direct HTTP API Test - kgateway-training + SGLang Sampling")
print("=" * 80)

# Step 1: Create model
print("\n1. Creating model...")
response = requests.post(
    f"{BASE_URL}/api/v1/create_model",
    headers=headers,
    json={
        "base_model": "/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
        "lora_config": {
            "rank": 0,  # No LoRA
            "seed": 42,
            "train_mlp": False,
            "train_attn": False,
            "train_unembed": False
        }
    }
)
print(f"Status: {response.status_code}")
create_result = response.json()
print(f"Response: {json.dumps(create_result, indent=2)}")

request_id = create_result["request_id"]
print(f"Request ID: {request_id}")

# Step 2: Poll for model creation completion
print("\n2. Polling for model creation completion...")
max_wait = 300  # 5 minutes
start = time.time()
while time.time() - start < max_wait:
    response = requests.post(
        f"{BASE_URL}/api/v1/retrieve_future",
        headers=headers,
        json={"request_id": request_id}
    )
    result = response.json()

    if response.status_code == 408:
        print(f"  Still waiting... ({int(time.time() - start)}s)")
        time.sleep(5)
        continue

    if response.status_code == 200:
        # Model creation complete
        print(f"✓ Model created successfully!")
        print(f"  Model ID: {result.get('model_id')}")
        model_id = result.get("model_id")
        break

    print(f"✗ Unexpected status {response.status_code}: {result}")
    exit(1)
else:
    print(f"✗ Timeout waiting for model creation")
    exit(1)

# Step 3: Wait for SGLang to be ready
print("\n3. Waiting for SGLang router to be ready...")
time.sleep(15)

# Step 4: Sample using async endpoint
print("\n4. Generating samples with SGLang...")
response = requests.post(
    f"{BASE_URL}/api/v1/asample",
    headers=headers,
    json={
        "model_id": model_id,
        "prompt": {"tokens": [1, 2, 3, 4]},  # API expects "tokens" not "input_ids"
        "num_samples": 2,
        "sampling_params": {
            "max_tokens": 20,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
)
print(f"Status: {response.status_code}")
sample_result = response.json()
print(f"Response: {json.dumps(sample_result, indent=2)}")

sample_request_id = sample_result["request_id"]

# Step 5: Poll for sampling completion
print("\n5. Polling for sampling completion...")
max_wait = 120
start = time.time()
while time.time() - start < max_wait:
    response = requests.post(
        f"{BASE_URL}/api/v1/retrieve_future",
        headers=headers,
        json={"request_id": sample_request_id}
    )
    result = response.json()

    if response.status_code == 408:
        print(f"  Still waiting... ({int(time.time() - start)}s)")
        time.sleep(5)
        continue

    if response.status_code == 200:
        print(f"✓ Sampling completed successfully!")
        sequences = result.get('sequences', [])
        print(f"  Generated {len(sequences)} samples:")
        for i, seq in enumerate(sequences):
            print(f"    Sample {i+1}: {len(seq.get('tokens', []))} tokens, stop_reason={seq.get('stop_reason')}")
            print(f"      First 5 tokens: {seq.get('tokens', [])[:5]}")
        break

    print(f"✗ Unexpected status {response.status_code}: {result}")
    exit(1)
else:
    print(f"✗ Timeout waiting for sampling")
    exit(1)

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED!")
print("=" * 80)
