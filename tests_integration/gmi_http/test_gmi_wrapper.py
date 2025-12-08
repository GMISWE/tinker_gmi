#!/usr/bin/env python3
"""Test script for GMI Wrapper V3 with Slime backend.

This script tests the Tinker API implementation via GMI wrapper.

Before running this script:
1. Ensure GMI Wrapper V3 is deployed and running
2. Set environment variables:
   export GMI_BASE_URL=http://<gmi-service>:8000
   export GMI_API_KEY=slime-dev-key
3. Run this script: python test_gmi_wrapper.py
"""

import os
import sys
import time
from typing import Any, Dict, Optional

import requests

# Configuration
GMI_BASE_URL = os.getenv("GMI_BASE_URL", "http://localhost:8000")
GMI_API_KEY = os.getenv("GMI_API_KEY", "slime-dev-key")
POLL_INTERVAL = 2.0  # seconds
MAX_WAIT_TIME = 300  # 5 minutes max


class GMIClient:
    """Client for GMI Wrapper V3 API"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": api_key})

    def _poll_future(self, request_id: str, timeout: float = MAX_WAIT_TIME) -> Dict[str, Any]:
        """Poll for future result until completed or timeout"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                resp = self.session.post(
                    f"{self.base_url}/api/v1/retrieve_future",
                    json={"request_id": request_id}
                )

                if resp.status_code == 200:
                    # Operation completed
                    return resp.json()

                elif resp.status_code == 408:
                    # Still pending - continue polling
                    time.sleep(POLL_INTERVAL)
                    continue

                elif resp.status_code == 500:
                    # Operation failed
                    raise Exception(f"Operation failed: {resp.text}")

                else:
                    raise Exception(f"Unexpected status code {resp.status_code}: {resp.text}")

            except requests.RequestException as e:
                raise Exception(f"Network error while polling future: {e}")

        raise TimeoutError(f"Operation timed out after {timeout} seconds")

    def create_model(self, base_model: str, lora_rank: int = 32) -> str:
        """
        Create a model and return model_id

        Returns:
            model_id: The created model ID
        """
        resp = self.session.post(
            f"{self.base_url}/api/v1/create_model",
            json={
                "base_model": base_model,
                "lora_config": {
                    "rank": lora_rank,
                    "alpha": lora_rank,
                    "dropout": 0.0
                }
            }
        )
        resp.raise_for_status()

        request_id = resp.json()["request_id"]
        print(f"  Created request: {request_id}, waiting for completion...")

        # Poll for result
        result = self._poll_future(request_id)
        return result["model_id"]

    def forward_backward(self, model_id: str, data: list = None, loss_fn: str = "cross_entropy") -> Dict[str, Any]:
        """
        Perform forward-backward pass

        Returns:
            Dict with loss, grad_norm, etc.
        """
        resp = self.session.post(
            f"{self.base_url}/api/v1/forward_backward",
            json={
                "model_id": model_id,
                "data": data or [],
                "loss_fn": loss_fn
            }
        )
        resp.raise_for_status()

        request_id = resp.json()["request_id"]

        # Poll for result
        return self._poll_future(request_id)

    def optim_step(self, model_id: str) -> Dict[str, Any]:
        """
        Apply optimizer step

        Returns:
            Dict with success status, grad_norm
        """
        resp = self.session.post(
            f"{self.base_url}/api/v1/optim_step",
            json={"model_id": model_id}
        )
        resp.raise_for_status()

        request_id = resp.json()["request_id"]

        # Poll for result
        return self._poll_future(request_id)

    def save_weights(self, model_id: str, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Save model weights

        Returns:
            Dict with checkpoint_path
        """
        payload = {"model_id": model_id}
        if checkpoint_path:
            payload["checkpoint_path"] = checkpoint_path

        resp = self.session.post(
            f"{self.base_url}/api/v1/save_weights",
            json=payload
        )
        resp.raise_for_status()

        request_id = resp.json()["request_id"]

        # Poll for result
        return self._poll_future(request_id)

    def health_check(self) -> Dict[str, Any]:
        """Check health status"""
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()


def test_health_check():
    """Test 1: Health Check"""
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)

    try:
        client = GMIClient(GMI_BASE_URL, GMI_API_KEY)
        health = client.health_check()

        print(f"✓ Health check passed:")
        print(f"  Status: {health.get('status')}")
        print(f"  Ray initialized: {health.get('ray_initialized')}")
        print(f"  Active training clients: {health.get('active_training_clients')}")
        print(f"  Pending futures: {health.get('futures_count')}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test 2: Model Creation"""
    print("\n" + "="*80)
    print("TEST 2: Model Creation")
    print("="*80)

    try:
        client = GMIClient(GMI_BASE_URL, GMI_API_KEY)

        print("Creating model with LoRA rank=32...")
        model_id = client.create_model(
            base_model="Qwen/Qwen2.5-0.5B_torch_dist",
            lora_rank=32
        )

        print(f"✓ Model created successfully:")
        print(f"  Model ID: {model_id}")

        return True, model_id

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_training_step(model_id: str):
    """Test 3: Single Training Step"""
    print("\n" + "="*80)
    print("TEST 3: Single Training Step")
    print("="*80)

    try:
        client = GMIClient(GMI_BASE_URL, GMI_API_KEY)

        # Forward-backward pass
        print("Performing forward-backward pass...")
        fb_result = client.forward_backward(model_id)

        print(f"✓ Forward-backward completed:")
        print(f"  Loss: {fb_result.get('loss')}")
        print(f"  Grad norm: {fb_result.get('grad_norm')}")
        print(f"  Valid step: {fb_result.get('valid_step')}")

        # Optimizer step
        print("\nApplying optimizer step...")
        optim_result = client.optim_step(model_id)

        print(f"✓ Optimizer step completed:")
        print(f"  Success: {optim_result.get('success')}")
        print(f"  Grad norm: {optim_result.get('grad_norm')}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_accumulation(model_id: str):
    """Test 4: Gradient Accumulation"""
    print("\n" + "="*80)
    print("TEST 4: Gradient Accumulation (Multiple Forward-Backward)")
    print("="*80)

    try:
        client = GMIClient(GMI_BASE_URL, GMI_API_KEY)

        num_accumulation_steps = 3
        print(f"Performing {num_accumulation_steps} forward-backward passes...")

        losses = []
        for i in range(num_accumulation_steps):
            print(f"\n  Step {i+1}/{num_accumulation_steps}...")
            fb_result = client.forward_backward(model_id)
            loss = fb_result.get('loss')
            losses.append(loss)
            print(f"    Loss: {loss}")
            print(f"    Grad norm: {fb_result.get('grad_norm')}")

        print(f"\n✓ Accumulated {num_accumulation_steps} gradients")

        # Apply optimizer step
        print("\nApplying optimizer step to accumulated gradients...")
        optim_result = client.optim_step(model_id)

        print(f"✓ Optimizer step completed:")
        print(f"  Success: {optim_result.get('success')}")
        print(f"  Grad norm: {optim_result.get('grad_norm')}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_save(model_id: str):
    """Test 5: Checkpoint Save"""
    print("\n" + "="*80)
    print("TEST 5: Checkpoint Save")
    print("="*80)

    try:
        client = GMIClient(GMI_BASE_URL, GMI_API_KEY)

        checkpoint_path = f"/data/checkpoints/test_{int(time.time())}"
        print(f"Saving checkpoint to: {checkpoint_path}")

        result = client.save_weights(model_id, checkpoint_path)

        print(f"✓ Checkpoint saved:")
        print(f"  Path: {result.get('checkpoint_path')}")
        print(f"  Status: {result.get('status')}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("GMI WRAPPER V3 TEST SUITE")
    print("="*80)
    print(f"Base URL: {GMI_BASE_URL}")
    print(f"API Key: {GMI_API_KEY[:10]}..." if len(GMI_API_KEY) > 10 else f"API Key: {GMI_API_KEY}")
    print("="*80)

    results = []
    model_id = None

    # Test 1: Health Check
    results.append(("Health Check", test_health_check()))

    # Test 2: Model Creation
    success, model_id = test_model_creation()
    results.append(("Model Creation", success))

    if not model_id:
        print("\n" + "="*80)
        print("CRITICAL: Model creation failed. Skipping remaining tests.")
        print("="*80)
        return 1

    # Test 3: Single Training Step
    results.append(("Single Training Step", test_training_step(model_id)))

    # Test 4: Gradient Accumulation
    results.append(("Gradient Accumulation", test_gradient_accumulation(model_id)))

    # Test 5: Checkpoint Save
    results.append(("Checkpoint Save", test_checkpoint_save(model_id)))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("="*80)
    print(f"Results: {passed}/{total} tests passed")
    print("="*80 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
