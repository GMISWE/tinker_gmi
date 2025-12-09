#!/usr/bin/env python3
"""
Test: Checkpoint Save and Resume

Tests Tinker client's checkpoint save/load workflow:
  1. Create training client
  2. Save state checkpoint
  3. Unload original model (free GPU resources)
  4. Create new training client from checkpoint
  5. Verify restored model works (sampling)

This test validates the /api/v1/weights_info endpoint for checkpoint resume.
"""
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set API key
os.environ["TINKER_API_KEY"] = "slime-dev-key"

import tinker

BASE_URL = "http://kgateway-training.miles-gmi-tinker:8000"
MODEL = "/data/models/Qwen2.5-0.5B-Instruct"


def unload_model_via_http(model_id: str) -> None:
    """Unload model via HTTP endpoint (fallback if SDK method unavailable).

    Uses Tinker's native /api/v1/unload_model endpoint which returns an async
    operation that must be polled via retrieve_future.

    retrieve_future returns:
    - 408 (Request Timeout) if operation is still pending
    - 200 with result if completed successfully
    - 500 if failed
    """
    import requests
    headers = {"X-API-Key": "slime-dev-key", "Content-Type": "application/json"}

    # Call unload_model (returns async operation)
    resp = requests.post(
        f"{BASE_URL}/api/v1/unload_model",
        headers=headers,
        json={"model_id": model_id, "type": "unload_model"}
    )
    resp.raise_for_status()
    result = resp.json()
    request_id = result.get("request_id")

    if request_id:
        # Poll for completion
        start = time.time()
        while time.time() - start < 30:
            poll_resp = requests.post(
                f"{BASE_URL}/api/v1/retrieve_future",
                headers=headers,
                json={"request_id": request_id}
            )
            if poll_resp.status_code == 200:
                # Success - operation completed
                poll_result = poll_resp.json()
                logger.info(f"Unload model response: {poll_result}")
                return
            elif poll_resp.status_code == 500:
                # Failed - operation errored
                error_detail = poll_resp.json().get("detail", "Unknown error")
                raise Exception(f"Unload failed: {error_detail}")
            elif poll_resp.status_code == 408:
                # Still pending - continue polling
                pass
            else:
                logger.warning(f"Unexpected status code: {poll_resp.status_code}")
            time.sleep(0.5)
        raise TimeoutError(f"Timeout waiting for unload of {model_id}")


def main():
    logger.info("=== Starting Checkpoint Save/Resume Test ===")

    # Step 1: Create service client and training client
    logger.info("Step 1: Creating service client...")
    service_client = tinker.ServiceClient(base_url=BASE_URL)
    logger.info("Service client created")

    logger.info("Step 2: Creating training client...")
    training_client = service_client.create_lora_training_client(
        base_model=MODEL,
        rank=16
    )
    logger.info(f"Training client created with model: {training_client.model_id}")

    # Step 3: Save state checkpoint
    logger.info("Step 3: Saving state checkpoint...")
    start_time = time.time()
    save_result = training_client.save_state("test_checkpoint").result()
    elapsed = time.time() - start_time
    logger.info(f"Save completed in {elapsed:.2f}s")
    logger.info(f"Checkpoint path: {save_result.path}")

    # Step 4: Unload original model to free GPU resources
    logger.info("Step 4: Unloading original model to free GPU resources...")
    original_model_id = training_client.model_id
    unload_model_via_http(original_model_id)
    logger.info(f"Original model {original_model_id} unloaded")

    # Step 5: Create new training client from checkpoint
    logger.info("Step 5: Creating new client from checkpoint...")
    start_time = time.time()

    # Use the path from save_result
    checkpoint_path = save_result.path
    logger.info(f"Loading from path: {checkpoint_path}")

    # Create new client from checkpoint state (with optimizer)
    new_client = service_client.create_training_client_from_state_with_optimizer(
        checkpoint_path
    )
    elapsed = time.time() - start_time
    logger.info(f"Load completed in {elapsed:.2f}s")
    logger.info(f"Restored client model ID: {new_client.model_id}")

    # Test sampling from restored client
    logger.info("Step 6: Testing sampling from restored client...")
    sampling_client = new_client.save_weights_and_get_sampling_client("test_sampler")
    logger.info(f"Sampling client created successfully")

    # Do a sample to verify the model works
    test_prompt = "What is 2 + 2?"
    logger.info(f"Sampling with prompt: {test_prompt}")

    # Get tokenizer and encode prompt
    tokenizer = new_client.get_tokenizer()
    tokens = tokenizer.encode(test_prompt, add_special_tokens=True)
    prompt = tinker.ModelInput.from_ints(tokens)
    sampling_params = tinker.SamplingParams(max_tokens=32)

    sample_future = sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params
    )
    sample_result = sample_future.result()
    logger.info(f"Sample result: {sample_result.sequences[0].tokens[:10]}...")

    # Cleanup
    logger.info("Step 7: Cleaning up...")
    unload_model_via_http(new_client.model_id)
    logger.info("Restored model unloaded")

    logger.info("=== Checkpoint Test PASSED ===")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)
