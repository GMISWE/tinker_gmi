#!/usr/bin/env python3
"""
Cleanup script for test environment - replaces kgateway restart

This script performs two cleanup operations:
1. Delete all active models (frees GPU resources via Ray)
2. Clear old futures from database (removes stale request state)

Usage:
    python cleanup_test_env.py
    
Environment variables:
    TINKER_BASE_URL - kgateway URL (default: http://kgateway-training.slime-gmi:8000)
    TINKER_API_KEY - API key (default: slime-dev-key)
"""
import requests
import os
import sys

base_url = os.environ.get("TINKER_BASE_URL", "http://kgateway-training.slime-gmi:8000")
api_key = os.environ.get("TINKER_API_KEY", "slime-dev-key")

print("=" * 80)
print("CLEANUP TEST ENVIRONMENT")
print("=" * 80)
print(f"Base URL: {base_url}")
print("=" * 80)

# Step 1: Get list of active models and delete them
print("\n[1/3] Checking for active models...")
try:
    response = requests.get(
        f"{base_url}/health",
        headers={"X-API-Key": api_key},
        timeout=5
    )

    if response.status_code == 200:
        health = response.json()
        model_ids = health.get("model_ids", [])
        active_models = len(model_ids)
        print(f"   Active training clients: {active_models}")

        if active_models == 0:
            print("   ✓ No active models to cleanup")
        else:
            print(f"   ⚠ Warning: {active_models} active models found")
            print(f"   Model IDs: {model_ids}")

            # Delete each model
            for model_id in model_ids:
                print(f"   Deleting model: {model_id}...")
                try:
                    del_response = requests.post(
                        f"{base_url}/api/v1/delete_model",
                        json={"model_id": model_id},
                        headers={"X-API-Key": api_key},
                        timeout=30
                    )

                    if del_response.status_code == 200:
                        print(f"   ✓ Deleted model: {model_id}")
                    else:
                        print(f"   ✗ Failed to delete {model_id}: {del_response.status_code}")

                except Exception as e:
                    print(f"   ✗ Error deleting {model_id}: {e}")
    else:
        print(f"   ✗ Health check failed: {response.status_code}")

except Exception as e:
    print(f"   ✗ Failed to check health: {e}")

# Step 2: Cleanup old futures
print("\n[2/3] Cleaning up old futures from database...")
try:
    response = requests.post(
        f"{base_url}/api/v1/cleanup_futures",
        json={"max_age_hours": 0},  # Delete all futures
        headers={"X-API-Key": api_key},
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        deleted_count = result.get("deleted_count", 0)
        print(f"   ✓ Deleted {deleted_count} futures from database")
    else:
        print(f"   ✗ Cleanup futures failed: {response.status_code} - {response.text}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Failed to cleanup futures: {e}")
    sys.exit(1)

# Step 3: Verify Ray cluster is clean
print("\n[3/3] Verifying Ray cluster state...")
print("   (Manual check recommended: kubectl exec slime-training-0 -- ray list actors)")

print("\n" + "=" * 80)
print("✓ CLEANUP COMPLETE")
print("=" * 80)
print("\nRecommendations:")
print("  1. Each test should call delete_model() to cleanup its own training client")
print("  2. Run this script before test sessions to clear stale futures")
print("  3. Verify Ray actors are cleaned up: ray list actors --filter state=ALIVE")
print("=" * 80)

sys.exit(0)
