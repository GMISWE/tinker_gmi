# GMI Wrapper Testing Plan

Based on the reference implementation in `/tmp/test_complete_workflow.py`

## Test Configuration

### Service Details
- **URL**: `http://gmi-wrapper.slime-gmi:8000`
- **API Key Header**: `X-API-Key: slime-dev-key`
- **Model Path**: `/data/models/Qwen2.5-0.5B-Instruct_torch_dist`
- **LoRA Config**: `{"rank": 0, "alpha": 0}` (disabled for simpler testing)

### Key Patterns from Reference
1. **Async polling pattern**: Submit operation → get `request_id` → poll `/api/v1/retrieve_future`
2. **Status codes**:
   - `200`: Operation completed, result available
   - `408`: Operation still pending, continue polling
   - `500`: Operation failed
3. **Timeouts**:
   - Model creation: 180 seconds (90 polls × 2s)
   - Forward-backward: 120 seconds (60 polls × 2s)
   - Optimizer step: 60 seconds (30 polls × 2s)

## Test 1: Model Creation Only

### Purpose
Validate that GMI wrapper can successfully initialize Slime RayTrainGroup actors.

### Steps

1. **Health Check** (optional but recommended)
   ```
   GET /health
   Expected: {"status": "healthy", "ray_initialized": true}
   ```

2. **Create Model**
   ```
   POST /api/v1/create_model
   Headers: X-API-Key: slime-dev-key
   Body: {
     "base_model": "/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
     "lora_config": {"rank": 0, "alpha": 0}
   }
   Expected: {"request_id": "req_..."}
   ```

3. **Poll for Completion**
   ```
   POST /api/v1/retrieve_future
   Body: {"request_id": "req_..."}

   Loop until:
   - Status 200: Success, extract model_id
   - Status 408: Continue polling (max 90 attempts × 2s = 180s)
   - Other: Error
   ```

4. **Verify Result**
   ```
   Expected response:
   {
     "model_id": "model_...",
     "base_model": "/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
     "lora_config": {"rank": 0, "alpha": 0},
     "status": "ready"
   }
   ```

5. **Health Check (post-creation)**
   ```
   GET /health
   Expected: {"active_training_clients": 1, ...}
   ```

### Success Criteria
- ✓ Model creation returns `request_id`
- ✓ Polling completes within timeout
- ✓ Result contains valid `model_id`
- ✓ Status is "ready"
- ✓ Health check shows 1 active training client

### Expected Duration
~60-120 seconds (model initialization with Ray actors)

## Test 2: Model Creation + Single Train Step

### Purpose
Validate complete training workflow: create model → forward-backward → optimizer step.

### Steps

1. **Health Check** (optional)
   ```
   GET /health
   ```

2. **Create Model** (same as Test 1)
   ```
   POST /api/v1/create_model
   Body: {
     "base_model": "/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
     "lora_config": {"rank": 0, "alpha": 0}
   }

   Poll /api/v1/retrieve_future until completion
   Extract model_id
   ```

3. **Forward-Backward Pass**
   ```
   POST /api/v1/forward_backward
   Headers: X-API-Key: slime-dev-key
   Body: {
     "model_id": "model_...",
     "data": [{"input": "test", "target": "response"}]
   }
   Expected: {"request_id": "req_..."}
   ```

4. **Poll Forward-Backward Result**
   ```
   POST /api/v1/retrieve_future
   Body: {"request_id": "req_..."}

   Loop until status 200 (max 60 attempts × 2s = 120s)
   ```

5. **Verify Forward-Backward Result**
   ```
   Expected response:
   {
     "loss": {...},  # Loss value (could be dict or float)
     "grad_norm": 0.0,  # Gradient norm
     "valid_step": true,
     "status": "completed"
   }
   ```

6. **Optimizer Step**
   ```
   POST /api/v1/optim_step
   Headers: X-API-Key: slime-dev-key
   Body: {
     "model_id": "model_..."
   }
   Expected: {"request_id": "req_..."}
   ```

7. **Poll Optimizer Step Result**
   ```
   POST /api/v1/retrieve_future
   Body: {"request_id": "req_..."}

   Loop until status 200 (max 30 attempts × 2s = 60s)
   ```

8. **Verify Optimizer Step Result**
   ```
   Expected response:
   {
     "success": true,
     "grad_norm": 0.0,
     "status": "completed"
   }
   ```

9. **Health Check (final state)**
   ```
   GET /health
   Expected: {"active_training_clients": 1, ...}
   ```

### Success Criteria
- ✓ Model creation succeeds
- ✓ Forward-backward pass completes and returns loss/grad_norm
- ✓ Optimizer step completes successfully
- ✓ All operations complete within timeout
- ✓ No errors in any step

### Expected Duration
~90-180 seconds total:
- Model creation: 60-120s
- Forward-backward: 10-30s
- Optimizer step: 10-20s

## Implementation Notes

### Error Handling
1. **Network errors**: Retry with exponential backoff
2. **Timeout**: Clear error message with context
3. **500 errors**: Print full error response for debugging

### Logging
- Print timestamp for each operation
- Show polling progress every 5-10 iterations
- Display final metrics (loss, grad_norm)

### Test Script Structure
```python
import httpx
import time
import sys

GMI_URL = "http://gmi-wrapper.slime-gmi:8000"
API_KEY = "slime-dev-key"
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

def poll_until_complete(request_id, max_attempts=60, interval=2):
    """Poll retrieve_future until completion or timeout"""
    for attempt in range(max_attempts):
        time.sleep(interval)
        response = httpx.post(
            f"{GMI_URL}/api/v1/retrieve_future",
            json={"request_id": request_id},
            headers=headers,
            timeout=30.0
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 408:
            if attempt % 10 == 0:
                print(f"  Polling... ({attempt+1}/{max_attempts})")
            continue
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

    raise TimeoutError(f"Operation timed out after {max_attempts * interval}s")

# Test implementation...
```

## Running the Tests

### Prerequisites
1. GMI wrapper deployed and running
2. Slime StatefulSet with Ray cluster running
3. Model files present at `/data/models/Qwen2.5-0.5B-Instruct_torch_dist`

### Execution
```bash
# Test 1: Model creation only
cd /tmp/tmux-tmp/tinker_gmi
python test_1_model_creation.py

# Test 2: Model creation + single train step
python test_2_single_train_step.py
```

### Expected Output

**Test 1 Success:**
```
================================================================================
GMI Wrapper - Test 1: Model Creation Only
================================================================================
[1/2] Creating Model
✓ Submitted: req_abc123...
  Polling for model creation...
✓ Model created: model_xyz456...

[2/2] Verifying State
✓ Active training clients: 1

================================================================================
✓ TEST 1 PASSED
================================================================================
```

**Test 2 Success:**
```
================================================================================
GMI Wrapper - Test 2: Model Creation + Single Train Step
================================================================================
[1/4] Creating Model
✓ Model created: model_xyz456...

[2/4] Forward-Backward Pass
✓ Completed: loss=2.5, grad_norm=1.2

[3/4] Optimizer Step
✓ Completed: success=True, grad_norm=1.2

[4/4] Final State
✓ Active training clients: 1

================================================================================
✓ TEST 2 PASSED - Training workflow functional!
================================================================================
```

## Troubleshooting

### Model creation times out
- Check Slime pod logs: `kubectl logs -n slime-gmi slime-training-0`
- Verify Ray cluster: `kubectl exec -it slime-training-0 -n slime-gmi -- ray status`
- Check GPU availability: `kubectl exec -it slime-training-0 -n slime-gmi -- nvidia-smi`

### Forward-backward fails
- Check GMI wrapper logs: `kubectl logs -n slime-gmi -l app=gmi-wrapper`
- Verify model exists in active_training_clients
- Check data format is correct

### Optimizer step fails
- Ensure forward-backward ran successfully first
- Check gradients were accumulated
- Review GMI wrapper logs for errors
