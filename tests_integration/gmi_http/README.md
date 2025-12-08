# GMI HTTP Tests

Tests that make **direct HTTP requests** to GMI wrapper endpoints using `httpx` or `requests`.

## Tests

### `test_1_model_creation.py`
Tests model creation endpoint only.

**Purpose:** Verify GMI wrapper can create Slime training actors.

**What it tests:**
- `/api/v1/create_model` endpoint
- `/api/v1/retrieve_future` polling
- Model initialization with Slime RayTrainGroup

**Run:**
```bash
python test_1_model_creation.py
```

**Expected output:**
- Request ID returned
- Future polling completes successfully
- Model ID returned
- Training client initialized

### `test_2_single_train_step.py`
Tests model creation + training step.

**Purpose:** Verify complete training workflow through GMI wrapper.

**What it tests:**
- Model creation
- `/api/v1/forward_backward` endpoint
- `/api/v1/optim_step` endpoint
- Loss computation and gradient updates

**Run:**
```bash
python test_2_single_train_step.py
```

**Expected output:**
- Model created successfully
- Forward-backward completes with non-zero loss
- Optimizer step completes
- Gradients computed and applied

### `test_gmi_wrapper.py`
Comprehensive HTTP API test suite.

**Purpose:** Test all GMI wrapper endpoints and error handling.

**What it tests:**
- Health check endpoint
- Server capabilities
- Model creation and management
- Training endpoints
- Future polling
- Error responses
- Request validation

**Run:**
```bash
export GMI_BASE_URL=http://gmi-wrapper.slime-gmi:8000
export GMI_API_KEY=slime-dev-key
python test_gmi_wrapper.py
```

## Configuration

Tests use environment variables or hardcoded defaults:
- `GMI_BASE_URL` - Default: `http://gmi-wrapper.slime-gmi:8000`
- `GMI_API_KEY` - Default: `slime-dev-key`

## Notes

- These tests use `requests` or `httpx` for HTTP calls
- Tests focus on HTTP API implementation details
- Useful for debugging request/response formats
- Faster iteration than E2E tests with Tinker client
- `dummy.jsonl` contains sample prompt data required by RolloutManager initialization (copy to `/tmp/dummy.jsonl` in slime-training pod before running tests)
