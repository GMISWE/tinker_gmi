# End-to-End Tinker API Tests

Tests that use the **official Tinker Python client library** to verify GMI wrapper compatibility.

## Tests

### `test_mock_server.py`
Tests the Tinker client against a mock server for baseline verification.

**Purpose:** Verify that the test environment and Tinker client work correctly before testing against GMI wrapper.

**Requirements:**
- Mock server running at `http://localhost:8000`
- API key: `test-key`

**Run:**
```bash
# Terminal 1: Start mock server
python mock_server.py

# Terminal 2: Run test
export TINKER_BASE_URL=http://localhost:8000
export TINKER_API_KEY=test-key
python test_mock_server.py
```

### `test_tinker_gmi_wrapper.py`
Tests the Tinker client against GMI wrapper for compatibility verification.

**Purpose:** Verify GMI wrapper correctly implements Tinker API endpoints.

**Requirements:**
- GMI wrapper deployed and accessible
- Slime backend initialized

**Run:**
```bash
export TINKER_BASE_URL=http://gmi-wrapper.slime-gmi:8000
export TINKER_API_KEY=slime-dev-key
python test_tinker_gmi_wrapper.py
```

**Tests:**
1. Service client health check
2. Server capabilities retrieval
3. Model creation
4. Forward-backward pass
5. Optimizer step

### `test_kgateway_training.py`
Tests the Tinker client against kgateway-training service for production compatibility.

**Purpose:** Verify kgateway-training correctly implements Tinker API with async model initialization.

**Requirements:**
- kgateway-training deployed and accessible at `http://kgateway-training.slime-gmi:8000`
- Slime backend initialized with Ray cluster

**Run:**
```bash
export TINKER_BASE_URL=http://kgateway-training.slime-gmi:8000
export TINKER_API_KEY=slime-dev-key
python test_kgateway_training.py
```

**Tests:**
1. Service client creation
2. Server capabilities retrieval
3. Model creation with async initialization (tests placement group and actor initialization)

**Kubernetes Deployment:**
See `arsenal/dyn/slime-rl-deploy/tinker-test-kgateway-pod.yaml` for running this test in a Kubernetes pod.

### `test_grpo_single_step.py` (TODO)
Full GRPO training step test including sampling and training.

**Purpose:** Verify complete GRPO workflow:
1. Create training client
2. Save weights for sampler
3. Create sampling client
4. Generate trajectory samples
5. Compute advantages
6. Train on trajectories
7. Update weights

**Requirements:**
- GMI wrapper with sampling endpoints implemented
- Sufficient GPU resources for training + sampling

**Run:**
```bash
export TINKER_BASE_URL=http://gmi-wrapper.slime-gmi:8000
export TINKER_API_KEY=slime-dev-key
python test_grpo_single_step.py
```

## Environment Variables

All tests require:
- `TINKER_BASE_URL` - Base URL of Tinker API server
- `TINKER_API_KEY` - API key for authentication

## Notes

- These tests use the official `tinker` Python package
- Tests verify API compatibility, not implementation details
- Use these tests before deploying GMI wrapper to production
