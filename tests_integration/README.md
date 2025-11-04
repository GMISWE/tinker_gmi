# Integration Tests for Tinker GMI Wrapper

This directory contains integration tests for the GMI Wrapper V3, organized by testing approach.

## Directory Structure

```
tests_integration/
├── e2e_tinker_api/     # Tests using official Tinker client library
├── gmi_http/           # Tests using direct HTTP calls to GMI wrapper
└── README.md           # This file
```

## Test Groups

### 1. E2E Tinker API Tests (`e2e_tinker_api/`)

These tests use the **official Tinker Python client library** to verify end-to-end compatibility with the Tinker API specification.

**Purpose:** Ensure the GMI wrapper correctly implements the Tinker API and can be used as a drop-in replacement for Tinker servers.

**Tests:**
- `test_mock_server.py` - Tests against a mock Tinker server for baseline verification
- `test_tinker_gmi_wrapper.py` - Tests Tinker client against GMI wrapper
- `test_grpo_single_step.py` - Full GRPO training step test (sampling + training)

**How to run:**
```bash
cd tests_integration/e2e_tinker_api
export TINKER_BASE_URL=http://gmi-wrapper.slime-gmi:8000
export TINKER_API_KEY=slime-dev-key
python test_tinker_gmi_wrapper.py
```

### 2. GMI HTTP Tests (`gmi_http/`)

These tests make **direct HTTP requests** to the GMI wrapper endpoints using `httpx` or `requests`.

**Purpose:** Test the GMI wrapper's HTTP API implementation details, request/response formats, and error handling.

**Tests:**
- `test_1_model_creation.py` - Tests model creation endpoint only
- `test_2_single_train_step.py` - Tests model creation + training step
- `test_gmi_wrapper.py` - Comprehensive HTTP API test suite

**How to run:**
```bash
cd tests_integration/gmi_http
python test_1_model_creation.py
python test_2_single_train_step.py
```

## When to Use Each Test Group

- **Use E2E Tinker API tests** when:
  - Verifying Tinker API compatibility
  - Testing with the official Tinker client library
  - Ensuring drop-in replacement capability
  - Running GRPO/RL training workflows

- **Use GMI HTTP tests** when:
  - Debugging HTTP request/response formats
  - Testing error handling and edge cases
  - Verifying specific endpoint behavior
  - Quick iteration on API changes

## Related Test Directories

- `tests/` - Unit tests for Tinker client library internals
- `tests/api_resources/` - Unit tests for API resource classes
