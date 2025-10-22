# Mock Server Testing Guide

## Overview

The Tinker mock server provides a complete implementation of the Tinker API for local testing and development. It supports all endpoints needed by the tinker-cookbook recipes, including REST API endpoints for training run management and checkpoint operations.

## Quick Start

### 1. Start the Mock Server

```bash
cd tinker_gmi
python mock_server.py
```

The server will start on `http://localhost:8000` with API documentation available at `http://localhost:8000/docs`.

### 2. Run Tests

In another terminal:

```bash
cd tinker_gmi
export TINKER_BASE_URL=http://localhost:8000
export TINKER_API_KEY=test-key
python test_mock_server.py
```

### Expected Output

```
MOCK TINKER API SERVER TEST SUITE
================================================================================
Base URL: http://localhost:8000
API Key: test-key
================================================================================

TEST 1: Service Client - Health Check
✓ Server capabilities retrieved

TEST 2: Training Client - Create Model and Training
✓ Training client created
✓ Forward-backward completed
✓ Optimizer step completed
✓ Weights saved

TEST 3: Sampling Client - Text Generation
✓ Sampling client created
✓ Generated 2 samples

TEST 4: REST Client - Training Runs API
✓ Training client created
✓ Training run retrieved
✓ Found training runs

TEST 5: Resume Training from State
✓ Created initial training client
✓ State saved
✓ Resumed training client
✓ LoRA rank verified

TEST 6: Checkpoint Operations
✓ Created training client
✓ Saved 2 checkpoints
✓ Found 2 checkpoints
✓ Downloaded checkpoint archive
✓ Checkpoint deleted

================================================================================
TEST SUMMARY
================================================================================
✓ PASS: Service Client
✓ PASS: Training Flow
✓ PASS: Sampling Flow
✓ PASS: REST Client - Training Runs
✓ PASS: Resume Training from State
✓ PASS: Checkpoint Operations
================================================================================
Results: 6/6 tests passed
================================================================================
```

## Test Coverage

### Core Training APIs
- ✅ Service client creation and health checks
- ✅ LoRA model creation with configuration
- ✅ Forward and forward-backward passes
- ✅ Optimizer steps with Adam parameters
- ✅ State and sampler weights saving
- ✅ Model loading from saved state

### Sampling APIs
- ✅ Sampling client creation (base model and custom weights)
- ✅ Synchronous and asynchronous sampling
- ✅ Prompt logprobs computation

### REST APIs
- ✅ Get training run metadata by ID
- ✅ List all training runs with pagination
- ✅ List checkpoints for a training run
- ✅ Download checkpoint archives
- ✅ Delete checkpoints

### Advanced Features
- ✅ Resume training from state (`create_training_client_from_state`)
- ✅ Checkpoint management and tracking
- ✅ Future-based async operations

## Supported Cookbook Recipes

The mock server now supports **all** cookbook recipes:

### Basic Examples
- ✅ `sl_loop.py` - Supervised learning loop
- ✅ `rl_loop.py` - Reinforcement learning loop
- ✅ `sl_basic.py` - Basic supervised fine-tuning

### Advanced Examples
- ✅ `preference/rlhf/rlhf_pipeline.py` - Full RLHF pipeline (SFT → RM → RL)
- ✅ `math_rl/train.py` - Math reasoning with RL
- ✅ `preference/dpo/train.py` - Direct preference optimization
- ✅ `eval/run_inspect_evals.py` - Evaluation scripts

## API Endpoints

### Service Endpoints
- `GET /api/v1/healthz` - Health check
- `GET /api/v1/get_server_capabilities` - Server capabilities

### Model Management
- `POST /api/v1/create_model` - Create model with LoRA config
- `POST /api/v1/get_info` - Get model information
- `POST /api/v1/unload_model` - Unload model

### Training Endpoints
- `POST /api/v1/forward` - Forward pass
- `POST /api/v1/forward_backward` - Forward-backward pass
- `POST /api/v1/optim_step` - Optimizer step

### Sampling Endpoints
- `POST /api/v1/sample` - Synchronous sampling
- `POST /api/v1/asample` - Asynchronous sampling

### Weights/Checkpoints
- `POST /api/v1/load_weights` - Load weights
- `POST /api/v1/save_weights` - Save training state
- `POST /api/v1/save_weights_for_sampler` - Save sampler weights
- `GET /api/v1/models/{model_id}/checkpoints` - List checkpoints
- `DELETE /api/v1/training_runs/{id}/checkpoints/{checkpoint_id}` - Delete checkpoint

### REST API (Training Runs)
- `GET /api/v1/training_runs/{training_run_id}` - Get training run metadata
- `GET /api/v1/training_runs` - List training runs with pagination
- `GET /api/v1/training_runs/{id}/checkpoints/{checkpoint_id}/archive` - Download checkpoint

### Futures
- `POST /api/v1/retrieve_future` - Retrieve async operation results
- `POST /api/v1/telemetry` - Send telemetry events

## Testing with Cookbook Recipes

To test cookbook recipes with the mock server:

```bash
# Terminal 1: Start mock server
cd tinker_gmi
python mock_server.py

# Terminal 2: Run a cookbook recipe
cd tinker-cookbook
export TINKER_BASE_URL=http://localhost:8000
export TINKER_API_KEY=test-key

# Example: Run basic supervised learning
python -m tinker_cookbook.recipes.sl_loop
```

## Troubleshooting

### Connection Refused
- Ensure the mock server is running on port 8000
- Check that `TINKER_BASE_URL` is set to `http://localhost:8000`

### Authentication Errors
- Ensure `TINKER_API_KEY` is set (any non-empty value works)
- The mock server accepts any API key for testing

### Test Failures
- Check server logs in Terminal 1 for error details
- Ensure no other process is using port 8000
- Restart the mock server if it becomes unresponsive

## Development

### Adding New Tests

Add test functions to `test_mock_server.py` following the pattern:

```python
def test_my_feature():
    """Test description."""
    print("\n" + "="*80)
    print("TEST N: My Feature")
    print("="*80)

    try:
        # Test implementation
        print("✓ Step completed")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
```

Then add to `main()`:

```python
results.append(("My Feature", test_my_feature()))
```

### Extending Mock Server

To add new endpoints to `mock_server.py`:

1. Define request/response models (if needed)
2. Add endpoint handler with `@app.post` or `@app.get`
3. Update in-memory storage if needed
4. Add corresponding test in `test_mock_server.py`

## Notes

- The mock server stores all data in-memory (resets on restart)
- Checkpoint downloads return mock tar.gz archives
- Futures complete nearly instantly (no actual training occurs)
- All model operations are simulated for fast testing
