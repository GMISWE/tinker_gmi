# Tinker API Testing with GMI Wrapper V3

This directory contains test scripts for validating the Tinker API implementation via GMI Wrapper V3 with Slime backend.

## Architecture

```
Tinker Client → GMI Wrapper V3 → Slime (Ray Actors)
```

- **Tinker Client**: Test scripts using Tinker API
- **GMI Wrapper V3**: FastAPI REST API translating Tinker calls to Slime operations
- **Slime**: Persistent Ray actors maintaining training state for gradient accumulation

## Files

### Test Scripts

- **`test_mock_server.py`**: Original test script for Tinker mock server
  - Tests synchronous Tinker API
  - Uses `tinker` Python package
  - For local mock server testing

- **`test_gmi_wrapper.py`**: Adapted test script for GMI Wrapper V3
  - Tests async GMI REST API
  - Uses direct HTTP requests
  - For deployed GMI wrapper on Kubernetes

### Mock Server

- **`mock_server.py`**: Local Tinker API mock server for development

## Quick Start

### Option 1: Test Deployed GMI Wrapper V3

1. **Get GMI service endpoint:**
   ```bash
   # If using Kubernetes port-forward
   kubectl port-forward -n slime-gmi svc/gmi-wrapper 8000:8000

   # Or check service endpoint
   kubectl get svc -n slime-gmi gmi-wrapper
   ```

2. **Set environment variables:**
   ```bash
   export GMI_BASE_URL=http://localhost:8000  # or http://$GMI_IP:8000
   export GMI_API_KEY=slime-dev-key
   ```

3. **Run tests:**
   ```bash
   cd tinker_gmi
   python test_gmi_wrapper.py
   ```

### Option 2: Test Local Mock Server

1. **Start mock server:**
   ```bash
   cd tinker_gmi
   python mock_server.py
   ```

2. **In another terminal:**
   ```bash
   cd tinker_gmi
   export TINKER_BASE_URL=http://localhost:8000
   export TINKER_API_KEY=test-key
   python test_mock_server.py
   ```

## Test Suite

### GMI Wrapper Tests (`test_gmi_wrapper.py`)

1. **Health Check**
   - Verifies GMI wrapper is running
   - Checks Ray initialization status
   - Shows active training clients

2. **Model Creation**
   - Creates persistent Slime RayTrainGroup actors
   - Configures LoRA training (rank=32)
   - Returns `model_id` for subsequent operations

3. **Single Training Step**
   - Forward-backward pass (accumulates gradients)
   - Optimizer step (applies gradients)
   - Returns loss and grad_norm metrics

4. **Gradient Accumulation**
   - Multiple forward-backward calls (3 steps)
   - Single optimizer step to apply accumulated gradients
   - Validates true gradient accumulation semantics

5. **Checkpoint Save**
   - Saves model checkpoint to JuiceFS/persistent storage
   - Returns checkpoint path

## Key Differences: Mock Server vs GMI Wrapper

| Feature | Mock Server | GMI Wrapper V3 |
|---------|-------------|----------------|
| **API Pattern** | Synchronous | Async (request_id + polling) |
| **Base URL** | `http://localhost:8000` | `http://<k8s-service>:8000` |
| **Authentication** | Optional header | Required `x-api-key` header |
| **Endpoint Prefix** | `/` | `/api/v1/` |
| **Model Creation** | `create_lora_training_client()` | POST `/api/v1/create_model` |
| **Training** | `forward_backward()` returns Future | POST `/api/v1/forward_backward` + poll `/retrieve_future` |
| **State Persistence** | In-memory (ephemeral) | Persistent Ray actors |
| **Gradient Accumulation** | Simulated | True (Slime RayTrainGroup) |

## GMI API Endpoints

### POST `/api/v1/create_model`
Create a model and initialize persistent training actors.

**Request:**
```json
{
  "base_model": "Qwen/Qwen2.5-0.5B_torch_dist",
  "lora_config": {
    "rank": 32,
    "alpha": 32,
    "dropout": 0.0
  }
}
```

**Response:**
```json
{
  "request_id": "req_abc123..."
}
```

### POST `/api/v1/forward_backward`
Perform forward-backward pass (accumulates gradients without optimizer step).

**Request:**
```json
{
  "model_id": "model_xyz...",
  "data": [],
  "loss_fn": "cross_entropy"
}
```

**Response:**
```json
{
  "request_id": "req_def456..."
}
```

### POST `/api/v1/optim_step`
Apply accumulated gradients with optimizer step.

**Request:**
```json
{
  "model_id": "model_xyz..."
}
```

**Response:**
```json
{
  "request_id": "req_ghi789..."
}
```

### POST `/api/v1/retrieve_future`
Poll for operation result.

**Request:**
```json
{
  "request_id": "req_abc123..."
}
```

**Response (completed):**
```json
{
  "model_id": "model_xyz...",
  "base_model": "Qwen/Qwen2.5-0.5B_torch_dist",
  "status": "ready"
}
```

**Response (pending) - HTTP 408:**
```json
{
  "request_id": "req_abc123...",
  "type": "try_again"
}
```

### POST `/api/v1/save_weights`
Save model checkpoint.

**Request:**
```json
{
  "model_id": "model_xyz...",
  "checkpoint_path": "/data/checkpoints/my_checkpoint"
}
```

**Response:**
```json
{
  "request_id": "req_jkl012..."
}
```

### GET `/health`
Health check endpoint (no auth required).

**Response:**
```json
{
  "status": "healthy",
  "ray_initialized": true,
  "active_training_clients": 1,
  "futures_count": 3
}
```

## Configuration

### Environment Variables

**For GMI Wrapper Tests:**
- `GMI_BASE_URL`: GMI wrapper endpoint (default: `http://localhost:8000`)
- `GMI_API_KEY`: API key for authentication (default: `slime-dev-key`)

**For Mock Server Tests:**
- `TINKER_BASE_URL`: Mock server endpoint (default: `http://localhost:8000`)
- `TINKER_API_KEY`: API key for mock server (default: `test-key`)

## Deployment Information

### GMI Wrapper V3 Deployment

The GMI Wrapper is deployed on Kubernetes with:
- **Namespace**: `slime-gmi`
- **Service**: `gmi-wrapper` (ClusterIP)
- **Image**: `us-west1-docker.pkg.dev/devv-404803/gmi-test-repo/gmi-wrapper:v3-box`
- **Port**: 8000
- **Slime StatefulSet**: `slime-training` (uses H200 node with 4 GPUs)
- **Model**: Qwen2.5-0.5B-Instruct at `/data/models/`

### Accessing the Deployed Service

**Port-forward (for testing):**
```bash
kubectl port-forward -n slime-gmi svc/gmi-wrapper 8000:8000
```

**Check deployment status:**
```bash
# Check GMI wrapper
kubectl get pods -n slime-gmi -l app=gmi-wrapper
kubectl logs -n slime-gmi -l app=gmi-wrapper --tail=50

# Check Slime training pod
kubectl get pods -n slime-gmi -l app=slime-training
kubectl logs -n slime-gmi slime-training-0 --tail=50
```

## Troubleshooting

### Test Failures

**Connection refused:**
- Check if GMI wrapper is running: `kubectl get pods -n slime-gmi`
- Verify port-forward or LoadBalancer is working
- Test health endpoint: `curl $GMI_BASE_URL/health`

**Authentication failed (401):**
- Verify API key matches deployment: `kubectl get deployment gmi-wrapper -n slime-gmi -o yaml | grep TINKER_API_KEY`
- Update `GMI_API_KEY` environment variable

**Operation timeout:**
- Check Ray cluster status: `kubectl exec -it slime-training-0 -n slime-gmi -- ray status`
- Review GMI wrapper logs: `kubectl logs -n slime-gmi -l app=gmi-wrapper`
- Check Slime pod resources: `kubectl describe pod slime-training-0 -n slime-gmi`

**Model creation fails:**
- Check GPU availability: `kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'`
- Verify model path exists in persistent storage
- Review Slime actor initialization in logs

### Checking Logs

**GMI Wrapper logs:**
```bash
kubectl logs -n slime-gmi -l app=gmi-wrapper --tail=100 -f
```

**Ray cluster logs:**
```bash
kubectl exec -it slime-training-0 -n slime-gmi -- tail -f /tmp/ray/session_latest/logs/dashboard.log
```

**Check Ray status:**
```bash
kubectl exec -it slime-training-0 -n slime-gmi -- ray status
```

**Check model files:**
```bash
kubectl exec -it slime-training-0 -n slime-gmi -- ls -lh /data/models/
```

## Next Steps

1. **Run basic tests** to verify deployment
2. **Implement data conversion** from Tinker datum format to Slime rollout format
3. **Add sampling support** for inference testing
4. **Performance tuning** for production workloads
5. **Multi-GPU testing** with distributed training

## References

- [Slime Documentation](https://github.com/THUDM/slime)
- [GMI Wrapper V3 Deployment Guide](../arsenal/dyn/slime-rl-deploy/DEPLOY_V3.md)
- [Tinker API Specification](./CLAUDE.md)
