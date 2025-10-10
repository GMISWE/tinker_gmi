#!/usr/bin/env python3
"""Mock Tinker API Server for testing SDK integration.

Usage:
    python mock_server.py

This creates a mock server that implements the Tinker API endpoints.
Configure your SDK to point to this server:
    export TINKER_BASE_URL=http://localhost:8000
    export TINKER_API_KEY=test-key
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Mock Tinker API Server", version="0.1.0")

# In-memory storage
futures_store: Dict[str, Dict[str, Any]] = {}
models_store: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class LoraConfig(BaseModel):
    rank: int = 32
    seed: Optional[int] = None
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True


class CreateModelRequest(BaseModel):
    base_model: str
    lora_config: Optional[LoraConfig] = None
    type: str = "create_model"


class GetInfoRequest(BaseModel):
    model_id: str
    type: str = "get_info"


class UnloadModelRequest(BaseModel):
    model_id: str
    type: str = "unload_model"


class ForwardBackwardInputDatum(BaseModel):
    model_input: Dict[str, Any]
    loss_fn_inputs: Dict[str, Any]


class ForwardBackwardInput(BaseModel):
    data: List[ForwardBackwardInputDatum]
    loss_fn: str


class ForwardRequest(BaseModel):
    model_id: str
    forward_input: ForwardBackwardInput


class ForwardBackwardRequest(BaseModel):
    model_id: str
    forward_backward_input: ForwardBackwardInput


class AdamParams(BaseModel):
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None


class OptimStepRequest(BaseModel):
    model_id: str
    adam_params: AdamParams
    type: str = "optim_step"


class ModelInput(BaseModel):
    chunks: List[Dict[str, Any]]
    length: int = 0


class SamplingParams(BaseModel):
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Any] = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0


class SampleRequest(BaseModel):
    num_samples: int = 1
    prompt: ModelInput
    sampling_params: SamplingParams
    base_model: Optional[str] = None
    model_path: Optional[str] = None
    prompt_logprobs: Optional[bool] = None
    type: str = "sample"


class LoadWeightsRequest(BaseModel):
    model_id: str
    path: str
    type: str = "load_weights"


class SaveWeightsRequest(BaseModel):
    model_id: str
    path: Optional[str] = None
    type: str = "save_weights"


class SaveWeightsForSamplerRequest(BaseModel):
    model_id: str
    path: Optional[str] = None
    type: str = "save_weights_for_sampler"


class FutureRetrieveRequest(BaseModel):
    request_id: str


# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    print(f"\n{'='*80}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {request.method} {request.url.path}")
    print(f"Headers: {dict(request.headers)}")
    response = await call_next(request)
    print(f"Response: {response.status_code}")
    print(f"{'='*80}\n")
    return response


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key from header."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    # In mock mode, accept any non-empty key
    return x_api_key


# ============================================================================
# Helper Functions
# ============================================================================

def create_future(result_type: str, data: Any, delay_seconds: float = 0.001) -> Dict[str, Any]:
    """Create a future object that will resolve after a delay."""
    future_id = str(uuid.uuid4())

    # Store the future with completion status
    futures_store[future_id] = {
        "request_id": future_id,
        "status": "pending",
        "result": data,
        "result_type": result_type,
        "created_at": time.time(),
        "complete_at": time.time() + delay_seconds,
    }

    return {"request_id": future_id}


async def process_future(future_id: str):
    """Simulate async processing of a future."""
    if future_id not in futures_store:
        return

    future_data = futures_store[future_id]
    wait_time = future_data["complete_at"] - time.time()

    if wait_time > 0:
        await asyncio.sleep(wait_time)

    futures_store[future_id]["status"] = "completed"


# ============================================================================
# Service Endpoints
# ============================================================================

@app.get("/api/v1/healthz")
async def health_check():
    """Check if the API server is ready."""
    return {"status": "healthy", "message": "Mock Tinker API is running"}


@app.get("/api/v1/get_server_capabilities")
async def get_server_capabilities(x_api_key: str = Header(None)):
    """Retrieve information about supported models and server capabilities."""
    verify_api_key(x_api_key)

    return {
        "supported_models": [
            {
                "model_name": "Qwen/Qwen2.5-7B",
                "max_context_length": 32768,
                "supports_lora": True,
            },
            {
                "model_name": "meta-llama/Llama-3.1-8B",
                "max_context_length": 8192,
                "supports_lora": True,
            }
        ]
    }


# ============================================================================
# Models Endpoints
# ============================================================================

@app.post("/api/v1/create_model")
async def create_model(
    request: CreateModelRequest,
    x_api_key: str = Header(None),
    x_idempotency_key: Optional[str] = Header(None)
):
    """Create a new model with optional LoRA configuration."""
    verify_api_key(x_api_key)

    model_id = f"model_{uuid.uuid4().hex[:8]}"

    models_store[model_id] = {
        "model_id": model_id,
        "base_model": request.base_model,
        "lora_config": request.lora_config.dict() if request.lora_config else None,
        "created_at": time.time(),
    }

    result = {
        "model_id": model_id,
        "message": f"Model created successfully with base model {request.base_model}",
    }

    # Return a future that completes quickly
    return create_future("CreateModelResponse", result, delay_seconds=0.0)


@app.post("/api/v1/get_info")
async def get_info(
    request: GetInfoRequest,
    x_api_key: str = Header(None),
):
    """Retrieve information about the current model."""
    verify_api_key(x_api_key)

    if request.model_id not in models_store:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

    model_data = models_store[request.model_id]

    return {
        "model_data": {
            "model_id": model_data["model_id"],
            "model_name": model_data["base_model"],
            "lora_config": model_data.get("lora_config"),
            "created_at": model_data["created_at"],
        }
    }


@app.post("/api/v1/unload_model")
async def unload_model(
    request: UnloadModelRequest,
    x_api_key: str = Header(None),
    x_idempotency_key: Optional[str] = Header(None)
):
    """Unload the model weights and end the user's session."""
    verify_api_key(x_api_key)

    if request.model_id in models_store:
        del models_store[request.model_id]

    result = {"message": "Model unloaded successfully"}
    return create_future("UnloadModelResponse", result, delay_seconds=0.01)


# ============================================================================
# Training Endpoints
# ============================================================================

@app.post("/api/v1/forward")
async def forward(
    request: ForwardRequest,
    x_api_key: str = Header(None),
    x_idempotency_key: Optional[str] = Header(None)
):
    """Perform a forward pass through the model."""
    verify_api_key(x_api_key)

    # Mock forward pass result
    result = {
        "loss_fn_outputs": [
            {
                "loss": 2.5 + i * 0.1,
                "logprobs": {
                    "data": [0.1 * j for j in range(10)],
                    "shape": [10],
                    "dtype": "float32",
                }
            }
            for i, _ in enumerate(request.forward_input.data)
        ],
        "metrics": {
            "total_loss": 2.5,
            "num_tokens": len(request.forward_input.data) * 10,
        }
    }

    return create_future("ForwardBackwardOutput", result, delay_seconds=0.01)


@app.post("/api/v1/forward_backward")
async def forward_backward(
    request: ForwardBackwardRequest,
    x_api_key: str = Header(None),
    x_idempotency_key: Optional[str] = Header(None)
):
    """Perform a forward and backward pass through the model."""
    verify_api_key(x_api_key)

    # Mock forward-backward result
    result = {
        "loss_fn_output_type": "cross_entropy",
        "loss_fn_outputs": [
            {
                "loss": {
                    "data": [2.3 + i * 0.1],
                    "shape": [1],
                    "dtype": "float32",
                },
                "logprobs": {
                    "data": [0.1 * j for j in range(10)],
                    "shape": [10],
                    "dtype": "float32",
                }
            }
            for i, _ in enumerate(request.forward_backward_input.data)
        ],
        "metrics": {
            "total_loss:sum": 2.3,
            "num_tokens:sum": len(request.forward_backward_input.data) * 10,
            "grad_norm:mean": 0.5,
        }
    }

    return create_future("ForwardBackwardOutput", result, delay_seconds=0.0)


@app.post("/api/v1/optim_step")
async def optim_step(
    request: OptimStepRequest,
    x_api_key: str = Header(None),
    x_idempotency_key: Optional[str] = Header(None)
):
    """Perform an optimization step using AdamW optimizer."""
    verify_api_key(x_api_key)

    result = {
        "step": 1,
        "learning_rate": request.adam_params.learning_rate,
        "message": "Optimization step completed successfully",
    }

    return create_future("OptimStepResponse", result, delay_seconds=0.0)


# ============================================================================
# Sampling Endpoints
# ============================================================================

@app.post("/api/v1/sample")
async def sample(
    request: SampleRequest,
    x_api_key: str = Header(None),
    x_idempotency_key: Optional[str] = Header(None)
):
    """Synchronous sampling endpoint."""
    verify_api_key(x_api_key)

    # Mock sampling result
    result = {
        "sequences": [
            {
                "stop_reason": "length",
                "tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "logprobs": [-0.1, -0.2, -0.15, -0.3, -0.25, -0.2, -0.1, -0.15, -0.2, -0.1],
            }
            for i in range(request.num_samples)
        ],
        "prompt_logprobs": [-0.1, -0.2, -0.15] if request.prompt_logprobs else None,
    }

    return result


@app.post("/api/v1/asample")
async def asample(
    request: SampleRequest,
    x_api_key: str = Header(None),
    x_idempotency_key: Optional[str] = Header(None),
    x_tinker_sampling_backpressure: Optional[str] = Header(None)
):
    """Asynchronous sampling endpoint."""
    verify_api_key(x_api_key)

    # Mock sampling result
    result = {
        "sequences": [
            {
                "stop_reason": "length",
                "tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "logprobs": [-0.1, -0.2, -0.15, -0.3, -0.25, -0.2, -0.1, -0.15, -0.2, -0.1],
            }
            for i in range(request.num_samples)
        ],
        "prompt_logprobs": [-0.1, -0.2, -0.15] if request.prompt_logprobs else None,
    }

    return create_future("SampleResponse", result, delay_seconds=0.0)


# ============================================================================
# Weights Endpoints
# ============================================================================

@app.post("/api/v1/load_weights")
async def load_weights(
    request: LoadWeightsRequest,
    x_api_key: str = Header(None),
    x_idempotency_key: Optional[str] = Header(None)
):
    """Load model weights from disk."""
    verify_api_key(x_api_key)

    result = {
        "message": f"Weights loaded from {request.path}",
        "path": request.path,
    }

    return create_future("LoadWeightsResponse", result, delay_seconds=0.01)


@app.post("/api/v1/save_weights")
async def save_weights(
    request: SaveWeightsRequest,
    x_api_key: str = Header(None),
    x_idempotency_key: Optional[str] = Header(None)
):
    """Save the current model weights to disk."""
    verify_api_key(x_api_key)

    path = request.path or f"tinker://{request.model_id}/checkpoint_{int(time.time())}"

    result = {
        "message": "Weights saved successfully",
        "path": path,
    }

    return create_future("SaveWeightsResponse", result, delay_seconds=0.01)


@app.post("/api/v1/save_weights_for_sampler")
async def save_weights_for_sampler(
    request: SaveWeightsForSamplerRequest,
    x_api_key: str = Header(None),
    x_idempotency_key: Optional[str] = Header(None)
):
    """Save weights in a format compatible with sampling/inference servers."""
    verify_api_key(x_api_key)

    path = request.path or f"tinker://{request.model_id}/sampler_checkpoint_{int(time.time())}"

    result = {
        "message": "Sampler weights saved successfully",
        "path": path,
    }

    return create_future("SaveWeightsForSamplerResponse", result, delay_seconds=0.01)


@app.get("/api/v1/models/{model_id}/checkpoints")
async def list_checkpoints(
    model_id: str,
    x_api_key: str = Header(None)
):
    """List available model checkpoints."""
    verify_api_key(x_api_key)

    return {
        "checkpoints": [
            {
                "checkpoint_id": f"ckpt_{i}",
                "path": f"tinker://{model_id}/checkpoint_{i}",
                "created_at": time.time() - (i * 3600),
                "type": "training" if i % 2 == 0 else "sampler",
            }
            for i in range(5)
        ]
    }


@app.delete("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}")
async def delete_checkpoint(
    model_id: str,
    checkpoint_id: str,
    x_api_key: str = Header(None)
):
    """Delete a checkpoint for the given training run."""
    verify_api_key(x_api_key)

    return {"message": f"Checkpoint {checkpoint_id} deleted successfully"}


# ============================================================================
# Futures Endpoint
# ============================================================================

@app.post("/api/v1/retrieve_future")
async def retrieve_future(
    request: FutureRetrieveRequest,
    x_api_key: str = Header(None)
):
    """Retrieve the result of an async operation."""
    verify_api_key(x_api_key)

    if request.request_id not in futures_store:
        raise HTTPException(status_code=404, detail="Future not found")

    future_data = futures_store[request.request_id]

    # Check if future is ready
    if time.time() < future_data["complete_at"]:
        # Still pending - return try_again response so SDK will poll again
        return {
            "request_id": request.request_id,
            "type": "try_again"
        }

    # Future is ready - return result directly (not wrapped)
    return future_data["result"]


# ============================================================================
# Telemetry Endpoint (optional)
# ============================================================================

@app.post("/api/v1/telemetry")
async def send_telemetry(
    payload: Dict[str, Any],
    x_api_key: str = Header(None)
):
    """Receive telemetry events."""
    verify_api_key(x_api_key)
    print(f"[TELEMETRY] Received: {payload}")
    return {"status": "accepted"}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*80)
    print("🚀 Starting Mock Tinker API Server")
    print("="*80)
    print("\nServer will be available at: http://localhost:8000")
    print("\nTo use this mock server with the Tinker SDK, set:")
    print("  export TINKER_BASE_URL=http://localhost:8000")
    print("  export TINKER_API_KEY=test-key")
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("="*80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
