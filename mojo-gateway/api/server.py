"""
BitNet T-MAC Inference API Server

FastAPI wrapper for the persistent Mojo BitNet inference server.
Model is loaded once at startup and kept in memory.

Fixed: Server starts immediately, model loads in background.
Health checks pass during loading phase.
"""
import os
import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/bitnet-2b.tmac2.bin")
BINARY_PATH = os.environ.get("BINARY_PATH", "/app/bin/bitnet_server")

# Global state
mojo_process = None
process_lock = asyncio.Lock()
server_status = "starting"  # "starting", "loading", "ready", "error"
status_message = "Initializing..."
startup_time = time.time()


async def start_mojo_server():
    """Start the persistent Mojo server process in background."""
    global mojo_process, server_status, status_message

    if not os.path.exists(MODEL_PATH):
        server_status = "error"
        status_message = f"Model not found: {MODEL_PATH}"
        print(f"ERROR: {status_message}")
        return

    if not os.path.exists(BINARY_PATH):
        server_status = "error"
        status_message = f"Binary not found: {BINARY_PATH}"
        print(f"ERROR: {status_message}")
        return

    server_status = "loading"
    status_message = "Loading model..."
    print(f"Starting Mojo server with model: {MODEL_PATH}")

    try:
        mojo_process = await asyncio.create_subprocess_exec(
            BINARY_PATH, MODEL_PATH, "--server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Wait for server to be ready
        while True:
            try:
                line = await asyncio.wait_for(
                    mojo_process.stdout.readline(),
                    timeout=300  # 5 minute timeout for model loading
                )
            except asyncio.TimeoutError:
                server_status = "error"
                status_message = "Timeout waiting for model to load"
                print(f"ERROR: {status_message}")
                return

            if not line:
                stderr = await mojo_process.stderr.read()
                server_status = "error"
                status_message = f"Mojo server failed: {stderr.decode()[:200]}"
                print(f"ERROR: {status_message}")
                return

            line_str = line.decode().strip()
            print(f"[Mojo] {line_str}")

            if line_str == "SERVER_READY":
                server_status = "ready"
                status_message = "Model loaded and ready"
                print("Mojo server is ready!")
                return

    except Exception as e:
        server_status = "error"
        status_message = f"Failed to start: {str(e)}"
        print(f"ERROR: {status_message}")


async def stop_mojo_server():
    """Stop the Mojo server process."""
    global mojo_process, server_status

    if mojo_process:
        try:
            mojo_process.stdin.write(b"QUIT\n")
            await mojo_process.stdin.drain()
            await asyncio.wait_for(mojo_process.wait(), timeout=5.0)
        except Exception:
            mojo_process.kill()
        mojo_process = None
        server_status = "stopped"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle - start model loading in background."""
    # Start model loading as background task (don't block startup)
    asyncio.create_task(start_mojo_server())
    yield
    await stop_mojo_server()


app = FastAPI(
    title="BitNet T-MAC Inference API",
    description="High-performance ternary LLM inference with T-MAC lookup tables",
    version="2.1.0",
    lifespan=lifespan
)


class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    num_tokens: int = 32
    temperature: float = 0.8
    top_p: float = 0.9


class GenerateResponse(BaseModel):
    tokens: list[int]
    num_tokens: int
    elapsed_seconds: float
    tokens_per_second: float


class HealthResponse(BaseModel):
    status: str
    server_status: str
    message: str
    model_path: str
    uptime_seconds: float


@app.get("/")
async def root():
    return {
        "service": "BitNet T-MAC Inference API",
        "version": "2.1.0",
        "status": server_status,
        "optimizations": ["T-MAC LUT (no multiplications)", "Persistent model"],
        "endpoints": {
            "/health": "Health check (always responds)",
            "/ready": "Readiness check (only when model loaded)",
            "/generate": "Generate tokens (POST)",
            "/v1/completions": "OpenAI-compatible completions (POST)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check - always returns 200 if server is running.
    Use /ready to check if model is loaded.
    """
    uptime = time.time() - startup_time

    # Return healthy as long as server is running (not in error state)
    is_healthy = server_status in ["starting", "loading", "ready"]

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        server_status=server_status,
        message=status_message,
        model_path=MODEL_PATH,
        uptime_seconds=round(uptime, 1)
    )


@app.get("/ready")
async def ready():
    """
    Readiness check - returns 200 only when model is loaded and ready.
    """
    if server_status == "ready":
        return {"status": "ready", "message": "Model loaded and accepting requests"}
    elif server_status == "error":
        raise HTTPException(status_code=503, detail=status_message)
    else:
        raise HTTPException(
            status_code=503,
            detail=f"Server is {server_status}: {status_message}"
        )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate tokens using persistent BitNet inference server."""
    global mojo_process

    # Check if server is ready
    if server_status != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Server not ready: {server_status} - {status_message}"
        )

    if mojo_process is None or mojo_process.returncode is not None:
        raise HTTPException(status_code=503, detail="Inference process not running")

    async with process_lock:
        try:
            # Send request: num_tokens,temperature,top_p
            request_line = f"{request.num_tokens},{request.temperature},{request.top_p}\n"
            mojo_process.stdin.write(request_line.encode())
            await mojo_process.stdin.drain()

            # Read response with timeout
            response_line = await asyncio.wait_for(
                mojo_process.stdout.readline(),
                timeout=300  # 5 minute timeout
            )

            if not response_line:
                raise HTTPException(status_code=500, detail="No response from inference server")

            # Parse response: tokens|elapsed
            response_str = response_line.decode().strip()
            parts = response_str.split("|")

            tokens_str = parts[0]
            elapsed = float(parts[1]) if len(parts) > 1 else 0.0

            tokens = [int(t) for t in tokens_str.split(",") if t]

            return GenerateResponse(
                tokens=tokens,
                num_tokens=len(tokens),
                elapsed_seconds=round(elapsed, 3),
                tokens_per_second=round(len(tokens) / elapsed, 2) if elapsed > 0 else 0
            )

        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Inference timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# OpenAI-compatible endpoint
class CompletionRequest(BaseModel):
    model: str = "bitnet-2b"
    prompt: str
    max_tokens: int = 32
    temperature: float = 0.8
    top_p: float = 0.9


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: dict


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions endpoint."""

    gen_request = GenerateRequest(
        num_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    result = await generate(gen_request)

    token_text = " ".join(str(t) for t in result.tokens)

    return CompletionResponse(
        id=f"cmpl-bitnet-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[
            CompletionChoice(
                text=f"[Token IDs: {token_text}]",
                index=0,
                finish_reason="length"
            )
        ],
        usage={
            "prompt_tokens": 0,
            "completion_tokens": result.num_tokens,
            "total_tokens": result.num_tokens
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
