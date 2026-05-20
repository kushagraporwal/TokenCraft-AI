"""
FastAPI server for LLM text generation.

Run from project root:
    uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import GPT_CONFIG_124M, GPT_CONFIG_124M_TRAIN
from tokenizer import BPETokenizer
from transformer import GPTModel
from training import generate_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "model.pth"


def _resolve_device() -> str:
    requested = os.getenv("LLM_DEVICE", "auto").lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested in ("auto", "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(checkpoint_path: Path, device: str) -> tuple[GPTModel, dict]:
    cfg = dict(GPT_CONFIG_124M)
    if checkpoint_path.is_file():
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            cfg = checkpoint.get("config", GPT_CONFIG_124M_TRAIN)
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            cfg = GPT_CONFIG_124M_TRAIN
        model = GPTModel(cfg)
        model.load_state_dict(state_dict)
    else:
        model = GPTModel(cfg)
    model.to(device)
    model.eval()
    return model, cfg


class AppState:
    model: Optional[GPTModel] = None
    tokenizer: Optional[BPETokenizer] = None
    device: str = "cpu"
    checkpoint_loaded: bool = False
    checkpoint_path: str = ""
    model_config: dict = {}


state = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.device = _resolve_device()
    checkpoint = Path(os.getenv("LLM_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    state.checkpoint_path = str(checkpoint)
    state.tokenizer = BPETokenizer()
    state.model, state.model_config = _load_model(checkpoint, state.device)
    state.checkpoint_loaded = checkpoint.is_file()
    yield


app = FastAPI(title="LLM Model API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_new_tokens: int = Field(50, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(50, ge=1, le=500)


class GenerateResponse(BaseModel):
    output: str
    checkpoint_loaded: bool
    device: str


class HealthResponse(BaseModel):
    status: str
    checkpoint_loaded: bool
    checkpoint_path: str
    device: str


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        checkpoint_loaded=state.checkpoint_loaded,
        checkpoint_path=state.checkpoint_path,
        device=state.device,
    )


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        output = generate_text(
            state.model,
            state.tokenizer,
            prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            device=state.device,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenerateResponse(
        output=output,
        checkpoint_loaded=state.checkpoint_loaded,
        device=state.device,
    )
