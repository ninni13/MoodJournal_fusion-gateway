import os
import asyncio
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv

load_dotenv()

TEXT_API_BASE   = os.getenv("TEXT_API_BASE", "").rstrip("/")
AUDIO_API_BASE  = os.getenv("AUDIO_API_BASE", "").rstrip("/")

TEXT_API_AUTH_HEADER  = os.getenv("TEXT_API_AUTH_HEADER", "")
TEXT_API_AUTH_VALUE   = os.getenv("TEXT_API_AUTH_VALUE", "")
AUDIO_API_AUTH_HEADER = os.getenv("AUDIO_API_AUTH_HEADER", "")
AUDIO_API_AUTH_VALUE  = os.getenv("AUDIO_API_AUTH_VALUE", "")

DEFAULT_ALPHA = float(os.getenv("DEFAULT_ALPHA", "0.5"))

# CORS
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

LABELS: List[str] = ["pos", "neu", "neg"]  # 務必與兩個模型一致

def normalize(probs: Optional[Dict[str, float]]) -> Dict[str, float]:
    probs = probs or {}
    total = float(sum(probs.get(k, 0.0) for k in LABELS))
    if total <= 0:
        return {"pos": 0.0, "neu": 1.0, "neg": 0.0}
    return {k: float(probs.get(k, 0.0)) / total for k in LABELS}

def fuse(text_probs: Dict[str, float], audio_probs: Dict[str, float], alpha: float) -> Dict[str, float]:
    t = normalize(text_probs)
    a = normalize(audio_probs)
    f = {k: alpha * t[k] + (1 - alpha) * a[k] for k in LABELS}
    return normalize(f)

def top1(probs: Dict[str, float]) -> str:
    return max(LABELS, key=lambda k: probs.get(k, 0.0))

# === 請確認你兩個後端的介面 ===
# 假設：
#   POST {TEXT_API_BASE}/predict-text   body: {"text": "..."}  -> {"probs": {"pos":0.6,"neu":0.3,"neg":0.1}}
#   POST {AUDIO_API_BASE}/predict-audio form-data: file -> {"probs": {"pos":0.4,"neu":0.4,"neg":0.2}}
# 若不同，請在下方 call_* 函式改成你的實際格式。

async def call_text_api(client: httpx.AsyncClient, text: str) -> Dict[str, float]:
    if not TEXT_API_BASE:
        return {}

    headers: Dict[str, str] = {}
    if TEXT_API_AUTH_HEADER and TEXT_API_AUTH_VALUE:
        headers[TEXT_API_AUTH_HEADER] = TEXT_API_AUTH_VALUE

    url = f"{TEXT_API_BASE}/infer"
    try:
        r = await client.post(url, json={"text": text}, headers=headers, timeout=20)
        if r.status_code == 200:
            j = r.json()
            return j.get("probs", {})
    except Exception:
        return {}
    return {}

async def call_audio_api(client: httpx.AsyncClient, file: Optional[UploadFile]) -> Dict[str, float]:
    if not AUDIO_API_BASE or file is None:
        return {}

    headers: Dict[str, str] = {}
    if AUDIO_API_AUTH_HEADER and AUDIO_API_AUTH_VALUE:
        headers[AUDIO_API_AUTH_HEADER] = AUDIO_API_AUTH_VALUE

    content = await file.read()
    if not content:
        print('[fusion] audio empty')
        return {}

    url = f"{AUDIO_API_BASE}/infer"
    files = {"file": (file.filename or "audio.webm", content, file.content_type or "audio/webm")}
    try:
        r = await client.post(url, files=files, headers=headers, timeout=40)
        if r.status_code == 200:
            j = r.json()
            return j.get("probs", {})
        print('[fusion] audio infer non-200:', r.status_code, str(r.text)[:200])
    except Exception as exc:
        print('[fusion] audio infer error:', repr(exc))
    return {}

# === FastAPI App ===
app = FastAPI(title="Fusion Gateway", version="1.0.0")

if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

class FusionResponse(BaseModel):
    text_pred: Dict[str, float] = Field(..., example={"pos":0.6,"neu":0.3,"neg":0.1})
    audio_pred: Dict[str, float] = Field(..., example={"pos":0.4,"neu":0.4,"neg":0.2})
    fusion_pred: Dict[str, float] = Field(..., example={"pos":0.55,"neu":0.33,"neg":0.12})
    text_top1: str = "pos"
    audio_top1: str = "neu"
    fusion_top1: str = "pos"
    alpha: float = 0.5
    labels: List[str] = LABELS

@app.get("/healthz")
async def healthz():
    ok = bool(TEXT_API_BASE) and bool(AUDIO_API_BASE)
    return {"ok": ok, "text_api": TEXT_API_BASE, "audio_api": AUDIO_API_BASE}

@app.post("/predict-fusion", response_model=FusionResponse)
async def predict_fusion(
    text: str = Form(""),
    file: Optional[UploadFile] = File(None),
    alpha: float = Form(DEFAULT_ALPHA),
):
    try:
        alpha = max(0.0, min(1.0, float(alpha)))
    except Exception:
        alpha = DEFAULT_ALPHA

    print('[fusion] text_len:', len(text or ''))
    print('[fusion] file:', None if file is None else (file.filename, file.content_type))

    async with httpx.AsyncClient() as client:
        text_task = call_text_api(client, text or "")
        audio_task = call_audio_api(client, file) if file is not None else asyncio.sleep(0, result={})
        text_probs, audio_probs = await asyncio.gather(text_task, audio_task)

    if not text_probs and not audio_probs:
        text_probs = {"pos": 0.0, "neu": 1.0, "neg": 0.0}
        audio_probs = {}
        alpha = 1.0

    if not audio_probs:
        alpha = 1.0
    if not text_probs:
        alpha = 0.0

    text_probs_n = normalize(text_probs)
    audio_probs_n = normalize(audio_probs if audio_probs else None)

    fusion_probs = {
        k: alpha * text_probs_n[k] + (1 - alpha) * audio_probs_n[k]
        for k in ('pos', 'neu', 'neg')
    }

    return FusionResponse(
        text_pred=text_probs_n,
        audio_pred=audio_probs_n if audio_probs else {'pos': 0.0, 'neu': 1.0, 'neg': 0.0},
        fusion_pred=fusion_probs,
        text_top1=top1(text_probs_n),
        audio_top1=top1(audio_probs_n),
        fusion_top1=top1(fusion_probs),
        alpha=alpha,
        labels=LABELS,
    )
