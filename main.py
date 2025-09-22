import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple
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

DEFAULT_ALPHA = float(os.getenv("DEFAULT_ALPHA", "0.75"))
BASE_THRESHOLD = float(os.getenv("FUSION_THRESHOLD", "0.65"))
TEXT_THRESHOLD = float(os.getenv("TEXT_THRESHOLD", str(BASE_THRESHOLD)))
FUSION_THRESHOLD = float(os.getenv("FUSION_FUSE_THRESHOLD", str(BASE_THRESHOLD)))
AUDIO_T_POS = float(os.getenv("AUDIO_T_POS", "0.55"))
AUDIO_T_NEG = float(os.getenv("AUDIO_T_NEG", "0.80"))
AUDIO_MIN_CONF = float(os.getenv("AUDIO_MIN_CONF", "0.60"))
AUDIO_MARGIN = float(os.getenv("AUDIO_MARGIN", "0.15"))

# CORS
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

LABELS: List[str] = ["pos", "neu", "neg"]  # 務必與兩個模型一致

def normalize(probs: Optional[Dict[str, float]]) -> Dict[str, float]:
    probs = probs or {}
    total = float(sum(probs.get(k, 0.0) for k in LABELS))
    if total <= 0:
        return {"pos": 0.0, "neu": 1.0, "neg": 0.0}
    return {k: float(probs.get(k, 0.0)) / total for k in LABELS}


def to_pos_neu_neg(probs: Optional[Dict[str, float]]) -> Dict[str, float]:
    """接受 {positive/neutral/negative} 或 {pos/neu/neg}，統一轉成 {pos, neu, neg}。"""
    if not isinstance(probs, dict):
        return {}
    if {"pos", "neu", "neg"} <= probs.keys():
        return {
            "pos": float(probs.get("pos", 0.0)),
            "neu": float(probs.get("neu", 0.0)),
            "neg": float(probs.get("neg", 0.0)),
        }
    mapping = {"positive": "pos", "neutral": "neu", "negative": "neg"}
    out = {"pos": 0.0, "neu": 0.0, "neg": 0.0}
    for key, value in probs.items():
        short = mapping.get(key)
        if short:
            try:
                out[short] = float(value)
            except (TypeError, ValueError):
                out[short] = 0.0
    return out


def apply_threshold(probs: Optional[Dict[str, float]], threshold: float = BASE_THRESHOLD):
    converted = to_pos_neu_neg(probs)
    normalized = normalize(converted)
    top = max(normalized, key=normalized.get)
    if normalized[top] < threshold:
        return 'neu', {'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
    return top, normalized


def apply_audio_class_thresholds(
    probs: Optional[Dict[str, float]],
    t_pos: float = AUDIO_T_POS,
    t_neg: float = AUDIO_T_NEG,
    min_conf: float = AUDIO_MIN_CONF,
    margin: float = AUDIO_MARGIN,
) -> Dict[str, float]:
    if not isinstance(probs, dict):
        return {'pos': 0.0, 'neu': 1.0, 'neg': 0.0}

    normalized = normalize(to_pos_neu_neg(probs))
    sorted_items = sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)
    if len(sorted_items) < 2:
        sorted_items.append(('neu', 0.0))
    top_label, top_val = sorted_items[0]
    second_label, second_val = sorted_items[1]

    # 若符合條件，才允許輸出負向/正向，否則強制回退中立
    if top_label == 'neg':
        if top_val >= t_neg and (top_val - second_val) >= margin:
            return {'pos':0.0,'neu':0.0,'neg':1.0}   # 直接 hard 判斷為負向
        else:
            return {'pos':0.0,'neu':1.0,'neg':0.0}

    if top_label == 'pos':
        if top_val >= t_pos and (top_val - second_val) >= margin:
            return {'pos':1.0,'neu':0.0,'neg':0.0}   # 直接 hard 判斷為正向
        else:
            return {'pos':0.0,'neu':1.0,'neg':0.0}

    # 其餘 → 視為中立
    return {'pos':0.0,'neu':1.0,'neg':0.0}


def is_neutral_default(probs: Optional[Dict[str, float]]) -> bool:
    if not probs or not isinstance(probs, dict):
        return True
    total = float(probs.get("pos", 0.0) + probs.get("neu", 0.0) + probs.get("neg", 0.0))
    if total <= 0:
        return True
    return (
        abs(probs.get("pos", 0.0)) < 1e-9
        and abs(probs.get("neg", 0.0)) < 1e-9
        and abs(probs.get("neu", 1.0) - 1.0) < 1e-9
    )

def fuse(text_probs: Dict[str, float], audio_probs: Dict[str, float], alpha: float) -> Dict[str, float]:
    t = normalize(text_probs)
    a = normalize(audio_probs)
    f = {k: alpha * t[k] + (1 - alpha) * a[k] for k in LABELS}
    return normalize(f)

def fusion_with_threshold(text_probs: Dict[str, float], audio_probs: Dict[str, float], alpha: float, threshold: float = FUSION_THRESHOLD):
    fused = {k: alpha * text_probs.get(k, 0.0) + (1 - alpha) * audio_probs.get(k, 0.0) for k in LABELS}
    top = max(fused, key=fused.get)
    if fused[top] < threshold:
        return 'neu', fused
    return top, fused

def top1(probs: Dict[str, float]) -> str:
    return max(LABELS, key=lambda k: probs.get(k, 0.0))

# === 請確認你兩個後端的介面 ===
# 假設：
#   POST {TEXT_API_BASE}/predict-text   body: {"text": "..."}  -> {"probs": {"pos":0.6,"neu":0.3,"neg":0.1}}
#   POST {AUDIO_API_BASE}/predict-audio form-data: file -> {"probs": {"pos":0.4,"neu":0.4,"neg":0.2}}
# 若不同，請在下方 call_* 函式改成你的實際格式。

async def call_text_api(client: httpx.AsyncClient, text: str) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    if not TEXT_API_BASE:
        return ({'pos': 0.0, 'neu': 1.0, 'neg': 0.0}, [])

    headers: Dict[str, str] = {}
    if TEXT_API_AUTH_HEADER and TEXT_API_AUTH_VALUE:
        headers[TEXT_API_AUTH_HEADER] = TEXT_API_AUTH_VALUE

    url = f"{TEXT_API_BASE}/infer"
    try:
        r = await client.post(url, json={"text": text}, headers=headers, timeout=20)
        if r.status_code != 200:
            print('[fusion] text infer non-200:', r.status_code)
            return ({'pos': 0.0, 'neu': 1.0, 'neg': 0.0}, [])
        j = r.json()
        raw = j.get('probs') or j
        _, filtered = apply_threshold(raw, TEXT_THRESHOLD)
        tokens = j.get('top_tokens') or j.get('tokens') or []
        if isinstance(tokens, list):
            sanitized_tokens = []
            for item in tokens:
                if isinstance(item, dict):
                    sanitized_tokens.append(item)
                else:
                    sanitized_tokens.append({'text': str(item)})
            sanitized_tokens = sanitized_tokens[:5]
        else:
            sanitized_tokens = []
        return (filtered, sanitized_tokens)
    except Exception as exc:
        print('[fusion] text infer error:', repr(exc))
        return ({'pos': 0.0, 'neu': 1.0, 'neg': 0.0}, [])
    return ({'pos': 0.0, 'neu': 1.0, 'neg': 0.0}, [])

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
    files = {"file": (file.filename or "note.webm", content, file.content_type or "audio/webm")}
    try:
        r = await client.post(url, files=files, headers=headers, timeout=40)
        if r.status_code != 200:
            print('[fusion] audio infer non-200:', r.status_code, str(r.text)[:200])
            return {}
        j = r.json()
        raw = j.get('probs') or j
        return apply_audio_class_thresholds(raw)
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
    text_top_tokens: List[Dict[str, Any]] = Field(default_factory=list)

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
        (text_probs, text_tokens), audio_probs = await asyncio.gather(text_task, audio_task)
    text_tokens = text_tokens or []

    if not text_probs and not audio_probs:
        text_probs = {"pos": 0.0, "neu": 1.0, "neg": 0.0}
        audio_probs = {}
        alpha = 1.0
        text_tokens = []

    audio_is_neutral = is_neutral_default(audio_probs)
    text_is_neutral = is_neutral_default(text_probs)

    if audio_is_neutral:
        audio_probs = {}
        alpha = 1.0
    elif text_is_neutral:
        alpha = 0.0

    text_probs_n = normalize(text_probs)
    audio_probs_n = normalize(audio_probs if audio_probs else None)

    fusion_top, fusion_probs = fusion_with_threshold(text_probs_n, audio_probs_n, alpha, FUSION_THRESHOLD)

    return FusionResponse(
        text_pred=text_probs_n,
        audio_pred=audio_probs_n if audio_probs else {'pos': 0.0, 'neu': 1.0, 'neg': 0.0},
        fusion_pred=fusion_probs,
        text_top1=top1(text_probs_n),
        audio_top1=top1(audio_probs_n),
        fusion_top1=fusion_top,
        alpha=alpha,
        labels=LABELS,
        text_top_tokens=text_tokens,
    )
