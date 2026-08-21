"""
Audio Translator API
=====================
Pipeline: Audio -> Speech-to-Text -> Subtitle (source) -> Translate ->
          Subtitle (target) -> Text-to-Speech -> Translated Audio

Models (chosen to run well on a 4GB VRAM GPU like the RTX 2050):
    - STT:         faster-whisper "small"           (~1 GB VRAM, fp16)
    - Translation: facebook/nllb-200-distilled-600M  (~1.2 GB VRAM, fp16)
    - TTS:         facebook/mms-tts-<lang>           (~150 MB each, lazy loaded per language)

----------------------------------------------------------------------
INSTALL:
    pip install fastapi uvicorn python-multipart
    pip install faster-whisper
    pip install transformers accelerate sentencepiece
    pip install torch --index-url https://download.pytorch.org/whl/cu121   # or your CUDA version
    pip install soundfile numpy

RUN:
    uvicorn main:app --reload --port 8000

The server auto-detects CUDA. If no GPU is found it falls back to CPU
(slow, but functional) automatically.
----------------------------------------------------------------------
"""

import base64
import logging
import os
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ============================================================
# LOGGING / DEBUGGING SETUP
# ============================================================
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("audio-translator")

# quiet down noisy third-party loggers, we log our own step summaries instead
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def log_step(request_id: str, step: str, msg: str, level: int = logging.INFO) -> None:
    logger.log(level, f"[{request_id}] {step}: {msg}")


class Timer:
    """Context manager that times a pipeline step and logs start/end/failure."""

    def __init__(self, request_id: str, step_name: str):
        self.request_id = request_id
        self.step_name = step_name

    def __enter__(self):
        self.start = time.time()
        log_step(self.request_id, self.step_name, "started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if exc_type:
            log_step(
                self.request_id,
                self.step_name,
                f"FAILED after {elapsed:.2f}s -> {exc_type.__name__}: {exc_val}",
                logging.ERROR,
            )
        else:
            log_step(self.request_id, self.step_name, f"completed in {elapsed:.2f}s")
        return False  # never swallow exceptions


# ============================================================
# LANGUAGE CONFIG
# nllb  = FLORES-200 code used by the NLLB translation model
# mms   = language code used by facebook/mms-tts-<mms> checkpoints
# NOTE: not every language has an MMS TTS checkpoint on the Hub.
#       If one is missing for your target language, swap the "mms"
#       value for the correct code from:
#       https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html
# ============================================================
LANGUAGES: Dict[str, Dict[str, str]] = {
    "en": {"name": "English", "nllb": "eng_Latn", "mms": "eng"},
    "hi": {"name": "Hindi", "nllb": "hin_Deva", "mms": "hin"},
    "es": {"name": "Spanish", "nllb": "spa_Latn", "mms": "spa"},
    "fr": {"name": "French", "nllb": "fra_Latn", "mms": "fra"},
    "de": {"name": "German", "nllb": "deu_Latn", "mms": "deu"},
    "zh": {"name": "Chinese (Simplified)", "nllb": "zho_Hans", "mms": "cmn"},
    "ja": {"name": "Japanese", "nllb": "jpn_Jpan", "mms": "jpn"},
    "ko": {"name": "Korean", "nllb": "kor_Hang", "mms": "kor"},
    "ru": {"name": "Russian", "nllb": "rus_Cyrl", "mms": "rus"},
    "ar": {"name": "Arabic", "nllb": "arb_Arab", "mms": "ara"},
    "pt": {"name": "Portuguese", "nllb": "por_Latn", "mms": "por"},
    "it": {"name": "Italian", "nllb": "ita_Latn", "mms": "ita"},
    "bn": {"name": "Bengali", "nllb": "ben_Beng", "mms": "ben"},
    "ur": {"name": "Urdu", "nllb": "urd_Arab", "mms": "urd"},
    "ta": {"name": "Tamil", "nllb": "tam_Taml", "mms": "tam"},
    "te": {"name": "Telugu", "nllb": "tel_Telu", "mms": "tel"},
    "tr": {"name": "Turkish", "nllb": "tur_Latn", "mms": "tur"},
    "vi": {"name": "Vietnamese", "nllb": "vie_Latn", "mms": "vie"},
    "nl": {"name": "Dutch", "nllb": "nld_Latn", "mms": "nld"},
    "pl": {"name": "Polish", "nllb": "pol_Latn", "mms": "pol"},
}

WHISPER_TO_NLLB = {k: v["nllb"] for k, v in LANGUAGES.items()}


def whisper_lang_to_nllb(code: str) -> str:
    """Map a whisper-detected 2-letter language code to an NLLB FLORES code."""
    return WHISPER_TO_NLLB.get(code, "eng_Latn")


# ============================================================
# MODEL HOLDERS (loaded once at startup, TTS lazily per language)
# ============================================================
class Models:
    device: str = "cpu"
    whisper = None
    nllb_tokenizer = None
    nllb_model = None
    tts_cache: Dict[str, Dict] = {}


models = Models()


def init_device() -> None:
    if torch.cuda.is_available():
        models.device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        models.device = "cpu"
        logger.warning("No CUDA GPU detected -> falling back to CPU (will be noticeably slower)")


def load_stt_model() -> None:
    from faster_whisper import WhisperModel

    compute_type = "float16" if models.device == "cuda" else "int8"
    logger.info(f"Loading faster-whisper 'small' (device={models.device}, compute_type={compute_type}) ...")
    models.whisper = WhisperModel("small", device=models.device, compute_type=compute_type)
    logger.info("STT model ready")


def load_translation_model() -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "facebook/nllb-200-distilled-600M"
    logger.info(f"Loading translation model '{model_name}' ...")
    models.nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if models.device == "cuda" else torch.float32
    models.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(models.device)
    models.nllb_model.eval()
    logger.info("Translation model ready")


def get_tts_model(lang_code: str) -> Dict:
    """Lazily load & cache an MMS-TTS model for a given target language."""
    if lang_code in models.tts_cache:
        return models.tts_cache[lang_code]

    from transformers import AutoTokenizer, VitsModel

    mms_code = LANGUAGES[lang_code]["mms"]
    model_name = f"facebook/mms-tts-{mms_code}"
    logger.info(f"Loading TTS model for '{lang_code}' -> {model_name} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = VitsModel.from_pretrained(model_name).to(models.device)
        model.eval()
    except Exception as e:
        logger.error(f"Could not load TTS model '{model_name}': {e}")
        raise RuntimeError(
            f"No TTS checkpoint available for language '{lang_code}' ({model_name}). "
            "Pick a different target language or update the 'mms' code in LANGUAGES."
        )

    models.tts_cache[lang_code] = {"model": model, "tokenizer": tokenizer}
    logger.info(f"TTS model for '{lang_code}' ready (sampling_rate={model.config.sampling_rate})")
    return models.tts_cache[lang_code]


# ============================================================
# APP LIFESPAN
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Starting Audio Translator API ===")
    init_device()
    try:
        load_stt_model()
        load_translation_model()
    except Exception:
        logger.error("Fatal error during model startup:\n" + traceback.format_exc())
        raise
    logger.info("=== Startup complete — ready to serve requests ===")
    yield
    logger.info("=== Shutting down ===")


app = FastAPI(title="Audio Translator API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================
def format_timestamp_srt(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segments: List[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        text = seg["text"].strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def transcribe_audio(request_id: str, audio_path: str):
    with Timer(request_id, "STEP 1/3 - Speech-to-Text"):
        segments_gen, info = models.whisper.transcribe(audio_path, beam_size=5, vad_filter=True)
        segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_gen]
        log_step(
            request_id,
            "STT",
            f"detected_language={info.language} confidence={info.language_probability:.2f} segments={len(segments)}",
        )
        if not segments:
            raise ValueError("No speech detected in the uploaded audio")
        return segments, info.language


def translate_segments(request_id: str, segments: List[dict], src_nllb: str, tgt_nllb: str):
    with Timer(request_id, "STEP 2/3 - Translation"):
        tokenizer = models.nllb_tokenizer
        model = models.nllb_model
        tokenizer.src_lang = src_nllb

        try:
            forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_nllb)
        except Exception:
            forced_bos_id = tokenizer.lang_code_to_id[tgt_nllb]  # older transformers versions

        translated_segments = []
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                translated_segments.append({**seg, "text": ""})
                continue
            inputs = tokenizer(text, return_tensors="pt").to(models.device)
            with torch.no_grad():
                generated = model.generate(**inputs, forced_bos_token_id=forced_bos_id, max_new_tokens=256)
            translated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            translated_segments.append({"start": seg["start"], "end": seg["end"], "text": translated_text})

        log_step(request_id, "Translation", f"translated {len(translated_segments)} segments")
        return translated_segments


def synthesize_speech(request_id: str, segments: List[dict], target_lang_code: str):
    with Timer(request_id, "STEP 3/3 - Text-to-Speech"):
        tts = get_tts_model(target_lang_code)
        model, tokenizer = tts["model"], tts["tokenizer"]
        sampling_rate = model.config.sampling_rate
        silence_gap = np.zeros(int(sampling_rate * 0.3), dtype=np.float32)  # 300ms between segments

        chunks = []
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            inputs = tokenizer(text, return_tensors="pt").to(models.device)
            with torch.no_grad():
                waveform = model(**inputs).waveform
            chunks.append(waveform.squeeze().cpu().float().numpy())
            chunks.append(silence_gap)

        if not chunks:
            raise ValueError("No translated text available to synthesize into speech")

        full_audio = np.concatenate(chunks)
        log_step(request_id, "TTS", f"generated {len(full_audio) / sampling_rate:.2f}s of audio @ {sampling_rate}Hz")
        return full_audio, sampling_rate


# ============================================================
# ROUTES
# ============================================================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": models.device,
        "cuda_available": torch.cuda.is_available(),
        "stt_loaded": models.whisper is not None,
        "translation_loaded": models.nllb_model is not None,
        "tts_loaded_languages": list(models.tts_cache.keys()),
    }


@app.get("/api/languages")
async def get_languages():
    return {code: info["name"] for code, info in LANGUAGES.items()}


@app.post("/api/process")
async def process_audio(audio: UploadFile = File(...), target_lang: str = Form(...)):
    request_id = str(uuid.uuid4())[:8]
    log_step(request_id, "REQUEST", f"file='{audio.filename}' target_lang='{target_lang}'")

    if target_lang not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported target_lang '{target_lang}'. See /api/languages")

    tmp_path = None
    try:
        # ---- persist upload to a temp file ----
        suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
        content = await audio.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        log_step(request_id, "UPLOAD", f"saved temp file {tmp_path} ({len(content) / 1024:.1f} KB)")

        # ---- STEP 1: speech to text ----
        segments, detected_lang = transcribe_audio(request_id, tmp_path)
        original_srt = build_srt(segments)
        original_text = " ".join(s["text"].strip() for s in segments)

        # ---- STEP 2: translate ----
        src_nllb = whisper_lang_to_nllb(detected_lang)
        tgt_nllb = LANGUAGES[target_lang]["nllb"]
        if src_nllb == tgt_nllb:
            log_step(request_id, "Translation", "source == target, skipping model call", logging.WARNING)
            translated_segments = segments
        else:
            translated_segments = translate_segments(request_id, segments, src_nllb, tgt_nllb)
        translated_srt = build_srt(translated_segments)
        translated_text = " ".join(s["text"].strip() for s in translated_segments)

        # ---- STEP 3: text to speech ----
        audio_array, sr = synthesize_speech(request_id, translated_segments, target_lang)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out_tmp:
            sf.write(out_tmp.name, audio_array, sr)
            out_path = out_tmp.name
        with open(out_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        os.remove(out_path)

        log_step(request_id, "REQUEST", "completed successfully ✅")

        return JSONResponse(
            {
                "request_id": request_id,
                "detected_source_language": detected_lang,
                "target_language": target_lang,
                "original_text": original_text,
                "translated_text": translated_text,
                "original_srt": original_srt,
                "translated_srt": translated_srt,
                "audio_base64": audio_b64,
                "audio_format": "wav",
            }
        )

    except HTTPException:
        raise
    except ValueError as ve:
        log_step(request_id, "REQUEST", f"validation error: {ve}", logging.ERROR)
        raise HTTPException(status_code=422, detail=str(ve))
    except RuntimeError as re_err:
        log_step(request_id, "REQUEST", f"runtime error: {re_err}", logging.ERROR)
        raise HTTPException(status_code=500, detail=str(re_err))
    except Exception as e:
        log_step(request_id, "REQUEST", f"unexpected error: {e}\n{traceback.format_exc()}", logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            log_step(request_id, "CLEANUP", f"removed temp file {tmp_path}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)