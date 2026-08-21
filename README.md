# 🎙️ Audio Translator

An end-to-end audio translation pipeline: upload or record audio in any language, and get back a **transcript, a translation, subtitles for both languages, and dubbed audio in your target language** — all in one request.

Built during my internship at **Pinnacle Lab**.

## How it works

```
Audio Input
    │
    ▼
1. Speech-to-Text  (faster-whisper)
    │  → transcript + timestamps → original-language subtitles (.srt)
    ▼
2. Translation  (Meta NLLB-200, 200+ languages)
    │  → translated transcript → translated-language subtitles (.srt)
    ▼
3. Text-to-Speech  (Meta MMS-TTS)
    │  → dubbed audio in the target language
    ▼
Output: transcript, translation, 2 subtitle files, translated audio
```

## Tech Stack

| Component      | Choice                                  | Why |
|----------------|------------------------------------------|-----|
| Backend        | FastAPI (Python)                         | Fast, async, simple to extend with new endpoints |
| Speech-to-Text | `faster-whisper` (small)                 | CTranslate2-optimized Whisper — accurate, but light enough to run on a 4GB VRAM GPU |
| Translation    | `facebook/nllb-200-distilled-600M`       | Supports 200+ languages with strong translation quality at a manageable model size |
| Text-to-Speech | `facebook/mms-tts-*`                     | Small (~150MB), per-language VITS models, lazy-loaded on demand |
| Frontend       | Plain HTML/CSS/JS                        | No build step — just open the file in a browser |

Models were specifically chosen to run on modest consumer GPUs (developed and tested on an **RTX 2050, 4GB VRAM**), with automatic CPU fallback if no GPU is available.

## Features

- 🎤 Upload an audio file **or** record directly from the browser mic
- 📝 Auto-generated subtitles (`.srt`) for both the original and translated language, with timestamps
- 🌍 20+ supported target languages (easily extendable)
- 🔊 Translated audio output, playable and downloadable
- 🪵 Structured logging with per-request IDs and step timing, for easy debugging
- 🛡️ Resilient pipeline — handles low-confidence language detection and TTS edge cases (e.g. very short text segments) gracefully instead of failing the whole request

## Project Structure

```
.
├── main1.py         # FastAPI backend (STT → Translate → TTS pipeline)
├── index.html      # Frontend UI
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/sehalvarshney/audio_translator.git
cd audio-translator
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

Then install PyTorch separately, matching your hardware (see comment in `requirements.txt`):
```bash
# GPU (NVIDIA, CUDA 12.1 example — check pytorch.org for your version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch
```

### 4. Run the backend
```bash
uvicorn main:app --reload --port 8000
```
On first run, the STT and translation models will download automatically (a few GB). TTS models download lazily, the first time each target language is used.

### 5. Open the frontend
Just open `index.html` in your browser. Make sure the **API Base URL** field points to `http://localhost:8000`.

## API Endpoints

| Endpoint            | Method | Description |
|---------------------|--------|-------------|
| `/health`            | GET    | Check server/model status |
| `/api/languages`     | GET    | List supported target languages |
| `/api/process`       | POST   | Full pipeline — takes an audio file + target language, returns transcript, translation, both subtitle files, and translated audio (base64 WAV) |

## Known Limitations

- Not every language has an MMS-TTS checkpoint on Hugging Face — if one is missing, the API returns a clear error for that language.
- Translated subtitles reuse the original segment timestamps rather than the dubbed audio's own timing, since the two won't perfectly match in length.
- Very short or noisy recordings can lead to low-confidence language detection; the app surfaces a warning in this case rather than failing silently.

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Meta AI - NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [Meta AI - MMS-TTS](https://huggingface.co/facebook/mms-tts)

---

*Built as part of my internship at Pinnacle Lab.*
