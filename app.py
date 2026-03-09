#!/usr/bin/env python3
import os
import uuid
import time
import json
import threading
import subprocess
import urllib.parse
import torch

# Fix PyTorch 2.6+ weights_only default change
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from flask import Flask, request, jsonify, send_from_directory, send_file
from faster_whisper import WhisperModel

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

UPLOAD_DIR = "/data/uploads"
TTS_DIR = "/data/tts"
PRESETS_DIR = "/data/tts/presets"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TTS_DIR, exist_ok=True)
os.makedirs(PRESETS_DIR, exist_ok=True)

# ─── Whisper models ───
whisper_models = {}
whisper_lock = threading.Lock()

AVAILABLE_MODELS = [
    "tiny", "base", "small", "medium",
    "large-v1", "large-v2", "large-v3", "large-v3-turbo",
]

AVAILABLE_LANGUAGES = {
    "auto": "Авто-определение",
    "ru": "Русский",
    "en": "English",
    "uk": "Українська",
    "kk": "Қазақша",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "ar": "العربية",
    "tr": "Türkçe",
}

TTS_LANGUAGES = {
    "ru": "Русский",
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "pt": "Português",
    "pl": "Polski",
    "tr": "Türkçe",
    "nl": "Nederlands",
    "cs": "Čeština",
    "ar": "العربية",
    "zh-cn": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "hu": "Magyar",
    "hi": "हिन्दी",
}

# Global state
jobs = {}
tts_jobs = {}
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")

# LLM cleanup settings (persisted to /data/llm_settings.json)
LLM_SETTINGS_PATH = "/data/llm_settings.json"


def load_llm_settings():
    if os.path.exists(LLM_SETTINGS_PATH):
        with open(LLM_SETTINGS_PATH, "r") as f:
            return json.load(f)
    return {"provider": "groq", "api_key": ""}


def save_llm_settings_file(settings):
    with open(LLM_SETTINGS_PATH, "w") as f:
        json.dump(settings, f)


llm_settings = load_llm_settings()

# ─── TTS (XTTS v2) ───
tts_model = None
tts_lock = threading.Lock()


def get_whisper(model_size, compute_type="int8"):
    key = f"{model_size}_{compute_type}"
    if key not in whisper_models:
        with whisper_lock:
            if key not in whisper_models:
                print(f"Loading whisper: {model_size} ({compute_type})...")
                whisper_models[key] = WhisperModel(model_size, device="cpu", compute_type=compute_type)
                print(f"Whisper {model_size} ready!")
    return whisper_models[key]


def get_tts():
    """Load XTTS v2 and return the raw Xtts model for direct inference."""
    global tts_model
    if tts_model is None:
        with tts_lock:
            if tts_model is None:
                from unittest.mock import patch
                from TTS.api import TTS
                print("Loading XTTS v2 model...")
                with patch("builtins.input", return_value="y"):
                    tts_api = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
                # Extract raw Xtts model for direct inference with advanced params
                tts_model = tts_api.synthesizer.tts_model
                tts_model.eval()
                print("XTTS v2 ready (direct inference mode)!")
    return tts_model


def compute_latents(model, wav_path):
    """Compute gpt_cond_latent + speaker_embedding from a voice sample."""
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[wav_path]
    )
    return gpt_cond_latent, speaker_embedding


def save_latents(gpt_cond_latent, speaker_embedding, path):
    """Save cached latents to disk."""
    torch.save({"gpt_cond_latent": gpt_cond_latent, "speaker_embedding": speaker_embedding}, path)


def load_latents(path):
    """Load cached latents from disk."""
    data = torch.load(path, weights_only=False)
    return data["gpt_cond_latent"], data["speaker_embedding"]


# ─── Presets catalog ───

def load_presets_catalog():
    catalog_path = os.path.join(PRESETS_DIR, "catalog.json")
    if os.path.exists(catalog_path):
        with open(catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_presets_catalog(catalog):
    catalog_path = os.path.join(PRESETS_DIR, "catalog.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 100)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:02d}"


def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extract_voice_sample(input_path, output_path, start=0, duration=30):
    """Extract a clean audio clip for voice cloning."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ss", str(start), "-t", str(duration),
        "-vn", "-ar", "22050", "-ac", "1", "-acodec", "pcm_s16le",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


# ─── Diarization (diarize by FoxNoseTech) ───

diarize_fn = None
diarize_lock = threading.Lock()


def get_diarize():
    global diarize_fn
    if diarize_fn is None:
        with diarize_lock:
            if diarize_fn is None:
                from diarize import diarize as _diarize
                print("Diarization engine loaded (diarize by FoxNoseTech)")
                diarize_fn = _diarize
    return diarize_fn


def assign_speakers(segments, diarization_result):
    """Match whisper segments to diarization speaker labels."""
    dia_segs = diarization_result.segments
    for seg in segments:
        best_speaker = None
        best_overlap = 0
        for d in dia_segs:
            overlap_start = max(seg["start"], d.start)
            overlap_end = min(seg["end"], d.end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d.speaker
        seg["speaker"] = best_speaker or "?"
    return segments


def convert_to_wav(input_path, output_path):
    """Convert any audio/video to WAV for diarization."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


# ─── Transcription worker ───

def transcribe_worker(job_id, filepath):
    job = jobs[job_id]
    opts = job["options"]
    try:
        job["status"] = "loading_model"
        m = get_whisper(opts["model"], opts["compute_type"])

        job["status"] = "transcribing"

        transcribe_kwargs = {
            "beam_size": opts["beam_size"],
            "best_of": opts["best_of"],
            "temperature": opts["temperature"],
            "vad_filter": opts["vad_filter"],
            "word_timestamps": opts["word_timestamps"],
            "condition_on_previous_text": opts["condition_on_previous_text"],
        }

        if opts["language"] != "auto":
            transcribe_kwargs["language"] = opts["language"]

        if opts["vad_filter"]:
            transcribe_kwargs["vad_parameters"] = {
                "min_silence_duration_ms": opts["vad_min_silence_ms"],
                "speech_pad_ms": opts["vad_speech_pad_ms"],
                "threshold": opts["vad_threshold"],
            }

        if opts["initial_prompt"]:
            transcribe_kwargs["initial_prompt"] = opts["initial_prompt"]

        if opts["max_segment_length"] > 0:
            transcribe_kwargs["max_new_tokens"] = opts["max_segment_length"]

        segments, info = m.transcribe(filepath, **transcribe_kwargs)

        job["duration"] = info.duration
        job["language"] = info.language
        job["language_prob"] = info.language_probability

        lines = []
        for seg in segments:
            line = {
                "start": seg.start,
                "end": seg.end,
                "start_fmt": format_time(seg.start),
                "end_fmt": format_time(seg.end),
                "text": seg.text.strip(),
            }
            if opts["word_timestamps"] and seg.words:
                line["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end, "prob": round(w.probability, 2)}
                    for w in seg.words
                ]
            lines.append(line)
            job["segments"] = lines

        # Speaker diarization
        if opts.get("diarize"):
            job["status"] = "diarizing"
            try:
                diarize = get_diarize()
                wav_path = filepath + ".diarize.wav"
                convert_to_wav(filepath, wav_path)
                diarize_kwargs = {}
                if opts.get("num_speakers"):
                    diarize_kwargs["num_speakers"] = int(opts["num_speakers"])
                else:
                    if opts.get("min_speakers"):
                        diarize_kwargs["min_speakers"] = int(opts["min_speakers"])
                    if opts.get("max_speakers"):
                        diarize_kwargs["max_speakers"] = int(opts["max_speakers"])
                result = diarize(wav_path, **diarize_kwargs)
                lines = assign_speakers(lines, result)
                job["segments"] = lines
                # Collect unique speakers
                speakers = sorted(set(s.get("speaker", "?") for s in lines))
                job["speakers"] = {s: s for s in speakers}
                print(f"Diarization done: {result.num_speakers} speakers found")
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
            except Exception as e:
                print(f"Diarization failed: {e}")
                import traceback
                traceback.print_exc()

        job["status"] = "done"
        elapsed = time.time() - job["started_at"]
        job["elapsed"] = elapsed
        job["speed"] = info.duration / elapsed if elapsed > 0 else 0

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        import traceback
        traceback.print_exc()


def split_text_for_tts(text, max_chars=240):
    """Split text into chunks suitable for XTTS (max ~400 tokens ≈ 240 chars).
    Splits on sentence boundaries first, then on commas/semicolons if still too long."""
    import re
    # Split into sentences
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())

    chunks = []
    current = ""
    for sent in sentences:
        # If single sentence is too long, split further
        if len(sent) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            # Split long sentence on commas, semicolons, dashes
            parts = re.split(r'(?<=[,;:—–])\s+', sent)
            for part in parts:
                if len(current) + len(part) + 1 <= max_chars:
                    current = (current + " " + part).strip()
                else:
                    if current:
                        chunks.append(current.strip())
                    # If a single part is still too long, force-split by words
                    if len(part) > max_chars:
                        words = part.split()
                        current = ""
                        for w in words:
                            if len(current) + len(w) + 1 <= max_chars:
                                current = (current + " " + w).strip()
                            else:
                                if current:
                                    chunks.append(current.strip())
                                current = w
                    else:
                        current = part
        elif len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current.strip())
            current = sent

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text[:max_chars]]


# ─── TTS worker ───

def tts_worker(job_id):
    job = tts_jobs[job_id]
    try:
        job["status"] = "loading_model"
        model = get_tts()

        job["status"] = "synthesizing"
        output_path = os.path.join(TTS_DIR, f"{job_id}.wav")

        # Get latents: from preset cache or compute from voice sample
        gpt_cond_latent = job.get("gpt_cond_latent")
        speaker_embedding = job.get("speaker_embedding")

        if gpt_cond_latent is None or speaker_embedding is None:
            gpt_cond_latent, speaker_embedding = compute_latents(model, job["voice_sample"])

        # Synthesis parameters (with defaults)
        params = job.get("params", {})
        temperature = float(params.get("temperature", 0.75))
        speed = float(params.get("speed", 1.0))
        top_k = int(params.get("top_k", 50))
        top_p = float(params.get("top_p", 0.85))
        repetition_penalty = float(params.get("repetition_penalty", 10.0))

        # Voice tuning sliders
        style_intensity = float(params.get("style_intensity", 1.0))
        timbre_intensity = float(params.get("timbre_intensity", 1.0))
        variability = float(params.get("variability", 0.0))
        pitch_shift_steps = float(params.get("pitch_shift", 0.0))

        # Apply latent manipulation
        if style_intensity != 1.0:
            gpt_cond_latent = gpt_cond_latent * style_intensity
        if timbre_intensity != 1.0:
            speaker_embedding = speaker_embedding * timbre_intensity
        if variability > 0:
            noise = torch.randn_like(gpt_cond_latent) * variability
            gpt_cond_latent = gpt_cond_latent + noise

        # Split long text into chunks (XTTS limit ~250 chars / 400 tokens)
        import re
        import numpy as np
        import wave

        full_text = job["text"]
        chunks = split_text_for_tts(full_text, max_chars=240)
        total_chunks = len(chunks)

        wav_pieces = []
        for i, chunk in enumerate(chunks):
            job["progress"] = f"{i+1}/{total_chunks}"
            print(f"[TTS {job_id}] chunk {i+1}/{total_chunks}: {chunk[:50]}...")

            out = model.inference(
                text=chunk,
                language=job["language"],
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=temperature,
                speed=speed,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            chunk_wav = out["wav"]
            if isinstance(chunk_wav, torch.Tensor):
                chunk_wav = chunk_wav.cpu().numpy()
            wav_pieces.append(chunk_wav)

            # Small silence between chunks (0.15s)
            wav_pieces.append(np.zeros(int(24000 * 0.15), dtype=np.float32))

        # Concatenate all pieces
        wav_data = np.concatenate(wav_pieces)
        wav_data = np.clip(wav_data, -1.0, 1.0)

        # Pitch shift post-processing
        if pitch_shift_steps != 0:
            import librosa
            wav_data = librosa.effects.pitch_shift(
                y=wav_data.astype(np.float32), sr=24000, n_steps=pitch_shift_steps
            )

        wav_int16 = (wav_data * 32767).astype(np.int16)
        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(wav_int16.tobytes())

        job["output"] = output_path
        job["status"] = "done"
        elapsed = time.time() - job["started_at"]
        job["elapsed"] = elapsed

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        import traceback
        traceback.print_exc()


# ─── Routes: Static ───

@app.route("/")
def index():
    return send_from_directory("/app/static", "index.html")


@app.route("/config")
def config():
    return jsonify({
        "models": AVAILABLE_MODELS,
        "languages": AVAILABLE_LANGUAGES,
        "tts_languages": TTS_LANGUAGES,
        "default_model": DEFAULT_MODEL,
    })


# ─── Routes: Transcription ───

def parse_transcribe_options(form):
    """Parse transcription options from request form data."""
    def get_opt(name, default, type_fn=str):
        val = form.get(name)
        if val is None or val == "":
            return default
        try:
            return type_fn(val)
        except (ValueError, TypeError):
            return default

    return {
        "model": get_opt("model", DEFAULT_MODEL),
        "language": get_opt("language", "ru"),
        "beam_size": get_opt("beam_size", 5, int),
        "best_of": get_opt("best_of", 5, int),
        "temperature": list(map(float, get_opt("temperature", "0.0,0.2,0.4,0.6,0.8,1.0").split(","))),
        "compute_type": get_opt("compute_type", "int8"),
        "vad_filter": get_opt("vad_filter", "true") == "true",
        "vad_min_silence_ms": get_opt("vad_min_silence_ms", 500, int),
        "vad_speech_pad_ms": get_opt("vad_speech_pad_ms", 400, int),
        "vad_threshold": get_opt("vad_threshold", 0.5, float),
        "word_timestamps": get_opt("word_timestamps", "false") == "true",
        "condition_on_previous_text": get_opt("condition_on_previous_text", "true") == "true",
        "initial_prompt": get_opt("initial_prompt", ""),
        "max_segment_length": get_opt("max_segment_length", 0, int),
        "diarize": get_opt("diarize", "false") == "true",
        "num_speakers": get_opt("num_speakers", 0, int),
        "min_speakers": get_opt("min_speakers", 0, int),
        "max_speakers": get_opt("max_speakers", 0, int),
    }


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    job_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(f.filename)[1]
    save_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    f.save(save_path)

    options = parse_transcribe_options(request.form)

    jobs[job_id] = {
        "id": job_id,
        "filename": f.filename,
        "filepath": save_path,
        "status": "queued",
        "segments": [],
        "started_at": time.time(),
        "options": options,
    }

    t = threading.Thread(target=transcribe_worker, args=(job_id, save_path), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


# ─── URL download + transcribe ───

COOKIES_PATH = "/data/cookies.txt"


def youtube_worker(job_id, url, minutes):
    job = jobs[job_id]
    try:
        job["status"] = "downloading"
        out_template = os.path.join(UPLOAD_DIR, f"{job_id}.%(ext)s")

        # Step 1: Get title
        title_cmd = ["yt-dlp", "--js-runtimes", "node", "--no-playlist",
                      "--print", "title", "--skip-download"]
        if os.path.exists(COOKIES_PATH):
            title_cmd += ["--cookies", COOKIES_PATH]
        title_cmd.append(url)
        title_res = subprocess.run(title_cmd, capture_output=True, text=True, timeout=60)
        title = title_res.stdout.strip().split('\n')[0] if title_res.stdout.strip() else url

        # Step 2: Download audio
        cmd = [
            "yt-dlp",
            "--js-runtimes", "node",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--output", out_template,
            "--no-playlist",
        ]
        if os.path.exists(COOKIES_PATH):
            cmd += ["--cookies", COOKIES_PATH]
        if minutes and minutes > 0:
            cmd += ["--download-sections", f"*0:00-{int(minutes)}:00"]
        cmd.append(url)

        print(f"[YT {job_id}] Downloading: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(f"[YT {job_id}] stdout: {result.stdout[-200:]}")
        print(f"[YT {job_id}] stderr: {result.stderr[-200:]}")

        if result.returncode != 0:
            stderr = result.stderr
            if "Sign in to confirm" in stderr or "cookies" in stderr.lower():
                job["error"] = "YouTube требует куки. Экспортируйте cookies.txt из браузера и положите в папку data/"
            else:
                job["error"] = f"Ошибка загрузки: {stderr[:500]}"
            job["status"] = "error"
            return

        # Find the downloaded file (could be .mp3, .m4a, .opus, etc.)
        import glob
        files = glob.glob(os.path.join(UPLOAD_DIR, f"{job_id}.*"))
        if not files:
            job["status"] = "error"
            job["error"] = "Файл не скачался"
            return

        filepath = files[0]
        job["filename"] = title + os.path.splitext(filepath)[1]
        job["filepath"] = filepath

        # Hand off to existing transcription pipeline
        transcribe_worker(job_id, filepath)

    except subprocess.TimeoutExpired:
        job["status"] = "error"
        job["error"] = "Загрузка превысила 10 минут"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        import traceback
        traceback.print_exc()


@app.route("/youtube", methods=["POST"])
def youtube_download():
    url = request.form.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    minutes = 0
    try:
        minutes = float(request.form.get("minutes", 0))
    except (ValueError, TypeError):
        pass

    job_id = str(uuid.uuid4())[:8]
    options = parse_transcribe_options(request.form)

    jobs[job_id] = {
        "id": job_id,
        "filename": url,
        "filepath": None,
        "status": "downloading",
        "segments": [],
        "started_at": time.time(),
        "options": options,
    }

    t = threading.Thread(target=youtube_worker, args=(job_id, url, minutes), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404
    job = jobs[job_id]
    return jsonify({
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "segments": job.get("segments", []),
        "duration": job.get("duration"),
        "elapsed": job.get("elapsed"),
        "speed": job.get("speed"),
        "error": job.get("error"),
        "language": job.get("language"),
        "language_prob": job.get("language_prob"),
        "options": job.get("options"),
        "speakers": job.get("speakers"),
        "cleaned_text": job.get("cleaned_text"),
    })


@app.route("/download/<job_id>")
def download_file(job_id):
    fmt = request.args.get("format", "txt")
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    job = jobs[job_id]
    base = os.path.splitext(job["filename"])[0]

    speakers_map = job.get("speakers", {})

    def spk_prefix(s):
        spk = s.get("speaker_name") or speakers_map.get(s.get("speaker", ""), "") or s.get("speaker", "")
        return f"{spk}: " if spk else ""

    if fmt == "srt":
        lines = []
        for i, s in enumerate(job["segments"], 1):
            prefix = spk_prefix(s)
            lines.append(f"{i}\n{format_srt_time(s['start'])} --> {format_srt_time(s['end'])}\n{prefix}{s['text']}\n")
        text = "\n".join(lines)
        filename = f"{base}.srt"
    elif fmt == "vtt":
        lines = ["WEBVTT\n"]
        for s in job["segments"]:
            prefix = spk_prefix(s)
            lines.append(f"{format_time(s['start'])} --> {format_time(s['end'])}\n{prefix}{s['text']}\n")
        text = "\n".join(lines)
        filename = f"{base}.vtt"
    elif fmt == "json":
        text = json.dumps(job["segments"], ensure_ascii=False, indent=2)
        filename = f"{base}.json"
    else:
        text = "\n".join(
            f"[{s['start_fmt']} -> {s['end_fmt']}] {spk_prefix(s)}{s['text']}"
            for s in job["segments"]
        )
        filename = f"{base}.txt"

    return text, 200, {
        "Content-Type": "text/plain; charset=utf-8",
        "Content-Disposition": f"attachment; filename*=UTF-8''{urllib.parse.quote(filename)}",
    }


# ─── Routes: Speakers ───

@app.route("/speakers/<job_id>", methods=["POST"])
def rename_speakers(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404
    job = jobs[job_id]
    mapping = request.json  # {"SPEAKER_00": "Андрей", "SPEAKER_01": "Марина"}
    if not mapping:
        return jsonify({"error": "No mapping"}), 400
    # Update speakers map
    if not job.get("speakers"):
        job["speakers"] = {}
    job["speakers"].update(mapping)
    # Update segment labels
    for seg in job.get("segments", []):
        orig = seg.get("speaker", "")
        if orig in mapping:
            seg["speaker_name"] = mapping[orig]
    return jsonify({"ok": True, "speakers": job["speakers"]})


# ─── Routes: TTS ───

@app.route("/tts/synthesize", methods=["POST"])
def tts_synthesize():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    language = request.form.get("language", "ru")

    # Synthesis parameters
    params = {
        "temperature": request.form.get("temperature", "0.75"),
        "speed": request.form.get("speed", "1.0"),
        "top_k": request.form.get("top_k", "50"),
        "top_p": request.form.get("top_p", "0.85"),
        "repetition_penalty": request.form.get("repetition_penalty", "10.0"),
        "style_intensity": request.form.get("style_intensity", "1.0"),
        "timbre_intensity": request.form.get("timbre_intensity", "1.0"),
        "variability": request.form.get("variability", "0.0"),
        "pitch_shift": request.form.get("pitch_shift", "0.0"),
    }

    # Voice source: preset, uploaded file, or from transcription job
    voice_sample_path = None
    gpt_cond_latent = None
    speaker_embedding = None

    preset_id = request.form.get("preset_id", "").strip()
    if preset_id:
        # Use cached latents from preset
        latents_path = os.path.join(PRESETS_DIR, preset_id, "latents.pt")
        if os.path.exists(latents_path):
            gpt_cond_latent, speaker_embedding = load_latents(latents_path)
            # Still need a dummy voice_sample_path for job tracking
            voice_sample_path = os.path.join(PRESETS_DIR, preset_id, "voice.wav")
        else:
            return jsonify({"error": "Preset latents not found"}), 404

    elif "voice_sample" in request.files and request.files["voice_sample"].filename:
        f = request.files["voice_sample"]
        sample_id = str(uuid.uuid4())[:8]
        ext = os.path.splitext(f.filename)[1]
        raw_path = os.path.join(TTS_DIR, f"voice_raw_{sample_id}{ext}")
        f.save(raw_path)
        voice_sample_path = os.path.join(TTS_DIR, f"voice_{sample_id}.wav")
        try:
            extract_voice_sample(raw_path, voice_sample_path, start=0, duration=30)
        except subprocess.CalledProcessError:
            voice_sample_path = raw_path

    elif request.form.get("source_job_id"):
        source_job_id = request.form["source_job_id"]
        if source_job_id in jobs and jobs[source_job_id].get("filepath"):
            source_path = jobs[source_job_id]["filepath"]
            sample_id = str(uuid.uuid4())[:8]
            voice_sample_path = os.path.join(TTS_DIR, f"voice_{sample_id}.wav")
            start = float(request.form.get("sample_start", 0))
            duration = float(request.form.get("sample_duration", 30))
            try:
                extract_voice_sample(source_path, voice_sample_path, start=start, duration=duration)
            except subprocess.CalledProcessError as e:
                return jsonify({"error": f"Failed to extract voice: {e}"}), 500

    if not voice_sample_path:
        return jsonify({"error": "No voice sample provided"}), 400

    job_id = str(uuid.uuid4())[:8]
    job_data = {
        "id": job_id,
        "text": text,
        "language": language,
        "voice_sample": voice_sample_path,
        "params": params,
        "status": "queued",
        "started_at": time.time(),
    }
    if gpt_cond_latent is not None:
        job_data["gpt_cond_latent"] = gpt_cond_latent
        job_data["speaker_embedding"] = speaker_embedding

    tts_jobs[job_id] = job_data
    t = threading.Thread(target=tts_worker, args=(job_id,), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/tts/status/<job_id>")
def tts_status(job_id):
    if job_id not in tts_jobs:
        return jsonify({"error": "Not found"}), 404
    job = tts_jobs[job_id]
    return jsonify({
        "id": job["id"],
        "status": job["status"],
        "elapsed": job.get("elapsed"),
        "error": job.get("error"),
        "text_length": len(job["text"]),
        "progress": job.get("progress"),
    })


@app.route("/tts/download/<job_id>")
def tts_download(job_id):
    if job_id not in tts_jobs or tts_jobs[job_id]["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    return send_file(tts_jobs[job_id]["output"], mimetype="audio/wav", as_attachment=True, download_name="synthesized.wav")


# ─── Routes: TTS Presets ───

@app.route("/tts/presets", methods=["GET"])
def list_presets():
    catalog = load_presets_catalog()
    return jsonify(catalog)


@app.route("/tts/presets", methods=["POST"])
def create_preset():
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    language = request.form.get("language", "ru")

    # Synthesis params to save with preset
    params = {
        "temperature": request.form.get("temperature", "0.75"),
        "speed": request.form.get("speed", "1.0"),
        "top_k": request.form.get("top_k", "50"),
        "top_p": request.form.get("top_p", "0.85"),
        "repetition_penalty": request.form.get("repetition_penalty", "10.0"),
        "style_intensity": request.form.get("style_intensity", "1.0"),
        "timbre_intensity": request.form.get("timbre_intensity", "1.0"),
        "variability": request.form.get("variability", "0.0"),
        "pitch_shift": request.form.get("pitch_shift", "0.0"),
    }

    # Voice sample: uploaded file or from transcription job
    voice_sample_path = None
    preset_id = str(uuid.uuid4())[:8]
    preset_dir = os.path.join(PRESETS_DIR, preset_id)
    os.makedirs(preset_dir, exist_ok=True)
    voice_wav = os.path.join(preset_dir, "voice.wav")

    if "voice_sample" in request.files and request.files["voice_sample"].filename:
        f = request.files["voice_sample"]
        ext = os.path.splitext(f.filename)[1]
        raw_path = os.path.join(preset_dir, f"raw{ext}")
        f.save(raw_path)
        try:
            extract_voice_sample(raw_path, voice_wav, start=0, duration=30)
        except subprocess.CalledProcessError:
            # If conversion fails, try using raw file
            import shutil
            shutil.copy(raw_path, voice_wav)
        voice_sample_path = voice_wav

    elif request.form.get("source_job_id"):
        source_job_id = request.form["source_job_id"]
        if source_job_id in jobs and jobs[source_job_id].get("filepath"):
            source_path = jobs[source_job_id]["filepath"]
            start = float(request.form.get("sample_start", 0))
            duration = float(request.form.get("sample_duration", 30))
            try:
                extract_voice_sample(source_path, voice_wav, start=start, duration=duration)
                voice_sample_path = voice_wav
            except subprocess.CalledProcessError as e:
                return jsonify({"error": f"Failed to extract voice: {e}"}), 500

    elif request.form.get("source_preset_id"):
        # Copy voice and latents from existing preset
        import shutil
        source_id = request.form["source_preset_id"]
        source_dir = os.path.join(PRESETS_DIR, source_id)
        source_voice = os.path.join(source_dir, "voice.wav")
        source_latents = os.path.join(source_dir, "latents.pt")
        if not os.path.exists(source_latents):
            shutil.rmtree(preset_dir, ignore_errors=True)
            return jsonify({"error": "Source preset not found"}), 404
        shutil.copy(source_latents, os.path.join(preset_dir, "latents.pt"))
        if os.path.exists(source_voice):
            shutil.copy(source_voice, voice_wav)
        voice_sample_path = voice_wav

    if not voice_sample_path:
        import shutil
        shutil.rmtree(preset_dir, ignore_errors=True)
        return jsonify({"error": "No voice sample provided"}), 400

    # Compute and cache latents (requires model — skip if copied from preset)
    latents_path = os.path.join(preset_dir, "latents.pt")
    if not os.path.exists(latents_path):
        try:
            model = get_tts()
            gpt_cond_latent, speaker_embedding = compute_latents(model, voice_sample_path)
            save_latents(gpt_cond_latent, speaker_embedding, latents_path)
        except Exception as e:
            import shutil
            shutil.rmtree(preset_dir, ignore_errors=True)
            return jsonify({"error": f"Failed to compute voice latents: {e}"}), 500

    # Update catalog
    import datetime
    catalog = load_presets_catalog()
    preset_entry = {
        "id": preset_id,
        "name": name,
        "language": language,
        "created_at": datetime.datetime.now().isoformat(),
        "params": params,
    }
    catalog.append(preset_entry)
    save_presets_catalog(catalog)

    return jsonify(preset_entry), 201


@app.route("/tts/presets/<preset_id>", methods=["DELETE"])
def delete_preset(preset_id):
    catalog = load_presets_catalog()
    new_catalog = [p for p in catalog if p["id"] != preset_id]
    if len(new_catalog) == len(catalog):
        return jsonify({"error": "Preset not found"}), 404

    # Remove files
    import shutil
    preset_dir = os.path.join(PRESETS_DIR, preset_id)
    shutil.rmtree(preset_dir, ignore_errors=True)

    save_presets_catalog(new_catalog)
    return jsonify({"ok": True})


# ─── Routes: List completed transcription jobs (for TTS source) ───

@app.route("/jobs")
def list_jobs():
    result = []
    for j in jobs.values():
        result.append({
            "id": j["id"],
            "filename": j["filename"],
            "status": j["status"],
            "duration": j.get("duration"),
        })
    return jsonify(result)


# ─── LLM Text Cleanup ───

LLM_SYSTEM_PROMPT = """Ты — редактор транскрибированной речи. Задача:
1. Убери слова-паразиты (ну, вот, как бы, типа, в общем, короче, значит, там, ну вот)
2. Убери повторы слов и фраз
3. Исправь пунктуацию — расставь точки, запятые, тире
4. Исправь грамматические ошибки
5. Разбей на абзацы по смыслу
6. Сохрани стиль и смысл речи — не переписывай, а чисти

Верни ТОЛЬКО исправленный текст, без комментариев."""

LLM_PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "format": "openai",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "format": "openai",
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-haiku-4-5-20251001",
        "format": "anthropic",
    },
}


def call_llm(text, provider, api_key):
    """Call LLM API to clean up transcribed text."""
    import urllib.request

    cfg = LLM_PROVIDERS.get(provider)
    if not cfg:
        raise ValueError(f"Unknown provider: {provider}")

    if cfg["format"] == "openai":
        payload = json.dumps({
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.3,
            "max_tokens": 8192,
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "whisper-tts-studio/1.0",
        }
    else:  # anthropic
        payload = json.dumps({
            "model": cfg["model"],
            "max_tokens": 8192,
            "system": LLM_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": text},
            ],
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "User-Agent": "whisper-tts-studio/1.0",
        }

    req = urllib.request.Request(cfg["url"], data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API error {e.code}: {body}")

    if cfg["format"] == "openai":
        return data["choices"][0]["message"]["content"].strip()
    else:
        return data["content"][0]["text"].strip()


@app.route("/llm/settings", methods=["GET"])
def get_llm_settings():
    return jsonify({
        "provider": llm_settings["provider"],
        "has_key": bool(llm_settings["api_key"]),
    })


@app.route("/llm/settings", methods=["POST"])
def set_llm_settings():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400
    if "provider" in data and data["provider"] in LLM_PROVIDERS:
        llm_settings["provider"] = data["provider"]
    if "api_key" in data:
        llm_settings["api_key"] = data["api_key"]
    save_llm_settings_file(llm_settings)
    return jsonify({"ok": True, "provider": llm_settings["provider"], "has_key": bool(llm_settings["api_key"])})


@app.route("/cleanup/<job_id>", methods=["POST"])
def cleanup_text(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    job = jobs[job_id]
    if job["status"] != "done":
        return jsonify({"error": "Job not ready"}), 400

    provider = llm_settings["provider"]
    api_key = llm_settings["api_key"]
    if not api_key:
        return jsonify({"error": "API key not configured"}), 400

    full_text = "\n".join(s["text"] for s in job.get("segments", []))
    if not full_text.strip():
        return jsonify({"error": "No text to clean"}), 400

    # Split into chunks of ~3000 chars to fit in max_tokens
    chunks = []
    current = ""
    for line in full_text.split("\n"):
        if len(current) + len(line) + 1 > 3000 and current:
            chunks.append(current)
            current = line
        else:
            current = (current + "\n" + line).strip()
    if current:
        chunks.append(current)

    cleaned_parts = []
    try:
        for i, chunk in enumerate(chunks):
            if i > 0:
                time.sleep(2)  # respect rate limits
            result = call_llm(chunk, provider, api_key)
            cleaned_parts.append(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    cleaned_text = "\n\n".join(cleaned_parts)
    job["cleaned_text"] = cleaned_text

    return jsonify({"ok": True, "cleaned_text": cleaned_text})


@app.route("/cleanup/<job_id>/undo", methods=["POST"])
def undo_cleanup(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404
    jobs[job_id].pop("cleaned_text", None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"Preloading whisper: {DEFAULT_MODEL} ...")
    get_whisper(DEFAULT_MODEL)
    print("Whisper ready! TTS (XTTS v2) will load on first use.")
    print("Starting server on :5000 ...")
    app.run(host="0.0.0.0", port=5000)
