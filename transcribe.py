#!/usr/bin/env python3
import sys
import os
import time
import torch
from faster_whisper import WhisperModel

def main():
    if len(sys.argv) < 2:
        print("Usage: transcribe.py <audio_or_video_file> [model_size]")
        print("Models: tiny, base, small, medium, large-v3, large-v3-turbo")
        print("Default: large-v3-turbo (best speed/quality for Russian)")
        sys.exit(1)

    input_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "large-v3-turbo"

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)

    print(f"Loading model: {model_size} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Using device: {device}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"Transcribing: {input_file}")
    start = time.time()

    segments, info = model.transcribe(
        input_file,
        language="ru",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    print(f"Detected language: {info.language} (prob: {info.language_probability:.2f})")
    print(f"Duration: {info.duration:.1f}s")
    print("=" * 60)

    output_lines = []
    for segment in segments:
        ts = f"[{format_time(segment.start)} -> {format_time(segment.end)}]"
        line = f"{ts} {segment.text.strip()}"
        print(line)
        output_lines.append(line)

    elapsed = time.time() - start
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s (x{info.duration / elapsed:.1f} realtime)")

    # Save to file
    base = os.path.splitext(input_file)[0]
    out_path = base + ".txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"Saved to: {out_path}")


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    main()
