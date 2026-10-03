import sys
import os
import csv
import io
import subprocess
import tempfile
import numpy as np
import tensorflow_hub as hub
import librosa

# =====================================
# Settings
# =====================================
SAMPLE_RATE = 16000
CHUNK_SECONDS = 2
TOP_N = 5


def extract_audio_ffmpeg(video_path, sr=SAMPLE_RATE):
    """ใช้ ffmpeg แตกเสียงออกมาเป็น wav ชั่วคราว (เชื่อถือได้กว่า moviepy)"""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",          # mono
        "-ar", str(sr),      # sample rate
        "-vn",                # ไม่เอา video stream
        tmp_wav.name,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[DEBUG] ffmpeg error output:")
        print(result.stderr)
        raise RuntimeError("ffmpeg แตกเสียงไม่สำเร็จ ดู error ด้านบน")

    return tmp_wav.name


def load_audio(path):
    """โหลดเสียงจากไฟล์วิดีโอหรือไฟล์เสียง แปลงเป็น mono float32 @ 16kHz"""
    ext = os.path.splitext(path)[1].lower()

    if ext in [".mp4", ".webm", ".mov", ".avi", ".mkv"]:
        wav_path = extract_audio_ffmpeg(path)
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        os.remove(wav_path)
    else:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    audio = audio.astype(np.float32)

    # =====================================
    # 🐞 DEBUG: เช็คว่า audio โหลดมาถูกต้องไหม
    # =====================================
    print("-" * 50)
    print("[DEBUG] Audio stats after loading:")
    print(f"  dtype        : {audio.dtype}")
    print(f"  shape        : {audio.shape}")
    print(f"  min value    : {np.min(audio):.6f}")
    print(f"  max value    : {np.max(audio):.6f}")
    print(f"  max |value|  : {np.max(np.abs(audio)):.6f}")
    print(f"  mean |value| : {np.mean(np.abs(audio)):.6f}")
    print(f"  RMS          : {np.sqrt(np.mean(audio ** 2)):.6f}")
    print(f"  unique values (first 5): {np.unique(audio)[:5]}")
    print("-" * 50)

    if np.min(audio) == np.max(audio):
        print("[DEBUG] ⚠️⚠️⚠️ Audio เป็นค่าคงที่ทั้งก้อน (flat) แปลว่าโหลดเสียงผิดพลาด!")

    peak = np.max(np.abs(audio))
    if peak > 0 and peak < 0.1 and np.min(audio) != np.max(audio):
        print(f"[DEBUG] ⚠️ Audio ดูเบาผิดปกติ (peak={peak:.6f}) -> normalize ให้อัตโนมัติ")
        audio = audio / peak

    return audio


def get_class_names(yamnet):
    class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
    class_names = []
    with tf_io_open(class_map_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            class_names.append(row[2])
    return class_names


def tf_io_open(path):
    import tensorflow as tf
    return io.TextIOWrapper(io.BytesIO(tf.io.read_file(path).numpy()))


def format_time(seconds):
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def chunk_audio(audio, sample_rate, chunk_seconds):
    chunk_size = int(chunk_seconds * sample_rate)
    chunks = []
    for start in range(0, len(audio), chunk_size):
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]
        if len(chunk) < int(0.96 * sample_rate):
            continue
        start_sec = start / sample_rate
        chunks.append((start_sec, chunk))
    return chunks


def main(video_path):
    print("=" * 50)
    print("Loading YAMNet...")
    print("=" * 50)
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    class_names = get_class_names(yamnet)

    gun_keywords = ["gun", "gunshot", "gunfire", "machine gun", "artillery", "explosion"]
    gun_indices = [
        i for i, name in enumerate(class_names)
        if any(k in name.lower() for k in gun_keywords)
    ]

    print("=" * 50)
    print(f"Loading audio from: {video_path}")
    print("=" * 50)
    audio = load_audio(video_path)
    total_duration = len(audio) / SAMPLE_RATE
    print(f"Audio length: {total_duration:.2f} seconds")

    chunks = chunk_audio(audio, SAMPLE_RATE, CHUNK_SECONDS)
    print(f"Split into {len(chunks)} chunk(s) of ~{CHUNK_SECONDS}s each")

    print("=" * 50)
    print("Running YAMNet per chunk...")
    print("=" * 50)

    for start_sec, chunk in chunks:
        end_sec = start_sec + (len(chunk) / SAMPLE_RATE)
        chunk_peak = np.max(np.abs(chunk))

        scores, embeddings, spectrogram = yamnet(chunk)
        scores_np = scores.numpy()

        max_scores = np.max(scores_np, axis=0)
        top_indices = np.argsort(max_scores)[::-1][:TOP_N]

        print(f"\n[{format_time(start_sec)} - {format_time(end_sec)}]  (chunk peak amplitude: {chunk_peak:.4f})")
        for i in top_indices:
            print(f"  {class_names[i]:30s}  {max_scores[i] * 100:.2f}%")

        print("  [debug: gun/explosion scores]")
        for i in gun_indices:
            rank = int(np.where(np.argsort(max_scores)[::-1] == i)[0][0]) + 1
            print(f"    {class_names[i]:30s}  {max_scores[i] * 100:.2f}%  (rank {rank})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_audio_or_video>")
        sys.exit(1)

    main(sys.argv[1])