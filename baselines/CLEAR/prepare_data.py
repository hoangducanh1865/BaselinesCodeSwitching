"""
Chuyển dữ liệu MUCS (định dạng Kaldi) -> audio đã cắt segment + file CSV.

Mỗi split (train/test) có cấu trúc:
    <split>/*.wav                       # các recording đầy đủ (theo session)
    <split>/transcripts/text            # "<utt_id> <transcript>"
    <split>/transcripts/segments        # "<utt_id> <rec_id> <start> <end>"
    <split>/transcripts/wav.scp         # "<rec_id> <path>"  (có thể bị lỗi, không bắt buộc dùng)

Script sẽ:
  1. Đọc text + segments.
  2. Với mỗi utterance: cắt đoạn [start, end] từ recording tương ứng,
     resample về 16 kHz mono, lưu thành <out_audio>/<utt_id>.wav
  3. Ghi CSV với 2 cột: path,text  (đúng định dạng CodeMixedWhisperDataset cần).

Cách dùng:
    python prepare_data.py --split-dir data/train --out-csv data/train.csv --out-audio data/train_segmented
    python prepare_data.py --split-dir data/test  --out-csv data/test.csv  --out-audio data/test_segmented
"""
import argparse
import csv
import os

import torch
import torchaudio


def read_text(path):
    """utt_id -> transcript"""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            utt_id, text = parts[0], parts[1].strip()
            if text:
                mapping[utt_id] = text
    return mapping


def read_segments(path):
    """utt_id -> (rec_id, start, end)"""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            utt_id, rec_id, start, end = parts[0], parts[1], parts[2], parts[3]
            try:
                mapping[utt_id] = (rec_id, float(start), float(end))
            except ValueError:
                continue
    return mapping


def build_wav_index(split_dir):
    """basename (không đuôi) -> đường dẫn tuyệt đối của file .wav"""
    index = {}
    for f in os.listdir(split_dir):
        if f.lower().endswith(".wav"):
            index[os.path.splitext(f)[0]] = os.path.join(split_dir, f)
    return index


def resolve_wav(rec_id, wav_index):
    """Tìm file recording cho rec_id.

    Tên file thực tế trong thư mục là phần hậu tố sau dấu '_' đầu tiên của rec_id
    (vd: rec_id '265143_TfBkN3Trta3SUkz4' -> 'TfBkN3Trta3SUkz4.wav').
    Vẫn thử cả rec_id đầy đủ để phòng trường hợp khác.
    """
    if rec_id in wav_index:
        return wav_index[rec_id]
    if "_" in rec_id:
        suffix = rec_id.split("_", 1)[1]
        if suffix in wav_index:
            return wav_index[suffix]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True, help="thư mục split, vd: data/train")
    ap.add_argument("--out-csv", required=True, help="đường dẫn CSV xuất ra")
    ap.add_argument("--out-audio", required=True, help="thư mục lưu audio đã cắt segment")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--min-dur", type=float, default=0.2, help="bỏ qua segment quá ngắn (giây)")
    ap.add_argument("--max-dur", type=float, default=30.0, help="bỏ qua segment dài hơn (giây, giới hạn Whisper)")
    args = ap.parse_args()

    split_dir = os.path.abspath(args.split_dir)
    out_audio = os.path.abspath(args.out_audio)
    os.makedirs(out_audio, exist_ok=True)

    text = read_text(os.path.join(split_dir, "transcripts", "text"))
    segments = read_segments(os.path.join(split_dir, "transcripts", "segments"))
    wav_index = build_wav_index(split_dir)

    print(f"[{split_dir}] text={len(text)}  segments={len(segments)}  recordings={len(wav_index)}")

    # cache recording đã load + sample rate gốc để không load lại nhiều lần
    cache = {}

    rows = []
    n_written = n_missing_audio = n_missing_text = n_skip_dur = 0

    for utt_id, (rec_id, start, end) in segments.items():
        if utt_id not in text:
            n_missing_text += 1
            continue
        dur = end - start
        if dur < args.min_dur or dur > args.max_dur:
            n_skip_dur += 1
            continue

        wav_path = resolve_wav(rec_id, wav_index)
        if wav_path is None:
            n_missing_audio += 1
            continue

        if rec_id not in cache:
            wav, sr = torchaudio.load(wav_path)
            if wav.size(0) > 1:  # stereo -> mono
                wav = wav.mean(dim=0, keepdim=True)
            if sr != args.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, args.sample_rate)
            cache[rec_id] = wav
        wav = cache[rec_id]

        s = int(round(start * args.sample_rate))
        e = int(round(end * args.sample_rate))
        chunk = wav[:, s:e]
        if chunk.numel() == 0:
            n_skip_dur += 1
            continue

        out_path = os.path.join(out_audio, f"{utt_id}.wav")
        torchaudio.save(out_path, chunk, args.sample_rate)
        rows.append((out_path, text[utt_id]))
        n_written += 1

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "text"])
        writer.writerows(rows)

    print(
        f"Đã ghi {n_written} utterances -> {args.out_csv}\n"
        f"  bỏ qua: thiếu audio={n_missing_audio}, thiếu text={n_missing_text}, "
        f"sai độ dài={n_skip_dur}"
    )


if __name__ == "__main__":
    main()
