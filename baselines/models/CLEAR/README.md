# CLEAR — Code-Mixed ASR (Hindi-English)

Fine-tune Whisper với *descriptive prompting* (đóng băng encoder, chỉ train decoder) cho dữ liệu code-switching MUCS, theo paper *CLEAR: Code-Mixed ASR with LLM-Driven Rescoring*.

> Mọi lệnh chạy từ thư mục `baselines/CLEAR/`.

## 1. Tạo môi trường

```bash
conda env create -f environment.yml
conda activate CLEAR
```

## 2. Chuẩn bị dữ liệu

Dữ liệu gốc ở định dạng Kaldi: `data/train/` và `data/test/` chứa các file `.wav` (recording đầy đủ) và `data/{train,test}/transcripts/{text,segments,wav.scp}`.

Script dưới đây cắt audio theo `segments`, resample về 16 kHz mono, và sinh CSV (`path,text`) mà code training cần:

```bash
python prepare_data.py --split-dir data/train --out-csv data/train.csv --out-audio data/train_segmented
python prepare_data.py --split-dir data/test  --out-csv data/test.csv  --out-audio data/test_segmented
```

Sau bước này có `data/train.csv` và `data/test.csv`.

## 3. Train (CLEAR)

CLEAR = prompt + đóng băng encoder (Whisper-small), 10 epoch, lr 1e-4, batch 16 (theo paper, hợp A100):

```bash
python whisper_fine.py \
  --dataset mucs \
  --model small \
  --freeze \
  --exp-name clear_small \
  --batch 16 \
  --epoch 10 \
  --lr 1e-4
```

- Kết quả/checkpoint lưu ở `whisper_prompt_2_results/results/clear_small/`.
- Theo dõi log: `tensorboard --logdir whisper_prompt_2_results/`.
- GPU: script dùng `CUDA_VISIBLE_DEVICES=0` (sửa ở đầu `whisper_fine.py` nếu cần card khác).

### Submit lên SLURM (A100)

```bash
mkdir -p logs
sbatch submit-job.sh
```

Theo dõi job:

```bash
squeue -u $USER          # xem trạng thái job
tail -f logs/train_<JOB_ID>.out   # xem log realtime
```

`submit-job.sh` request 1× A100, 8 CPU, 80 GB RAM. Điều chỉnh `--mem-per-cpu` hoặc `--batch` nếu OOM.

> Dữ liệu hiện chỉ có train + test (không có blind set), nên eval và test đều dùng `test.csv`.

### Baseline Whisper fine-tune thường (không prompt)

```bash
python finetune.py --dataset codemixed --model small --exp-name ft_small --batch 16 --epoch 10
```

## 4. (Tùy chọn) LLM Rescoring

Sau khi train, sinh n-best hypotheses rồi rescore bằng LLM để giảm WER thêm:

```bash
python generate_pred_ref.py     # sinh n-best từ model đã train
python rescoring_inference.py   # rescore bằng LLM (vd GPT-2)
```

## Ghi chú

- `transformers_prompt/` là bản fork của HuggingFace Transformers (đã có sẵn `WhisperPromptForConditionalGeneration` để chèn prompt sau token `<|sop|>`). `whisper_fine.py` import từ fork này, không cần chỉnh.
- Prompt mô tả được hardcode trong `data/dataloader.py` (`CodeMixedWhisperDataset`).
- Tokenizer cấu hình ngôn ngữ Hindi (`language='hi'`).
