# Google Colab edition of V4.0 Transformer trainer
import argparse
import os
import re
import shutil
import sys
import tempfile
import threading
import unicodedata
from contextlib import nullcontext
from datetime import datetime
import math
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import openpyxl

try:
    import sentencepiece as spm
    spm_import_error = None
except Exception as e:
    spm = None
    spm_import_error = str(e)


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def load_state_dict_safely(model: nn.Module, state_dict: dict) -> None:
    model_obj = unwrap_model(model)
    try:
        model_obj.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass
    if any(k.startswith("module.") for k in state_dict.keys()):
        stripped = {k[len("module.") :]: v for k, v in state_dict.items()}
        model_obj.load_state_dict(stripped)
        return
    prefixed = {f"module.{k}": v for k, v in state_dict.items()}
    model_obj.load_state_dict(prefixed)


def set_seed(seed: int) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _find_first_under(root: Path, filename: str) -> str | None:
    try:
        for p in root.rglob(filename):
            if p.is_file():
                return str(p)
    except Exception:
        return None
    return None


def _in_colab() -> bool:
    return "google.colab" in sys.modules or Path("/content").exists()


def maybe_mount_google_drive(enabled: bool = True) -> None:
    if not enabled or not _in_colab():
        return
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        return
    try:
        from google.colab import drive

        print("检测到 Google Colab，正在挂载 Google Drive...", flush=True)
        drive.mount("/content/drive", force_remount=False)
    except Exception as e:
        print(f"Google Drive 挂载跳过: {e}", flush=True)


def default_model_dir() -> str:
    drive_dir = Path("/content/drive/MyDrive/v4_transformer_colab")
    if drive_dir.parent.exists():
        return str(drive_dir)
    return "/content/v4_transformer_colab"


def default_inputs() -> list[str]:
    candidates = [
        Path("/content/drive/MyDrive/translation_data"),
        Path("/content/drive/MyDrive"),
        Path("/content"),
    ]
    for p in candidates:
        if p.exists():
            return [str(p)]
    return ["/content"]


def _maybe_copy_resume_artifacts_from_search_roots(model_dir: str, search_roots: list[str] | None) -> None:
    roots: list[Path] = []
    for raw in (search_roots or []):
        if not raw:
            continue
        p = Path(raw)
        if p.exists():
            roots.append(p)

    if not roots:
        return

    dst_dir = Path(model_dir)
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    wanted = [
        "best_model_v4_transformer.pth",
        "best_model_v4_transformer.ckpt",
        "spm_ko_v4.model",
        "spm_ko_v4.vocab",
        "spm_zh_v4.model",
        "spm_zh_v4.vocab",
    ]

    copied = 0
    for name in wanted:
        dst = dst_dir / name
        if dst.exists():
            continue
        found = None
        for root in roots:
            found = _find_first_under(root, name)
            if found:
                break
        if not found:
            continue
        try:
            shutil.copy2(found, str(dst))
            copied += 1
        except Exception:
            continue

    if copied:
        print(f"从搜索目录复制恢复文件到 model_dir: copied={copied}", flush=True)


# ==========================================
# 文本清洗与处理逻辑 (继承自 V3.2.1)
# ==========================================

def _normalize_text(sentence: str) -> str:
    if not isinstance(sentence, str):
        return ""
    s = sentence.replace("\r\n", "\n").replace("\r", "\n")
    circled_map = {
        "⓪": 0,
        "①": 1,
        "②": 2,
        "③": 3,
        "④": 4,
        "⑤": 5,
        "⑥": 6,
        "⑦": 7,
        "⑧": 8,
        "⑨": 9,
        "⑩": 10,
        "⑪": 11,
        "⑫": 12,
        "⑬": 13,
        "⑭": 14,
        "⑮": 15,
        "⑯": 16,
        "⑰": 17,
        "⑱": 18,
        "⑲": 19,
        "⑳": 20,
    }
    for ch, n in circled_map.items():
        s = s.replace(ch, f"({n})")
    s = s.replace("…", "...")
    s = re.sub(r"-{1,4}>", "->", s)
    for arrow in ("→", "⇒", "➡", "⟶", "⟶", "⟹", "➔", "➜", "➝", "➞", "➟", "➠"):
        s = s.replace(arrow, "->")
    for q in ("“", "”", "„", "‟", "＂"):
        s = s.replace(q, '"')
    s = unicodedata.normalize("NFKC", s)
    return s


def _separate_punct_boundaries(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    s = text
    s = s.replace("->", " -> ").replace("...", " ... ")
    for ch in [
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        ",",
        ":",
        ";",
        "?",
        "!",
        "/",
        "\\",
        "$",
        "#",
        "@",
        "~",
        "&",
        "*",
        "%",
        "+",
        "=",
        '"',
        "_",
        "-",
        "·",
    ]:
        s = s.replace(ch, f" {ch} ")
    s = re.sub(r"(?<!\d)\.(?!\d)", " . ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_text(sentence: str) -> str:
    if not isinstance(sentence, str):
        return ""
    s = _normalize_text(sentence).replace("\n", " ")
    kept = []
    for ch in s:
        if ch.isspace() or ("0" <= ch <= "9") or ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            kept.append(ch)
            continue
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3 or 0x4E00 <= code <= 0x9FFF:
            kept.append(ch)
            continue
        cat = unicodedata.category(ch)
        if (cat and cat[0] == "P") or cat in ("Sm", "Sc"):
            kept.append(ch)
            continue
    s = "".join(kept)
    s = _separate_punct_boundaries(s)
    return re.sub(r"\s+", " ", s).strip()


def find_excel_files(inputs: list[str]) -> list[str]:
    file_paths = []
    for raw in inputs:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() == ".xlsx":
            file_paths.append(str(p))
        elif p.is_dir():
            file_paths.extend(str(x) for x in p.rglob("*.xlsx"))
        elif "*" in raw or "?" in raw:
            file_paths.extend(str(x) for x in Path().glob(raw))
    return sorted(set(file_paths))


def read_corpus(file_paths: list[str]) -> tuple[list[str], list[str]]:
    all_korean, all_chinese = [], []
    for path in file_paths:
        try:
            wb = openpyxl.load_workbook(path, data_only=True)
            ws = wb.active
            for row in ws.iter_rows(min_row=2, values_only=True):
                if len(row) >= 4 and row[1] and row[3]:
                    all_korean.append(str(row[1]))
                    all_chinese.append(str(row[3]))
            wb.close()
        except Exception as e:
            print(f"读取文件 {path} 出错: {e}")
    return all_korean, all_chinese


# ==========================================
# Transformer 模型核心组件
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_src_vocab,
        n_trg_vocab,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(n_src_vocab, d_model)
        self.trg_embedding = nn.Embedding(n_trg_vocab, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(d_model, n_trg_vocab)
        self.src_pad_idx = 0
        self.trg_pad_idx = 0

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, trg):
        src_mask = None
        trg_mask = self.generate_square_subsequent_mask(trg.size(0)).to(trg.device)

        src_key_padding_mask = (src == self.src_pad_idx).transpose(0, 1)
        trg_key_padding_mask = (trg == self.trg_pad_idx).transpose(0, 1)

        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        trg_emb = self.pos_decoder(self.trg_embedding(trg) * math.sqrt(self.d_model))

        output = self.transformer(
            src_emb,
            trg_emb,
            src_mask=src_mask,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(output)


# ==========================================
# 训练与评估逻辑
# ==========================================

class TranslationIdDataset(Dataset):
    def __init__(self, ko_ids, zh_ids):
        self.ko_ids = ko_ids
        self.zh_ids = zh_ids

    def __len__(self):
        return len(self.ko_ids)

    def __getitem__(self, idx):
        return torch.LongTensor(self.ko_ids[idx]), torch.LongTensor(self.zh_ids[idx])


def collate_fn(batch):
    ko_batch, zh_batch = zip(*batch)
    ko_padded = nn.utils.rnn.pad_sequence(ko_batch, padding_value=0)
    zh_padded = nn.utils.rnn.pad_sequence(zh_batch, padding_value=0)
    return ko_padded, None, zh_padded, None


def train_sentencepiece_models(ko_sents, zh_sents, model_folder, vocab_size, num_threads=None, character_coverage=0.9995):
    os.makedirs(model_folder, exist_ok=True)
    ko_prefix = os.path.join(model_folder, "spm_ko_v4")
    zh_prefix = os.path.join(model_folder, "spm_zh_v4")
    ko_model = f"{ko_prefix}.model"
    zh_model = f"{zh_prefix}.model"
    if os.path.exists(ko_model) and os.path.exists(zh_model):
        return ko_model, zh_model

    if spm is None:
        raise RuntimeError(f"缺少 sentencepiece，错误: {spm_import_error}")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("正在写入 SentencePiece 训练文本（ko/zh）...", flush=True)

        def _write(path, sents):
            with open(path, "w", encoding="utf-8") as f:
                for s in sents:
                    f.write((s or "").replace("\n", " ") + "\n")

        ko_txt, zh_txt = os.path.join(tmpdir, "ko.txt"), os.path.join(tmpdir, "zh.txt")
        _write(ko_txt, ko_sents)
        _write(zh_txt, zh_sents)
        print("SentencePiece 训练文本写入完成。", flush=True)

        def _start_heartbeat(name: str):
            stop_evt = threading.Event()
            started = datetime.now()

            def _runner():
                while not stop_evt.wait(15.0):
                    elapsed = (datetime.now() - started).total_seconds()
                    print(f"SentencePiece({name}) 仍在训练中... elapsed={elapsed:.0f}s", flush=True)

            th = threading.Thread(target=_runner, daemon=True)
            th.start()
            return stop_evt

        nt = int(num_threads) if isinstance(num_threads, int) and int(num_threads) > 0 else max(1, min(8, os.cpu_count() or 2))

        def _train_spm(inp, pref):
            spm.SentencePieceTrainer.Train(
                input=inp,
                model_prefix=pref,
                vocab_size=vocab_size,
                model_type="bpe",
                character_coverage=character_coverage,
                pad_id=0,
                bos_id=1,
                eos_id=2,
                unk_id=3,
                hard_vocab_limit=False,
                num_threads=int(nt),
            )

        print(f"开始训练 SentencePiece(ko) vocab_size={vocab_size} ...", flush=True)
        hb = _start_heartbeat("ko")
        try:
            _train_spm(ko_txt, ko_prefix)
        finally:
            hb.set()
        print("SentencePiece(ko) 完成。", flush=True)

        print(f"开始训练 SentencePiece(zh) vocab_size={vocab_size} ...", flush=True)
        hb = _start_heartbeat("zh")
        try:
            _train_spm(zh_txt, zh_prefix)
        finally:
            hb.set()
        print("SentencePiece(zh) 完成。", flush=True)
    return ko_model, zh_model


def encode_with_spm(sentences, sp, max_len):
    bos, eos, all_ids, truncated = int(sp.bos_id()), int(sp.eos_id()), [], 0
    for s in sentences:
        ids = sp.encode(s, out_type=int)
        if len(ids) > (max_len - 2):
            ids = ids[: max_len - 2]
            truncated += 1
        all_ids.append([bos] + ids + [eos])
    return all_ids, truncated


def _truncate_seq(seq, eos_id: int, pad_id: int) -> list[int]:
    out: list[int] = []
    for x in seq:
        xi = int(x)
        if xi == pad_id:
            continue
        if xi == eos_id:
            break
        out.append(xi)
    return out


def _corpus_bleu(pred_seqs: list[list[int]], ref_seqs: list[list[int]], max_n: int = 4) -> float:
    if not pred_seqs or not ref_seqs:
        return 0.0
    max_n = int(max_n)
    max_n = 4 if max_n <= 0 else max_n

    import math as _math

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        match = 0
        total = 0
        for hyp, ref in zip(pred_seqs, ref_seqs):
            if len(hyp) < n:
                continue
            hyp_ngrams = {}
            for i in range(len(hyp) - n + 1):
                ng = tuple(hyp[i : i + n])
                hyp_ngrams[ng] = hyp_ngrams.get(ng, 0) + 1
            ref_ngrams = {}
            if len(ref) >= n:
                for i in range(len(ref) - n + 1):
                    ng = tuple(ref[i : i + n])
                    ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1
            for ng, c in hyp_ngrams.items():
                total += c
                match += min(c, ref_ngrams.get(ng, 0))
        precisions.append((match + 1.0) / (total + 1.0))

    hyp_len = sum(len(x) for x in pred_seqs)
    ref_len = sum(len(x) for x in ref_seqs)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else _math.exp(1.0 - (ref_len / float(hyp_len)))
    score = bp * _math.exp(sum(_math.log(p) for p in precisions) / float(len(precisions)))
    return float(score)


def _chrf(pred_texts: list[str], ref_texts: list[str], n: int = 6, beta: float = 2.0) -> float:
    if not pred_texts or not ref_texts:
        return 0.0
    n = int(n)
    if n <= 0:
        n = 6
    beta2 = float(beta) * float(beta)

    def _ngrams(s: str, k: int) -> dict[str, int]:
        out: dict[str, int] = {}
        if not s:
            return out
        for i in range(0, max(0, len(s) - k + 1)):
            ng = s[i : i + k]
            out[ng] = out.get(ng, 0) + 1
        return out

    tot_f = 0.0
    cnt = 0
    for hyp, ref in zip(pred_texts, ref_texts):
        hyp_s = hyp or ""
        ref_s = ref or ""
        p_sum = 0.0
        r_sum = 0.0
        for k in range(1, n + 1):
            h = _ngrams(hyp_s, k)
            r = _ngrams(ref_s, k)
            if not h and not r:
                p = 1.0
                rr = 1.0
            else:
                overlap = 0
                for ng, c in h.items():
                    overlap += min(c, r.get(ng, 0))
                h_total = sum(h.values())
                r_total = sum(r.values())
                p = (overlap / float(h_total)) if h_total else 0.0
                rr = (overlap / float(r_total)) if r_total else 0.0
            p_sum += p
            r_sum += rr
        p_avg = p_sum / float(n)
        r_avg = r_sum / float(n)
        denom = (beta2 * p_avg) + r_avg
        f = 0.0 if denom == 0.0 else ((1.0 + beta2) * p_avg * r_avg / denom)
        tot_f += f
        cnt += 1
    return float(tot_f / float(max(1, cnt)))


def train_model_transformer(train_loader, test_loader, ko_sp, zh_sp, device, args):
    model = TransformerModel(
        n_src_vocab=ko_sp.get_piece_size(),
        n_trg_vocab=zh_sp.get_piece_size(),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    if torch.cuda.device_count() > 1 and not args.no_multi_gpu:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    best_loss = float("inf")
    best_model_path = os.path.join(args.model_dir, "best_model_v4_transformer.pth")
    best_ckpt_path = os.path.join(args.model_dir, "best_model_v4_transformer.ckpt")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and (not args.no_amp)))

    pad_id = 0
    eos_id = int(zh_sp.eos_id())
    unk_id = int(zh_sp.unk_id())

    os.makedirs(args.model_dir, exist_ok=True)
    start_epoch = 0
    no_improve = 0
    patience = int(args.early_stop_patience)

    grad_accum_steps = max(1, int(args.grad_accum_steps))
    log_every = max(1, int(args.log_every))
    show_samples = max(0, int(getattr(args, "show_samples", 0)))
    metric_samples = max(0, int(getattr(args, "metric_samples", 1000)))

    import math as _math
    import torch as _torch

    updates_per_epoch = (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps
    total_updates = max(1, int(args.epochs) * int(updates_per_epoch))
    warmup_steps = int(getattr(args, "warmup_steps", 4000))
    if warmup_steps <= 0:
        warmup_steps = max(100, total_updates // 20)
    warmup_steps = min(warmup_steps, max(1, total_updates // 2))

    base_lr = float(args.lr)
    min_lr = float(args.lr_min)
    min_ratio = 0.0 if base_lr <= 0 else max(0.0, min(1.0, min_lr / base_lr))

    def _lr_lambda(step: int) -> float:
        s = int(step)
        if s < warmup_steps:
            return float(s + 1) / float(warmup_steps)
        t = float(s - warmup_steps)
        T = float(max(1, total_updates - warmup_steps))
        cosine = 0.5 * (1.0 + _math.cos(_math.pi * min(1.0, t / T)))
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = _torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    if str(args.resume_mode) != "none" and os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            if str(args.resume_mode) == "model" and os.path.exists(best_model_path):
                load_state_dict_safely(model, torch.load(best_model_path, map_location="cpu"))
            else:
                load_state_dict_safely(model, ckpt["model_state_dict"])
            if str(args.resume_mode) == "full":
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if "scaler_state_dict" in ckpt and scaler.is_enabled():
                    scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = 0 if str(args.resume_mode) == "model" else int(ckpt.get("epoch", 0))
            best_loss = float(ckpt.get("best_test_loss", best_loss))
            no_improve = 0 if str(args.resume_mode) == "model" else int(ckpt.get("no_improve", 0))
            print(f"检测到已存在的 Transformer Checkpoint，继续训练: {best_ckpt_path}", flush=True)
            if str(args.resume_mode) == "model":
                print("Resume Mode = model：从 best 权重开始精调（优化器/调度器重置，epoch 从 1 重新计）", flush=True)

    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)

        for i, (src, _, trg, _) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)
            trg_input = trg[:-1, :]
            trg_real = trg[1:, :]

            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=(device.type == "cuda" and (not args.no_amp)))
                if hasattr(torch, "amp")
                else (torch.cuda.amp.autocast(enabled=(device.type == "cuda" and (not args.no_amp))) if device.type == "cuda" else nullcontext())
            )
            with autocast_ctx:
                output = model(src, trg_input)
                loss = criterion(output.reshape(-1, output.shape[-1]), trg_real.reshape(-1))
                loss = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (i + 1) % grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * grad_accum_steps
            if (i + 1) % log_every == 0:
                print(f"Epoch: {epoch+1:02}, Batch: {i+1}/{len(train_loader)}, Loss: {float(loss.item())*grad_accum_steps:.4f}", flush=True)

        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        correct_tokens = 0
        total_tokens = 0
        unk_tokens = 0
        exact_match = 0
        total_seqs = 0
        pred_len_sum = 0
        trg_len_sum = 0
        shown = 0
        bleu_pred: list[list[int]] = []
        bleu_ref: list[list[int]] = []
        chrf_pred_text: list[str] = []
        chrf_ref_text: list[str] = []

        with torch.no_grad():
            for src, _, trg, _ in test_loader:
                src, trg = src.to(device), trg.to(device)
                trg_input, trg_real = trg[:-1, :], trg[1:, :]
                output = model(src, trg_input)
                loss = criterion(output.reshape(-1, output.shape[-1]), trg_real.reshape(-1))
                val_loss += float(loss.item())

                preds = output.argmax(dim=-1)
                mask = trg_real != pad_id
                correct_tokens += int(((preds == trg_real) & mask).sum().item())
                total_tokens += int(mask.sum().item())
                unk_tokens += int(((preds == unk_id) & mask).sum().item())

                bsz = int(trg_real.shape[1])
                exact_match += int((((preds == trg_real) | (~mask)).all(dim=0)).sum().item())
                total_seqs += bsz

                for bi in range(bsz):
                    trg_seq = trg_real[:, bi]
                    trg_valid = int((trg_seq != pad_id).sum().item())
                    trg_len_sum += trg_valid

                    pred_seq = preds[:, bi]
                    eos_pos = (pred_seq == eos_id).nonzero(as_tuple=False)
                    if eos_pos.numel() > 0:
                        pred_len = int(eos_pos[0].item()) + 1
                        pred_len_sum += max(1, pred_len)
                    else:
                        pred_len_sum += max(1, trg_valid)

                    if metric_samples and len(bleu_pred) < metric_samples:
                        hyp = _truncate_seq(pred_seq.tolist(), eos_id=eos_id, pad_id=pad_id)
                        ref = _truncate_seq(trg_seq.tolist(), eos_id=eos_id, pad_id=pad_id)
                        bleu_pred.append(hyp)
                        bleu_ref.append(ref)
                        chrf_pred_text.append(zh_sp.decode(hyp) if hyp else "")
                        chrf_ref_text.append(zh_sp.decode(ref) if ref else "")

                if show_samples and shown < show_samples:
                    for bi in range(int(trg_real.shape[1])):
                        if shown >= show_samples:
                            break
                        pred_seq = preds[:, bi].tolist()
                        trg_seq = trg_real[:, bi].tolist()
                        if eos_id in pred_seq:
                            pred_seq = pred_seq[: pred_seq.index(eos_id)]
                        if eos_id in trg_seq:
                            trg_seq = trg_seq[: trg_seq.index(eos_id)]
                        pred_text = zh_sp.decode(pred_seq) if pred_seq else ""
                        trg_text = zh_sp.decode(trg_seq) if trg_seq else ""
                        print(f"SAMPLE_PRED: {pred_text}", flush=True)
                        print(f"SAMPLE_GOLD: {trg_text}", flush=True)
                        shown += 1

        avg_val_loss = val_loss / max(1, len(test_loader))
        ppl = math.exp(min(20.0, avg_val_loss))
        acc = (correct_tokens / total_tokens) if total_tokens else 0.0
        unk_rate = (unk_tokens / total_tokens) if total_tokens else 0.0
        em = (exact_match / total_seqs) if total_seqs else 0.0
        len_ratio = (pred_len_sum / trg_len_sum) if trg_len_sum else 0.0
        lr_now = float(optimizer.param_groups[0]["lr"])
        bleu = _corpus_bleu(bleu_pred, bleu_ref, max_n=4) if bleu_pred else 0.0
        chrf = _chrf(chrf_pred_text, chrf_ref_text, n=6, beta=2.0) if chrf_pred_text else 0.0
        print(
            f"Epoch: {epoch+1:02} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_val_loss:.4f} | PPL: {ppl:.2f} | Acc: {acc:.3f} | EM: {em:.3f} | UNK: {unk_rate:.3f} | LenRatio: {len_ratio:.3f} | BLEU: {bleu:.3f} | chrF: {chrf:.3f} | LR: {lr_now:.6f}",
            flush=True,
        )

        state = {
            "epoch": epoch + 1,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "best_test_loss": best_loss,
            "no_improve": no_improve,
            "ko_spm_model": os.path.join(args.model_dir, "spm_ko_v4.model"),
            "zh_spm_model": os.path.join(args.model_dir, "spm_zh_v4.model"),
            "multi_gpu": bool(isinstance(model, nn.DataParallel)),
        }

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve = 0
            torch.save(unwrap_model(model).state_dict(), best_model_path)
            state["best_test_loss"] = best_loss
            state["no_improve"] = no_improve
            torch.save(state, best_ckpt_path)
            print("  -> 测试损失下降，保存最佳模型与训练状态！", flush=True)
        else:
            no_improve += 1
            state["no_improve"] = no_improve
            torch.save(state, best_ckpt_path)
            print(f"  -> 测试损失未下降 (Patience: {no_improve}/{patience})", flush=True)
            if no_improve >= patience:
                print("早停触发，训练结束。", flush=True)
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=None)
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--enc_layers", type=int, default=6)
    parser.add_argument("--dec_layers", type=int, default=6)
    parser.add_argument("--dim_ff", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_ko_len", type=int, default=128)
    parser.add_argument("--max_zh_len", type=int, default=96)
    parser.add_argument("--spm_threads", type=int, default=2)
    parser.add_argument("--spm_character_coverage", type=float, default=0.9995)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_multi_gpu", action="store_true")
    parser.add_argument("--no_drive_mount", action="store_true")
    parser.add_argument("--resume_mode", choices=["none", "model", "full"], default="none")
    parser.add_argument("--resume_search_roots", nargs="*", default=None)
    parser.add_argument("--lr_factor", type=float, default=0.7)
    parser.add_argument("--lr_patience", type=int, default=4)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--show_samples", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--metric_samples", type=int, default=500)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"忽略额外参数: {unknown}", flush=True)

    maybe_mount_google_drive(enabled=(not args.no_drive_mount))

    if not args.model_dir:
        args.model_dir = default_model_dir()
    if not args.inputs:
        args.inputs = default_inputs()
    if args.resume_search_roots is None:
        args.resume_search_roots = []

    if spm is None:
        raise RuntimeError(
            f"缺少 sentencepiece，错误: {spm_import_error}\n"
            "Google Colab 可先运行: !pip install -q sentencepiece openpyxl scikit-learn"
        )

    os.makedirs(str(args.model_dir), exist_ok=True)
    _maybe_copy_resume_artifacts_from_search_roots(str(args.model_dir), list(args.resume_search_roots))
    print(f"输入路径: {args.inputs}", flush=True)
    print(f"模型目录: {args.model_dir}", flush=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_paths = find_excel_files(list(args.inputs))
    if not file_paths:
        raise RuntimeError(f"未找到任何 xlsx 文件，inputs={args.inputs}")
    print(f"发现 xlsx 文件数量: {len(file_paths)}", flush=True)

    print("开始读取语料...", flush=True)
    ko_sents, zh_sents = read_corpus(file_paths)
    ko_sents = [clean_text(s) for s in ko_sents]
    zh_sents = [clean_text(s) for s in zh_sents]
    nonempty = [(k, z) for k, z in zip(ko_sents, zh_sents) if k and z]
    ko_sents = [k for k, _ in nonempty]
    zh_sents = [z for _, z in nonempty]
    print(f"读取完成，句对数量: {len(ko_sents)}", flush=True)

    print("开始准备 SentencePiece（复用或训练）...", flush=True)
    ko_mod, zh_mod = train_sentencepiece_models(
        ko_sents,
        zh_sents,
        args.model_dir,
        int(args.vocab_size),
        num_threads=(int(args.spm_threads) if int(args.spm_threads) > 0 else None),
        character_coverage=float(args.spm_character_coverage),
    )
    ko_sp = spm.SentencePieceProcessor(model_file=ko_mod)
    zh_sp = spm.SentencePieceProcessor(model_file=zh_mod)

    ko_ids, ko_trunc = encode_with_spm(ko_sents, ko_sp, int(args.max_ko_len))
    zh_ids, zh_trunc = encode_with_spm(zh_sents, zh_sp, int(args.max_zh_len))
    print(
        f"BPE 编码完成。max_ko_len={int(args.max_ko_len)} 截断={ko_trunc} | max_zh_len={int(args.max_zh_len)} 截断={zh_trunc}",
        flush=True,
    )

    ko_train, ko_test, zh_train, zh_test = train_test_split(
        ko_ids,
        zh_ids,
        test_size=float(args.test_size),
        random_state=int(args.seed),
    )

    train_ds = TranslationIdDataset(ko_train, zh_train)
    test_ds = TranslationIdDataset(ko_test, zh_test)
    print(f"训练集: {len(train_ds)} | 测试集: {len(test_ds)}", flush=True)

    pin_memory = torch.cuda.is_available()
    persistent_workers = bool(args.num_workers and args.num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"使用设备: {device} | GPU 数量: {torch.cuda.device_count()}", flush=True)
    else:
        print(f"使用设备: {device}", flush=True)

    train_model_transformer(train_loader, test_loader, ko_sp, zh_sp, device, args)


if __name__ == "__main__":
    main()
