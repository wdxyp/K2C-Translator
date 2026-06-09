import argparse
import os
import re
import shutil
import tempfile
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
    for arrow in ("→", "⇒", "➡", "⟶", "⟹", "➔", "➜", "➝", "➞", "➟", "➠"):
        s = s.replace(arrow, "->")

    for q in ("“", "”", "„", "‟", "＂"):
        s = s.replace(q, '"')

    s = unicodedata.normalize("NFKC", s)
    return s


def _separate_punct_boundaries(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""

    s = text
    s = s.replace("->", " -> ")
    s = s.replace("...", " ... ")

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
    s = _normalize_text(sentence)
    s = s.replace("\n", " ")

    kept: list[str] = []
    for ch in s:
        if ch.isspace():
            kept.append(" ")
            continue
        if "0" <= ch <= "9" or "A" <= ch <= "Z" or "a" <= ch <= "z":
            kept.append(ch)
            continue
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            kept.append(ch)
            continue
        if 0x4E00 <= code <= 0x9FFF:
            kept.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat and cat[0] == "P":
            kept.append(ch)
            continue
        if cat in ("Sm", "Sc"):
            kept.append(ch)
            continue

    s = "".join(kept)
    s = _separate_punct_boundaries(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_excel_files(inputs: list[str]) -> list[str]:
    file_paths: list[str] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() == ".xlsx":
            file_paths.append(str(p))
            continue
        if p.is_dir():
            file_paths.extend(str(x) for x in p.rglob("*.xlsx"))
            continue
        if "*" in raw or "?" in raw:
            file_paths.extend(str(x) for x in Path().glob(raw))
            continue
    file_paths = sorted(set(file_paths))
    return file_paths


def read_corpus(file_paths: list[str]) -> tuple[list[str], list[str]]:
    all_korean: list[str] = []
    all_chinese: list[str] = []
    for path in file_paths:
        try:
            wb = openpyxl.load_workbook(path, data_only=True)
            ws = wb.active
            for row in ws.iter_rows(min_row=2, values_only=True):
                if len(row) >= 4:
                    ko, zh = row[1], row[3]
                    if ko and zh:
                        all_korean.append(str(ko))
                        all_chinese.append(str(zh))
            wb.close()
        except Exception as e:
            print(f"读取文件 {path} 出错: {e}")
    return all_korean, all_chinese


class TranslationIdDataset(Dataset):
    def __init__(self, ko_ids: list[list[int]], zh_ids: list[list[int]]):
        self.ko_ids = ko_ids
        self.zh_ids = zh_ids

    def __len__(self):
        return len(self.ko_ids)

    def __getitem__(self, idx):
        return torch.LongTensor(self.ko_ids[idx]), torch.LongTensor(self.zh_ids[idx])


def collate_fn(batch):
    ko_batch, zh_batch = zip(*batch)
    ko_lens = [int(x.numel()) for x in ko_batch]
    zh_lens = [int(x.numel()) for x in zh_batch]
    ko_padded = nn.utils.rnn.pad_sequence(ko_batch, padding_value=0)
    zh_padded = nn.utils.rnn.pad_sequence(zh_batch, padding_value=0)
    return ko_padded, ko_lens, zh_padded, zh_lens


def train_sentencepiece_models(
    ko_sents: list[str],
    zh_sents: list[str],
    model_folder: str,
    vocab_size: int,
    ko_prefix_name: str = "spm_ko_v3_2_1",
    zh_prefix_name: str = "spm_zh_v3_2_1",
    num_threads: int | None = None,
    character_coverage: float = 0.9995,
) -> tuple[str, str]:
    model_folder = str(model_folder)
    ko_prefix = os.path.join(model_folder, ko_prefix_name)
    zh_prefix = os.path.join(model_folder, zh_prefix_name)
    ko_model_path = f"{ko_prefix}.model"
    zh_model_path = f"{zh_prefix}.model"

    if os.path.exists(ko_model_path) and os.path.exists(zh_model_path):
        print(f"检测到已存在 SentencePiece 模型，直接复用: {model_folder}", flush=True)
        return ko_model_path, zh_model_path

    if spm is None:
        raise RuntimeError(f"未安装 sentencepiece，错误: {spm_import_error}")

    os.makedirs(model_folder, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ko_txt = os.path.join(tmpdir, "ko.txt")
        zh_txt = os.path.join(tmpdir, "zh.txt")

        print("正在写入 SentencePiece 训练文本（ko/zh）...", flush=True)
        with open(ko_txt, "w", encoding="utf-8") as f:
            for s in ko_sents:
                f.write((s or "").replace("\n", " ") + "\n")

        with open(zh_txt, "w", encoding="utf-8") as f:
            for s in zh_sents:
                f.write((s or "").replace("\n", " ") + "\n")
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

        nt = int(num_threads) if isinstance(num_threads, int) and num_threads > 0 else max(1, min(8, os.cpu_count() or 2))

        print(f"开始训练 SentencePiece(ko) vocab_size={vocab_size} ...", flush=True)
        hb = _start_heartbeat("ko")
        spm.SentencePieceTrainer.Train(
            input=ko_txt,
            model_prefix=ko_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            hard_vocab_limit=False,
            character_coverage=float(character_coverage),
            num_threads=int(nt),
        )
        hb.set()
        print("SentencePiece(ko) 完成。", flush=True)

        print(f"开始训练 SentencePiece(zh) vocab_size={vocab_size} ...", flush=True)
        hb = _start_heartbeat("zh")
        spm.SentencePieceTrainer.Train(
            input=zh_txt,
            model_prefix=zh_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            hard_vocab_limit=False,
            character_coverage=float(character_coverage),
            num_threads=int(nt),
        )
        hb.set()
        print("SentencePiece(zh) 完成。", flush=True)

    return ko_model_path, zh_model_path


def encode_with_spm(sentences: list[str], sp, max_len: int | None = None) -> tuple[list[list[int]], int]:
    bos = int(sp.bos_id())
    eos = int(sp.eos_id())
    all_ids: list[list[int]] = []
    truncated = 0
    for s in sentences:
        piece_ids = sp.encode(s, out_type=int)
        if isinstance(max_len, int) and max_len > 2 and len(piece_ids) > (max_len - 2):
            piece_ids = piece_ids[: max_len - 2]
            truncated += 1
        all_ids.append([bos] + piece_ids + [eos])
    return all_ids, truncated


class Attention(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_dropout = float(dropout) if int(n_layers) > 1 else 0.0
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, bidirectional=True, dropout=rnn_dropout)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        if src_len is None:
            src_len = (src != 0).long().sum(dim=0).to(device="cpu")
        elif isinstance(src_len, list):
            src_len = torch.tensor(src_len, dtype=torch.long, device="cpu")
        else:
            src_len = src_len.to(device="cpu", dtype=torch.long)
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float, attention):
        super().__init__()
        self.output_dim = output_dim
        self.n_layers = int(n_layers)
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_dropout = float(dropout) if int(n_layers) > 1 else 0.0
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim, n_layers, dropout=rnn_dropout)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        hidden_for_rnn = hidden.unsqueeze(0)
        if self.n_layers > 1:
            hidden_for_rnn = hidden_for_rnn.repeat(self.n_layers, 1, 1)
        output, hidden = self.rnn(rnn_input, hidden_for_rnn)
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden[-1]


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, trg, teacher_forcing_ratio: float = 0.5):
        dev = src.device
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size, device=dev)
        src_len = (src != 0).long().sum(dim=0).to(device="cpu")
        encoder_outputs, hidden = self.encoder(src, src_len)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = torch.rand(1, device=dev).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs


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


def train_model_bpe(
    train_loader,
    test_loader,
    ko_sp,
    zh_sp,
    device: torch.device,
    model_folder: str,
    max_epochs: int,
    lr: float,
    use_multi_gpu: bool,
    use_amp: bool,
    grad_accum_steps: int,
    log_every: int,
    show_samples: int,
    teacher_forcing_ratio: float,
    dropout: float,
    weight_decay: float,
    lr_factor: float,
    lr_patience: int,
    lr_min: float,
    early_stop_patience: int,
    resume_mode: str,
):
    input_dim = ko_sp.get_piece_size()
    output_dim = zh_sp.get_piece_size()

    enc_emb_dim = 256
    dec_emb_dim = 256
    hid_dim = 512
    n_layers = 2
    dropout = float(dropout)

    attn = Attention(hid_dim)
    enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, dropout).to(device)
    dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dropout, attn).to(device)
    model = Seq2Seq(enc, dec, device).to(device)

    if device.type == "cuda" and use_multi_gpu and torch.cuda.device_count() >= 2:
        model = nn.DataParallel(model, dim=1)

    optimizer = optim.Adam(unwrap_model(model).parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(lr_factor),
        patience=int(lr_patience),
        min_lr=float(lr_min),
    )

    os.makedirs(model_folder, exist_ok=True)
    best_state_path = os.path.join(model_folder, "best_model_v3_2_1_bpe_attn.pth")
    best_ckpt_path = os.path.join(model_folder, "best_model_v3_2_1_bpe_attn.ckpt")

    start_epoch = 0
    best_test_loss = float("inf")
    patience = int(early_stop_patience)
    no_improve = 0

    if hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and use_amp))

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            if resume_mode == "model" and os.path.exists(best_state_path):
                load_state_dict_safely(model, torch.load(best_state_path, map_location="cpu"))
            else:
                load_state_dict_safely(model, ckpt["model_state_dict"])
            if resume_mode == "full":
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if "scaler_state_dict" in ckpt and scaler.is_enabled():
                    scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = 0 if resume_mode == "model" else int(ckpt.get("epoch", 0))
            best_test_loss = float(ckpt.get("best_test_loss", best_test_loss))
            no_improve = 0 if resume_mode == "model" else int(ckpt.get("no_improve", 0))
            print(f"检测到已存在的 BPE Checkpoint，继续训练: {best_ckpt_path}")
            if resume_mode == "model":
                print("Resume Mode = model：从 best 权重开始精调（优化器/调度器重置，epoch 从 1 重新计）", flush=True)

    grad_accum_steps = max(1, int(grad_accum_steps))
    log_every = max(1, int(log_every))
    show_samples = max(0, int(show_samples))
    teacher_forcing_ratio = float(teacher_forcing_ratio)
    teacher_forcing_ratio = max(0.0, min(1.0, teacher_forcing_ratio))
    pad_id = 0
    eos_id = int(zh_sp.eos_id())
    unk_id = int(zh_sp.unk_id())

    for epoch in range(start_epoch, max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs} 开始...", flush=True)
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, (src, src_len, trg, trg_len) in enumerate(train_loader):
            src = src.to(device, non_blocking=True)
            trg = trg.to(device, non_blocking=True)

            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=use_amp)
                if (device.type == "cuda" and hasattr(torch, "amp"))
                else (torch.cuda.amp.autocast(enabled=use_amp) if device.type == "cuda" else nullcontext())
            )
            with autocast_ctx:
                output = model(src, src_len, trg, teacher_forcing_ratio=teacher_forcing_ratio)
                vocab_size = output.shape[-1]
                output = output[1:].reshape(-1, vocab_size)
                trg_flat = trg[1:].reshape(-1)
                loss = criterion(output, trg_flat) / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += float(loss.item()) * grad_accum_steps

            if (i + 1) % grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (i + 1) % log_every == 0:
                current_loss = float(loss.item()) * grad_accum_steps
                print(f"Epoch: {epoch+1:02}, Batch: {i+1}/{len(train_loader)}, Loss: {current_loss:.4f}", flush=True)

        model.eval()
        test_loss = 0.0
        correct_tokens = 0
        total_tokens = 0
        unk_tokens = 0
        exact_match = 0
        total_seqs = 0
        pred_len_sum = 0
        trg_len_sum = 0
        shown = 0
        with torch.no_grad():
            for src, src_len, trg, trg_len in test_loader:
                src = src.to(device, non_blocking=True)
                trg = trg.to(device, non_blocking=True)
                autocast_ctx = (
                    torch.amp.autocast("cuda", enabled=use_amp)
                    if (device.type == "cuda" and hasattr(torch, "amp"))
                    else (torch.cuda.amp.autocast(enabled=use_amp) if device.type == "cuda" else nullcontext())
                )
                with autocast_ctx:
                    output = model(src, src_len, trg, 0.0)
                    vocab_size = output.shape[-1]
                    output = output[1:].reshape(-1, vocab_size)
                    trg_flat = trg[1:].reshape(-1)
                    loss = criterion(output, trg_flat)
                test_loss += float(loss.item())

                pred = output.reshape(-1, vocab_size).argmax(dim=1).reshape(trg.shape[0] - 1, trg.shape[1])
                trg_tokens = trg[1:]
                mask = trg_tokens != pad_id
                correct = (pred == trg_tokens) & mask
                correct_tokens += int(correct.sum().item())
                total_tokens += int(mask.sum().item())
                unk_tokens += int(((pred == unk_id) & mask).sum().item())
                exact_match += int((correct.sum(dim=0) == mask.sum(dim=0)).sum().item())
                total_seqs += int(trg.shape[1])

                if show_samples and shown < show_samples:
                    bsz = int(trg.shape[1])
                    for bi in range(bsz):
                        if shown >= show_samples:
                            break
                        pred_seq = pred[:, bi].tolist()
                        trg_seq = trg_tokens[:, bi].tolist()
                        if eos_id in pred_seq:
                            pred_seq = pred_seq[: pred_seq.index(eos_id)]
                        if eos_id in trg_seq:
                            trg_seq = trg_seq[: trg_seq.index(eos_id)]
                        pred_text = zh_sp.decode(pred_seq) if pred_seq else ""
                        trg_text = zh_sp.decode(trg_seq) if trg_seq else ""
                        print(f"SAMPLE_PRED: {pred_text}", flush=True)
                        print(f"SAMPLE_GOLD: {trg_text}", flush=True)
                        shown += 1

                for bi in range(int(trg.shape[1])):
                    trg_seq = trg_tokens[:, bi]
                    trg_valid = int((trg_seq != pad_id).sum().item())
                    trg_len_sum += trg_valid
                    pred_seq = pred[:, bi]
                    if (pred_seq == eos_id).any():
                        pred_len = int((pred_seq == eos_id).float().argmax(dim=0).item())
                        pred_len_sum += max(1, pred_len)
                    else:
                        pred_len_sum += max(1, trg_valid)

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        avg_test_loss = test_loss / max(1, len(test_loader))
        lr_now = float(optimizer.param_groups[0]["lr"])
        ppl = math.exp(min(20.0, avg_test_loss))
        acc = (correct_tokens / total_tokens) if total_tokens else 0.0
        unk_rate = (unk_tokens / total_tokens) if total_tokens else 0.0
        em = (exact_match / total_seqs) if total_seqs else 0.0
        len_ratio = (pred_len_sum / trg_len_sum) if trg_len_sum else 0.0
        print(
            f"Epoch: {epoch+1:02} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | PPL: {ppl:.2f} | Acc: {acc:.3f} | EM: {em:.3f} | UNK: {unk_rate:.3f} | LenRatio: {len_ratio:.3f} | LR: {lr_now:.6f}",
            flush=True,
        )

        scheduler.step(avg_test_loss)

        state = {
            "epoch": epoch + 1,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "best_test_loss": best_test_loss,
            "no_improve": no_improve,
            "ko_spm_model": os.path.join(model_folder, "spm_ko_v3_2_1.model"),
            "zh_spm_model": os.path.join(model_folder, "spm_zh_v3_2_1.model"),
            "multi_gpu": bool(isinstance(model, nn.DataParallel)),
        }

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            no_improve = 0
            torch.save(unwrap_model(model).state_dict(), best_state_path)
            state["best_test_loss"] = best_test_loss
            state["no_improve"] = no_improve
            torch.save(state, best_ckpt_path)
            print("  -> 测试损失下降，保存最佳模型与训练状态！", flush=True)
        else:
            no_improve += 1
            state["no_improve"] = no_improve
            torch.save(state, best_ckpt_path)
            if no_improve >= patience:
                print("早停触发，训练结束。", flush=True)
                break

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    torch.save(unwrap_model(model).state_dict(), os.path.join(model_folder, f"model_v3_2_1_{timestamp}.pth"))
    print("训练全过程完成！", flush=True)


def resolve_default_model_dir() -> str:
    if Path("/kaggle").exists():
        return str(Path("/kaggle/working/Translate Model/v3_2_1"))
    return "Translate Model/v3_2_1"


def _find_first_under(root: Path, filename: str) -> str | None:
    try:
        for p in root.rglob(filename):
            if p.is_file():
                return str(p)
    except Exception:
        return None
    return None


def _maybe_copy_resume_artifacts_from_kaggle_input(model_dir: str) -> None:
    if not Path("/kaggle/input").exists():
        return

    dst_dir = Path(model_dir)
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    wanted = [
        "best_model_v3_2_1_bpe_attn.pth",
        "best_model_v3_2_1_bpe_attn.ckpt",
        "spm_ko_v3_2_1.model",
        "spm_ko_v3_2_1.vocab",
        "spm_zh_v3_2_1.model",
        "spm_zh_v3_2_1.vocab",
    ]

    input_root = Path("/kaggle/input")
    copied = 0
    for name in wanted:
        dst = dst_dir / name
        if dst.exists():
            continue
        src = _find_first_under(input_root, name)
        if not src:
            continue
        try:
            shutil.copy2(src, str(dst))
            copied += 1
        except Exception:
            continue

    if copied:
        print(f"从 /kaggle/input 复制恢复文件到 model_dir: copied={copied}", flush=True)


def set_seed(seed: int) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(Path("/kaggle/input"))],
    )
    parser.add_argument("--model_dir", default=resolve_default_model_dir())
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--max_ko_len", type=int, default=128)
    parser.add_argument("--max_zh_len", type=int, default=96)
    parser.add_argument("--spm_threads", type=int, default=0)
    parser.add_argument("--spm_character_coverage", type=float, default=0.9995)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--teacher_forcing", type=float, default=0.3)
    parser.add_argument("--lr_factor", type=float, default=0.7)
    parser.add_argument("--lr_patience", type=int, default=4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--resume_mode", choices=["full", "model"], default="model")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=min(4, (os.cpu_count() or 2)))
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_multi_gpu", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--show_samples", type=int, default=0)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"忽略额外参数: {unknown}", flush=True)

    if spm is None:
        raise RuntimeError(f"缺少 sentencepiece，错误: {spm_import_error}")

    _maybe_copy_resume_artifacts_from_kaggle_input(str(args.model_dir))

    set_seed(args.seed)

    file_paths = find_excel_files(args.inputs)
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
    ko_model_path, zh_model_path = train_sentencepiece_models(
        ko_sents,
        zh_sents,
        args.model_dir,
        vocab_size=int(args.vocab_size),
        num_threads=(int(args.spm_threads) if int(args.spm_threads) > 0 else None),
        character_coverage=float(args.spm_character_coverage),
    )
    ko_sp = spm.SentencePieceProcessor(model_file=ko_model_path)
    zh_sp = spm.SentencePieceProcessor(model_file=zh_model_path)

    ko_ids, ko_trunc = encode_with_spm(ko_sents, ko_sp, max_len=int(args.max_ko_len))
    zh_ids, zh_trunc = encode_with_spm(zh_sents, zh_sp, max_len=int(args.max_zh_len))
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        gpu_count = torch.cuda.device_count()
        print(f"使用设备: {device} | GPU 数量: {gpu_count}", flush=True)
    else:
        print(f"使用设备: {device}", flush=True)

    train_model_bpe(
        train_loader=train_loader,
        test_loader=test_loader,
        ko_sp=ko_sp,
        zh_sp=zh_sp,
        device=device,
        model_folder=str(args.model_dir),
        max_epochs=int(args.epochs),
        lr=float(args.lr),
        use_multi_gpu=(not args.no_multi_gpu),
        use_amp=(not args.no_amp),
        grad_accum_steps=int(args.grad_accum_steps),
        log_every=int(args.log_every),
        show_samples=int(args.show_samples),
        teacher_forcing_ratio=float(args.teacher_forcing),
        dropout=float(args.dropout),
        weight_decay=float(args.weight_decay),
        lr_factor=float(args.lr_factor),
        lr_patience=int(args.lr_patience),
        lr_min=float(args.lr_min),
        early_stop_patience=int(args.early_stop_patience),
        resume_mode=str(args.resume_mode),
    )


if __name__ == "__main__":
    main()

