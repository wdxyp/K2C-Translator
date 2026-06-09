import os
import re
import pickle
import unicodedata
from datetime import datetime
from collections import Counter
import glob
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import openpyxl

MAX_KO_LEN = 128
MAX_ZH_LEN = 96
CKPT_NAME = "checkpoint_v3_0_1_attn.ckpt"
BEST_MODEL_NAME = "best_model_v3_0_1_attn.pth"
BEST_KO_VOCAB_NAME = "best_ko_vocab_v3_0_1_attn.pkl"
BEST_ZH_VOCAB_NAME = "best_zh_vocab_v3_0_1_attn.pkl"

DEFAULT_LR = 0.0002
DEFAULT_TEACHER_FORCING = 0.3
DEFAULT_DROPOUT = 0.3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_TEST_EVERY = 1


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

    kept = []
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


def _build_tokenizers():
    try:
        from konlpy.tag import Okt

        okt = Okt()

        def tok_ko(s: str):
            return okt.morphs(s)

        ko_name = "Okt"
    except Exception as e:
        raise RuntimeError(
            "Kaggle 环境未能导入 konlpy/Okt。请先在 Notebook 单独执行安装：\n"
            "!pip -q install konlpy jpype1\n"
            "如果仍失败，请确认 Kaggle 已启用 Internet，并重启 Kernel。\n"
            f"原始错误: {e}"
        )

    try:
        import jieba

        def tok_zh(s: str):
            return jieba.lcut(s)

        zh_name = "jieba"
    except Exception as e:
        raise RuntimeError(
            "Kaggle 环境未能导入 jieba。请先在 Notebook 单独执行安装：\n"
            "!pip -q install jieba\n"
            "如果仍失败，请确认 Kaggle 已启用 Internet，并重启 Kernel。\n"
            f"原始错误: {e}"
        )

    return tok_ko, tok_zh, ko_name, zh_name


def read_corpus(file_paths):
    all_korean = []
    all_chinese = []
    for path in file_paths:
        try:
            wb = openpyxl.load_workbook(path, data_only=True)
            ws = wb.active
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row is None or len(row) < 4:
                    continue
                ko, zh = row[1], row[3]
                if ko and zh:
                    all_korean.append(str(ko))
                    all_chinese.append(str(zh))
            wb.close()
        except Exception as e:
            print(f"读取文件出错: {path} / {e}")
    return all_korean, all_chinese


def build_vocab(tokenized_sentences, min_freq=2, max_size=30000):
    counter = Counter()
    for tokens in tokenized_sentences:
        counter.update(tokens)

    most_common = counter.most_common(max_size)
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for word, freq in most_common:
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


class TranslationDataset(Dataset):
    def __init__(self, ko_tokens, zh_tokens, ko_vocab, zh_vocab, max_ko_len: int | None = None, max_zh_len: int | None = None):
        self.ko_data = ko_tokens
        self.zh_data = zh_tokens
        self.ko_vocab = ko_vocab
        self.zh_vocab = zh_vocab
        self.max_ko_len = int(max_ko_len) if max_ko_len is not None else None
        self.max_zh_len = int(max_zh_len) if max_zh_len is not None else None
        if self.max_ko_len is not None and self.max_ko_len < 2:
            self.max_ko_len = 2
        if self.max_zh_len is not None and self.max_zh_len < 2:
            self.max_zh_len = 2
        self.ko_lens = [min(len(x) + 2, self.max_ko_len) if self.max_ko_len is not None else (len(x) + 2) for x in self.ko_data]
        self.zh_lens = [min(len(x) + 2, self.max_zh_len) if self.max_zh_len is not None else (len(x) + 2) for x in self.zh_data]

    def __len__(self):
        return len(self.ko_data)

    def __getitem__(self, idx):
        ko_eos = self.ko_vocab["<eos>"]
        ko_sos = self.ko_vocab["<sos>"]
        zh_eos = self.zh_vocab["<eos>"]
        zh_sos = self.zh_vocab["<sos>"]

        ko_body = [self.ko_vocab.get(token, self.ko_vocab["<unk>"]) for token in self.ko_data[idx]]
        zh_body = [self.zh_vocab.get(token, self.zh_vocab["<unk>"]) for token in self.zh_data[idx]]

        if self.max_ko_len is not None:
            max_body = max(0, self.max_ko_len - 2)
            if len(ko_body) > max_body:
                ko_body = ko_body[:max_body]
        if self.max_zh_len is not None:
            max_body = max(0, self.max_zh_len - 2)
            if len(zh_body) > max_body:
                zh_body = zh_body[:max_body]

        ko_idx = [ko_sos] + ko_body + [ko_eos]
        zh_idx = [zh_sos] + zh_body + [zh_eos]
        return torch.LongTensor(ko_idx), torch.LongTensor(zh_idx)


def collate_fn(batch):
    ko_batch, zh_batch = zip(*batch)
    ko_padded = nn.utils.rnn.pad_sequence(ko_batch, padding_value=0)
    zh_padded = nn.utils.rnn.pad_sequence(zh_batch, padding_value=0)
    return ko_padded, zh_padded


class BucketBatchSampler:
    def __init__(self, lengths, batch_size: int, bucket_size: int = 2048, drop_last: bool = False, seed: int = 42):
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.bucket_size = int(bucket_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = torch.argsort(torch.tensor(self.lengths), stable=True).tolist()
        bucket_size = max(self.batch_size, self.bucket_size)
        buckets = [indices[i : i + bucket_size] for i in range(0, len(indices), bucket_size)]
        for b in buckets:
            perm = torch.randperm(len(b), generator=g).tolist()
            b[:] = [b[j] for j in perm]

        batches = []
        for b in buckets:
            for i in range(0, len(b), self.batch_size):
                batch = b[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        perm_batches = torch.randperm(len(batches), generator=g).tolist()
        for j in perm_batches:
            yield batches[j]

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class Attention(nn.Module):
    def __init__(self, hid_dim):
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
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_dropout = float(dropout) if int(n_layers) > 1 else 0.0
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, bidirectional=True, dropout=rnn_dropout)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src_len = (src != 0).sum(dim=0)
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
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

    def forward(self, src, trg, teacher_forcing_ratio=0.5, criterion_none=None):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        dev = src.device
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :]

        if criterion_none is not None:
            loss_sum = torch.zeros((batch_size,), device=dev)
            tok_count = torch.zeros((batch_size,), device=dev)
            pad_id = 0
            for t in range(1, trg_len):
                output, hidden = self.decoder(input, hidden, encoder_outputs)
                target = trg[t]
                loss_t = criterion_none(output, target)
                mask = (target != pad_id).float()
                loss_sum = loss_sum + (loss_t * mask)
                tok_count = tok_count + mask
                teacher_force = torch.rand(1, device=dev).item() < teacher_forcing_ratio
                top1 = output.argmax(1)
                input = target if teacher_force else top1
            loss_vec = loss_sum / torch.clamp(tok_count, min=1)
            return loss_vec.unsqueeze(0)

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size, device=dev)
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = torch.rand(1, device=dev).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs


def _make_grad_scaler(device: torch.device):
    if device.type != "cuda":
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def _autocast(device: torch.device):
    if device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _is_oom_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def _empty_cuda_cache():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _backward_loss(loss: torch.Tensor, scaler, optimizer):
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        scaler.scale(loss).backward()
    else:
        loss.backward()


def _optimizer_step(optimizer, scaler):
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1)
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def _microbatch_fallback_step(
    model,
    src: torch.Tensor,
    trg: torch.Tensor,
    teacher_forcing_ratio: float,
    criterion_none,
    device: torch.device,
    optimizer,
    scaler,
    base_div: int,
):
    total_bs = int(src.shape[1])

    def run_slice(src_s: torch.Tensor, trg_s: torch.Tensor, weight: float):
        nonlocal device, optimizer, scaler, base_div
        try:
            with _autocast(device):
                loss_mat = model(src_s, trg_s, teacher_forcing_ratio=teacher_forcing_ratio, criterion_none=criterion_none)
                loss = (loss_mat.mean() * float(weight)) / max(1, int(base_div))
            _backward_loss(loss, scaler, optimizer)
            return
        except Exception as e:
            if not _is_oom_error(e):
                raise
            _empty_cuda_cache()
            bs = int(src_s.shape[1])
            if bs <= 1:
                raise
            mid = bs // 2
            run_slice(src_s[:, :mid], trg_s[:, :mid], weight * (mid / bs))
            run_slice(src_s[:, mid:], trg_s[:, mid:], weight * ((bs - mid) / bs))

    optimizer.zero_grad(set_to_none=True)
    try:
        run_slice(src, trg, 1.0)
        _optimizer_step(optimizer, scaler)
    finally:
        optimizer.zero_grad(set_to_none=True)


def train_model(train_loader, test_loader, ko_vocab, zh_vocab, device, model_folder, max_epochs=50, grad_accum_steps=4, teacher_forcing_ratio=0.5):
    input_dim = len(ko_vocab)
    output_dim = len(zh_vocab)
    enc_emb_dim = 256
    dec_emb_dim = 256
    hid_dim = 512
    n_layers = int(os.environ.get("N_LAYERS", "1"))
    dropout = float(os.environ.get("DROPOUT", str(DEFAULT_DROPOUT)))
    lr = float(os.environ.get("LR", str(DEFAULT_LR)))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", str(DEFAULT_WEIGHT_DECAY)))
    test_every = int(os.environ.get("TEST_EVERY", str(DEFAULT_TEST_EVERY)))
    resume_from_best = os.environ.get("RESUME_FROM_BEST", "0").strip() in ("1", "true", "True", "yes", "YES")

    if teacher_forcing_ratio is None:
        teacher_forcing_ratio = float(os.environ.get("TEACHER_FORCING", str(DEFAULT_TEACHER_FORCING)))
    else:
        teacher_forcing_ratio = float(teacher_forcing_ratio)

    attn = Attention(hid_dim)
    enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, dropout)
    dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dropout, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    if torch.cuda.device_count() >= 2 and device.type == "cuda":
        primary = int(device.index) if device.index is not None else 0
        secondary = 0 if primary != 0 else 1
        device_ids = [primary, secondary]
        output_device = primary
        model = nn.DataParallel(model, device_ids=device_ids, output_device=output_device, dim=1)
        print(
            "使用多GPU DataParallel, device_count=",
            torch.cuda.device_count(),
            "device_ids=",
            device_ids,
            "output_device=",
            output_device,
        )
    else:
        print("使用单GPU/CPU, device_count=", torch.cuda.device_count())

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_none = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_test_loss = float("inf")
    patience = 10
    no_improve = 0

    best_model_path = os.path.join(model_folder, BEST_MODEL_NAME)
    best_ko_vocab_path = os.path.join(model_folder, BEST_KO_VOCAB_NAME)
    best_zh_vocab_path = os.path.join(model_folder, BEST_ZH_VOCAB_NAME)
    ckpt_path = os.path.join(model_folder, CKPT_NAME)

    scaler = _make_grad_scaler(device)
    grad_accum_steps = max(1, int(grad_accum_steps))

    start_epoch = 0
    if resume_from_best and os.path.exists(best_model_path):
        try:
            state_dict = torch.load(best_model_path, map_location=device)
            (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(state_dict)
            start_epoch = 0
            best_test_loss = float("inf")
            no_improve = 0
            print(f"RESUME_FROM_BEST=1，加载 best 模型权重: {best_model_path}，从 epoch=0 继续训练")
        except Exception as e:
            print(f"best 模型加载失败，将尝试加载 checkpoint: {e}")

    if (not resume_from_best) and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict):
                model_state = ckpt.get("model_state_dict")
                if model_state is not None:
                    (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(model_state)
                opt_state = ckpt.get("optimizer_state_dict")
                if opt_state is not None:
                    optimizer.load_state_dict(opt_state)
                sched_state = ckpt.get("scheduler_state_dict")
                if sched_state is not None:
                    scheduler.load_state_dict(sched_state)
                scaler_state = ckpt.get("scaler_state_dict")
                if scaler is not None and scaler_state is not None:
                    try:
                        scaler.load_state_dict(scaler_state)
                    except Exception:
                        pass
                start_epoch = int(ckpt.get("epoch", 0))
                best_test_loss = float(ckpt.get("best_test_loss", best_test_loss))
                no_improve = int(ckpt.get("no_improve", no_improve))
                print(f"检测到 checkpoint，继续训练: {ckpt_path} (start_epoch={start_epoch})")
        except Exception as e:
            print(f"checkpoint 加载失败，将从头开始训练: {e}")

    print(
        "训练参数:",
        f"lr={lr}",
        f"weight_decay={weight_decay}",
        f"dropout={dropout}",
        f"teacher_forcing={teacher_forcing_ratio}",
        f"n_layers={n_layers}",
        f"grad_accum_steps={grad_accum_steps}",
        f"max_ko_len={MAX_KO_LEN}",
        f"max_zh_len={MAX_ZH_LEN}",
        flush=True,
    )

    for epoch in range(start_epoch, max_epochs):
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        if device.type == "cuda":
            for d in range(torch.cuda.device_count()):
                try:
                    torch.cuda.reset_peak_memory_stats(d)
                except Exception:
                    pass
        epoch_t0 = time.time()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, (src, trg) in enumerate(train_loader):
            src = src.to(device, non_blocking=True)
            trg = trg.to(device, non_blocking=True)

            try:
                with _autocast(device):
                    loss_mat = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio, criterion_none=criterion_none)
                    loss = loss_mat.mean() / grad_accum_steps
                _backward_loss(loss, scaler, optimizer)
            except Exception as e:
                if not _is_oom_error(e):
                    raise
                print(
                    f"OOM at epoch={epoch+1} batch={i+1}/{len(train_loader)} "
                    f"src_shape={tuple(src.shape)} trg_shape={tuple(trg.shape)}; "
                    "trying microbatch fallback...",
                    flush=True,
                )
                _empty_cuda_cache()
                _microbatch_fallback_step(
                    model=model,
                    src=src,
                    trg=trg,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    criterion_none=criterion_none,
                    device=device,
                    optimizer=optimizer,
                    scaler=scaler,
                    base_div=1,
                )
                continue

            if (i + 1) % grad_accum_steps == 0:
                _optimizer_step(optimizer, scaler)
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item()) * grad_accum_steps
            if (i + 1) % 100 == 0:
                msg = f"Epoch: {epoch+1:02} Batch: {i+1}/{len(train_loader)} Loss: {(loss.item()*grad_accum_steps):.4f}"
                try:
                    cur_lr = optimizer.param_groups[0].get("lr")
                    if cur_lr is not None:
                        msg += f" lr={cur_lr:.6g}"
                except Exception:
                    pass
                if device.type == "cuda":
                    try:
                        m0 = torch.cuda.max_memory_allocated(0) / (1024**3)
                        m1 = torch.cuda.max_memory_allocated(1) / (1024**3) if torch.cuda.device_count() > 1 else None
                        msg += f" peak_mem_gb=[{m0:.2f}"
                        if m1 is not None:
                            msg += f", {m1:.2f}"
                        msg += "]"
                    except Exception:
                        pass
                print(msg, flush=True)

        avg_train_loss = epoch_loss / max(1, len(train_loader))

        avg_test_loss = None
        if test_every <= 1 or ((epoch + 1) % test_every == 0):
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for src, trg in test_loader:
                    src = src.to(device, non_blocking=True)
                    trg = trg.to(device, non_blocking=True)
                    with _autocast(device):
                        loss_mat = model(src, trg, teacher_forcing_ratio=0.0, criterion_none=criterion_none)
                        loss = loss_mat.mean()
                    test_loss += float(loss.item())
            avg_test_loss = test_loss / max(1, len(test_loader))
            scheduler.step(avg_test_loss)

        epoch_s = time.time() - epoch_t0
        if avg_test_loss is not None:
            print(f"Epoch: {epoch+1:02} Train Loss: {avg_train_loss:.4f} Test Loss: {avg_test_loss:.4f} time_s={epoch_s:.1f}", flush=True)
        else:
            print(f"Epoch: {epoch+1:02} Train Loss: {avg_train_loss:.4f} time_s={epoch_s:.1f}", flush=True)

        if avg_test_loss is not None:
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                no_improve = 0
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, best_model_path)
                with open(best_ko_vocab_path, "wb") as f:
                    pickle.dump(ko_vocab, f)
                with open(best_zh_vocab_path, "wb") as f:
                    pickle.dump(zh_vocab, f)
                print("保存最佳模型", flush=True)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("早停触发，训练结束", flush=True)
                    break

        try:
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_test_loss": best_test_loss,
                "no_improve": no_improve,
                "max_ko_len": MAX_KO_LEN,
                "max_zh_len": MAX_ZH_LEN,
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "teacher_forcing_ratio": teacher_forcing_ratio,
                "n_layers": n_layers,
                "grad_accum_steps": grad_accum_steps,
            }
            if scaler is not None:
                try:
                    ckpt["scaler_state_dict"] = scaler.state_dict()
                except Exception:
                    pass
            torch.save(ckpt, ckpt_path)
        except Exception as e:
            print(f"checkpoint 保存失败: {e}")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    final_model_path = os.path.join(model_folder, f"model_v3_0_1_{timestamp}.pth")
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, final_model_path)
    with open(os.path.join(model_folder, f"ko_vocab_v3_0_1_{timestamp}.pkl"), "wb") as f:
        pickle.dump(ko_vocab, f)
    with open(os.path.join(model_folder, f"zh_vocab_v3_0_1_{timestamp}.pkl"), "wb") as f:
        pickle.dump(zh_vocab, f)
    print("训练完成")


def _find_xlsx_files(search_root: str):
    patterns = [
        os.path.join(search_root, "*.xlsx"),
        os.path.join(search_root, "*", "*.xlsx"),
        os.path.join(search_root, "*", "*", "*.xlsx"),
        os.path.join(search_root, "*", "*", "*", "*.xlsx"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    dedup = []
    seen = set()
    for f in files:
        if f not in seen:
            seen.add(f)
            dedup.append(f)
    return dedup


def _find_first_checkpoint_dir(search_root: str) -> str | None:
    patterns = [
        os.path.join(search_root, CKPT_NAME),
        os.path.join(search_root, "*", CKPT_NAME),
        os.path.join(search_root, "*", "*", CKPT_NAME),
        os.path.join(search_root, "*", "*", "*", CKPT_NAME),
        os.path.join(search_root, "*", "*", "*", "*", CKPT_NAME),
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        return None
    best = max(candidates, key=os.path.getmtime)
    return os.path.dirname(best)


def _copy_if_exists(src_path: str, dst_path: str):
    if not src_path or not os.path.isfile(src_path):
        return
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        import shutil

        shutil.copy2(src_path, dst_path)
    except Exception:
        try:
            import shutil

            shutil.copy(src_path, dst_path)
        except Exception:
            pass


def _restore_resume_files(output_dir: str, resume_dir: str | None):
    if not resume_dir or not os.path.isdir(resume_dir):
        return
    _copy_if_exists(os.path.join(resume_dir, CKPT_NAME), os.path.join(output_dir, CKPT_NAME))
    _copy_if_exists(os.path.join(resume_dir, BEST_MODEL_NAME), os.path.join(output_dir, BEST_MODEL_NAME))
    _copy_if_exists(os.path.join(resume_dir, BEST_KO_VOCAB_NAME), os.path.join(output_dir, BEST_KO_VOCAB_NAME))
    _copy_if_exists(os.path.join(resume_dir, BEST_ZH_VOCAB_NAME), os.path.join(output_dir, BEST_ZH_VOCAB_NAME))


def main():
    corpus_roots = ["/kaggle/input"]
    corpus_files = []
    for r in corpus_roots:
        if os.path.exists(r):
            corpus_files.extend(_find_xlsx_files(r))

    corpus_files = [f for f in corpus_files if os.path.isfile(f)]
    if not corpus_files:
        raise RuntimeError("未找到任何 .xlsx 语料文件，请把数据集添加到 /kaggle/input")

    print("检测到语料文件数:", len(corpus_files))
    for f in corpus_files[:20]:
        print("  ", f)
    if len(corpus_files) > 20:
        print("  ...")

    tok_ko, tok_zh, ko_name, zh_name = _build_tokenizers()
    print("tokenizer ko:", ko_name)
    print("tokenizer zh:", zh_name)

    ko_sents, zh_sents = read_corpus(corpus_files)
    print("pairs read:", len(ko_sents))

    ko_sents = [clean_text(s) for s in ko_sents]
    zh_sents = [clean_text(s) for s in zh_sents]
    nonempty = [(k, z) for k, z in zip(ko_sents, zh_sents) if k and z]
    ko_sents = [k for k, _ in nonempty]
    zh_sents = [z for _, z in nonempty]
    print("pairs after clean+drop empty:", len(ko_sents))

    ko_tokens = [tok_ko(s) for s in ko_sents]
    zh_tokens = [tok_zh(s) for s in zh_sents]

    ko_vocab = build_vocab(ko_tokens)
    zh_vocab = build_vocab(zh_tokens)
    print("ko_vocab size:", len(ko_vocab))
    print("zh_vocab size:", len(zh_vocab))

    ko_train, ko_test, zh_train, zh_test = train_test_split(ko_tokens, zh_tokens, test_size=0.1, random_state=42)
    train_ds = TranslationDataset(ko_train, zh_train, ko_vocab, zh_vocab, max_ko_len=MAX_KO_LEN, max_zh_len=MAX_ZH_LEN)
    test_ds = TranslationDataset(ko_test, zh_test, ko_vocab, zh_vocab, max_ko_len=MAX_KO_LEN, max_zh_len=MAX_ZH_LEN)

    if torch.cuda.is_available():
        device = torch.device("cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0")
    else:
        device = torch.device("cpu")
    print("device:", device, "gpu_count:", torch.cuda.device_count())
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    per_step_batch = 8 if device.type == "cuda" else 16
    train_sampler = BucketBatchSampler(train_ds.zh_lens, batch_size=per_step_batch, bucket_size=2048, drop_last=False, seed=42)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=per_step_batch if device.type == "cuda" else 16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(device.type == "cuda"),
    )

    model_folder = os.environ.get("MODEL_FOLDER", "/kaggle/working/Translate Model")
    os.makedirs(model_folder, exist_ok=True)

    resume_folder = os.environ.get("RESUME_FOLDER", "").strip() or None
    if resume_folder is None:
        resume_folder = _find_first_checkpoint_dir("/kaggle/input")
    if resume_folder:
        print("检测到可恢复目录:", resume_folder)
        _restore_resume_files(model_folder, resume_folder)

    train_model(train_loader, test_loader, ko_vocab, zh_vocab, device, model_folder, max_epochs=50, grad_accum_steps=12, teacher_forcing_ratio=None)


if __name__ == "__main__":
    main()
