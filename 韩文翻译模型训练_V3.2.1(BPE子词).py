import os
import re
import pickle
import tempfile
import unicodedata
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import openpyxl
import tkinter as tk
from tkinter import filedialog

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


def clean_text(sentence):
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


def read_corpus(file_paths):
    all_korean = []
    all_chinese = []
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
    def __init__(self, ko_ids, zh_ids):
        self.ko_ids = ko_ids
        self.zh_ids = zh_ids

    def __len__(self):
        return len(self.ko_ids)

    def __getitem__(self, idx):
        return torch.LongTensor(self.ko_ids[idx]), torch.LongTensor(self.zh_ids[idx])


def collate_fn(batch):
    ko_batch, zh_batch = zip(*batch)
    ko_lens = [len(x) for x in ko_batch]
    zh_lens = [len(x) for x in zh_batch]

    ko_padded = nn.utils.rnn.pad_sequence(ko_batch, padding_value=0)
    zh_padded = nn.utils.rnn.pad_sequence(zh_batch, padding_value=0)

    return ko_padded, torch.LongTensor(ko_lens), zh_padded, torch.LongTensor(zh_lens)


def train_sentencepiece_models(ko_sents, zh_sents, model_folder, vocab_size=16000):
    ko_prefix = os.path.join(model_folder, "spm_ko_v3_2_1")
    zh_prefix = os.path.join(model_folder, "spm_zh_v3_2_1")
    ko_model_path = f"{ko_prefix}.model"
    zh_model_path = f"{zh_prefix}.model"

    if os.path.exists(ko_model_path) and os.path.exists(zh_model_path):
        return ko_model_path, zh_model_path

    if spm is None:
        raise RuntimeError("未安装 sentencepiece，请先执行：pip install sentencepiece")

    with tempfile.TemporaryDirectory() as tmpdir:
        ko_txt = os.path.join(tmpdir, "ko.txt")
        zh_txt = os.path.join(tmpdir, "zh.txt")

        with open(ko_txt, "w", encoding="utf-8") as f:
            for s in ko_sents:
                f.write((s or "").replace("\n", " ") + "\n")

        with open(zh_txt, "w", encoding="utf-8") as f:
            for s in zh_sents:
                f.write((s or "").replace("\n", " ") + "\n")

        spm.SentencePieceTrainer.Train(
            input=ko_txt,
            model_prefix=ko_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
        )
        spm.SentencePieceTrainer.Train(
            input=zh_txt,
            model_prefix=zh_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
        )

    return ko_model_path, zh_model_path


def encode_with_spm(sentences, sp):
    bos = sp.bos_id()
    eos = sp.eos_id()
    all_ids = []
    for s in sentences:
        piece_ids = sp.encode(s, out_type=int)
        all_ids.append([bos] + piece_ids + [eos])
    return all_ids


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
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_len)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs


def train_model_bpe(train_loader, test_loader, ko_sp, zh_sp, device, model_folder, max_epochs=50):
    input_dim = ko_sp.get_piece_size()
    output_dim = zh_sp.get_piece_size()

    enc_emb_dim = 256
    dec_emb_dim = 256
    hid_dim = 512
    n_layers = 1

    attn = Attention(hid_dim)
    enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, 0).to(device)
    dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, 0, attn).to(device)
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_state_path = os.path.join(model_folder, "best_model_v3_2_1_bpe_attn.pth")
    best_ckpt_path = os.path.join(model_folder, "best_model_v3_2_1_bpe_attn.ckpt")

    start_epoch = 0
    best_test_loss = float("inf")
    patience = 10
    no_improve = 0

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_test_loss = float(ckpt.get("best_test_loss", best_test_loss))
            no_improve = int(ckpt.get("no_improve", 0))
            print(f"检测到已存在的 BPE Checkpoint，继续训练: {best_ckpt_path}")

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_loss = 0.0
        for i, (src, src_len, trg, trg_len) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, src_len, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch: {epoch+1:02}, Batch: {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for src, src_len, trg, trg_len in test_loader:
                src, trg = src.to(device), trg.to(device)
                output = model(src, src_len, trg, 0)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                loss = criterion(output, trg)
                test_loss += loss.item()

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        avg_test_loss = test_loss / max(1, len(test_loader))
        print(f"Epoch: {epoch+1:02} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        scheduler.step(avg_test_loss)

        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_test_loss": best_test_loss,
            "no_improve": no_improve,
            "ko_spm_model": os.path.join(model_folder, "spm_ko_v3_2_1.model"),
            "zh_spm_model": os.path.join(model_folder, "spm_zh_v3_2_1.model"),
        }

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            no_improve = 0
            torch.save(model.state_dict(), best_state_path)
            state["best_test_loss"] = best_test_loss
            state["no_improve"] = no_improve
            torch.save(state, best_ckpt_path)
            print("  -> 测试损失下降，保存最佳模型与训练状态！")
        else:
            no_improve += 1
            state["no_improve"] = no_improve
            torch.save(state, best_ckpt_path)
            if no_improve >= patience:
                print("早停触发，训练结束。")
                break

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    torch.save(model.state_dict(), os.path.join(model_folder, f"model_v3_2_1_{timestamp}.pth"))
    print("训练全过程完成！")


def start_training_bpe():
    global corpus_file_paths
    if not corpus_file_paths:
        status_label.config(text="请先选择文件！")
        return
    if spm is None:
        import sys

        msg = "缺少 sentencepiece（当前解释器无法导入）。"
        if spm_import_error:
            msg += f" 错误: {spm_import_error}"
        msg += f" 解释器: {sys.executable}"
        msg += "（建议用 .venv\\Scripts\\python 运行本脚本）"
        status_label.config(text=msg)
        return

    status_label.config(text="正在准备 BPE，请稍候...")
    root.update()

    ko_sents, zh_sents = read_corpus(corpus_file_paths)
    ko_sents = [clean_text(s) for s in ko_sents]
    zh_sents = [clean_text(s) for s in zh_sents]

    model_folder = "Translate Model"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    ko_model_path, zh_model_path = train_sentencepiece_models(ko_sents, zh_sents, model_folder, vocab_size=16000)
    ko_sp = spm.SentencePieceProcessor(model_file=ko_model_path)
    zh_sp = spm.SentencePieceProcessor(model_file=zh_model_path)

    ko_ids = encode_with_spm(ko_sents, ko_sp)
    zh_ids = encode_with_spm(zh_sents, zh_sp)

    ko_train, ko_test, zh_train, zh_test = train_test_split(ko_ids, zh_ids, test_size=0.1, random_state=42)
    train_ds = TranslationIdDataset(ko_train, zh_train)
    test_ds = TranslationIdDataset(ko_test, zh_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    train_model_bpe(train_loader, test_loader, ko_sp, zh_sp, device, model_folder)


def select_files():
    global corpus_file_paths
    files = filedialog.askopenfilenames(title="选择语料库文件 (Excel)", filetypes=[("Excel files", "*.xlsx")])
    if files:
        corpus_file_paths = list(files)
        file_list_text.delete(1.0, tk.END)
        for f in corpus_file_paths:
            file_list_text.insert(tk.END, f + "\n")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("韩中翻译训练 V3.2.1 (BPE)")
    corpus_file_paths = []

    btn_select = tk.Button(root, text="1. 选择语料文件", command=select_files)
    btn_select.pack(pady=10)

    file_list_text = tk.Text(root, height=5, width=60)
    file_list_text.pack(pady=5)

    btn_train = tk.Button(root, text="2. 开始训练 V3.2.1 (BPE)", command=start_training_bpe, bg="blue", fg="white")
    btn_train.pack(pady=10)

    status_label = tk.Label(root, text="等待操作...")
    status_label.pack(pady=5)

    root.mainloop()

