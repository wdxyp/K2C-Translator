import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import os
import re
import unicodedata
import jieba
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import time
from datetime import datetime
import openpyxl
import tkinter as tk
from tkinter import filedialog


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

    for ch in ["(", ")", "[", "]", "{", "}", "<", ">", ",", ":", ";", "?", "!", "/", "\\", "$", "#", "@", "~", "&", "*", "%", "+", "=", '"', "·"]:
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


def tokenize(sentences, lang):
    tokenized = []
    total = len(sentences)
    print(f"正在对 {lang} 语料进行分词，总计 {total} 条句子...")
    if lang == "ko":
        try:
            okt = Okt()
            for i, sentence in enumerate(sentences):
                tokens = okt.morphs(sentence)
                tokenized.append(tokens)
                if (i + 1) % 1000 == 0:
                    print(f"  已完成 {i+1}/{total} ({(i+1)/total*100:.1f}%)", end="\r")
        except Exception as e:
            print(f"韩文分词器启动失败: {e}，使用空格分词...")
            for i, sentence in enumerate(sentences):
                tokenized.append(sentence.split())
                if (i + 1) % 1000 == 0:
                    print(f"  已完成 {i+1}/{total} ({(i+1)/total*100:.1f}%)", end="\r")
    elif lang == "zh":
        for i, sentence in enumerate(sentences):
            tokenized.append(jieba.lcut(sentence))
            if (i + 1) % 1000 == 0:
                print(f"  已完成 {i+1}/{total} ({(i+1)/total*100:.1f}%)", end="\r")
    print(f"\n{lang} 分词完成！")
    return tokenized


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


def build_vocab(tokenized_sentences, min_freq=2, max_size=30000):
    counter = Counter()
    for tokens in tokenized_sentences:
        counter.update(tokens)

    most_common = counter.most_common(max_size)
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for word, freq in most_common:
        if freq >= min_freq:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


class TranslationDataset(Dataset):
    def __init__(self, ko_tokens, zh_tokens, ko_vocab, zh_vocab):
        self.ko_data = ko_tokens
        self.zh_data = zh_tokens
        self.ko_vocab = ko_vocab
        self.zh_vocab = zh_vocab

    def __len__(self):
        return len(self.ko_data)

    def __getitem__(self, idx):
        ko_idx = [self.ko_vocab["<sos>"]] + [self.ko_vocab.get(token, self.ko_vocab["<unk>"]) for token in self.ko_data[idx]] + [
            self.ko_vocab["<eos>"]
        ]
        zh_idx = [self.zh_vocab["<sos>"]] + [self.zh_vocab.get(token, self.zh_vocab["<unk>"]) for token in self.zh_data[idx]] + [
            self.zh_vocab["<eos>"]
        ]
        return torch.LongTensor(ko_idx), torch.LongTensor(zh_idx)


def collate_fn(batch):
    ko_batch, zh_batch = zip(*batch)
    ko_lens = [len(x) for x in ko_batch]
    zh_lens = [len(x) for x in zh_batch]

    ko_padded = nn.utils.rnn.pad_sequence(ko_batch, padding_value=0)
    zh_padded = nn.utils.rnn.pad_sequence(zh_batch, padding_value=0)

    return ko_padded, torch.LongTensor(ko_lens), zh_padded, torch.LongTensor(zh_lens)


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
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


def train_model(train_loader, test_loader, ko_vocab, zh_vocab, device):
    INPUT_DIM = len(ko_vocab)
    OUTPUT_DIM = len(zh_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 1
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    attn = Attention(HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT if N_LAYERS > 1 else 0)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT if N_LAYERS > 1 else 0, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_test_loss = float("inf")
    patience = 10
    no_improve = 0

    model_folder = "Translate Model"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model_path = os.path.join(model_folder, "best_model_v3_0_1_attn.pth")
    best_ko_vocab_path = os.path.join(model_folder, "best_ko_vocab_v3_0_1_attn.pkl")
    best_zh_vocab_path = os.path.join(model_folder, "best_zh_vocab_v3_0_1_attn.pkl")

    for epoch in range(50):
        model.train()
        epoch_loss = 0
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
        test_loss = 0
        with torch.no_grad():
            for src, src_len, trg, trg_len in test_loader:
                src, trg = src.to(device), trg.to(device)
                output = model(src, src_len, trg, 0)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                loss = criterion(output, trg)
                test_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch: {epoch+1:02} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        scheduler.step(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            with open(best_ko_vocab_path, "wb") as f:
                pickle.dump(ko_vocab, f)
            with open(best_zh_vocab_path, "wb") as f:
                pickle.dump(zh_vocab, f)
            print("  -> 测试损失下降，保存最佳模型！")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("早停触发，训练结束。")
                break

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    torch.save(model.state_dict(), os.path.join(model_folder, f"model_v3_0_1_{timestamp}.pth"))
    with open(os.path.join(model_folder, f"ko_vocab_v3_0_1_{timestamp}.pkl"), "wb") as f:
        pickle.dump(ko_vocab, f)
    with open(os.path.join(model_folder, f"zh_vocab_v3_0_1_{timestamp}.pkl"), "wb") as f:
        pickle.dump(zh_vocab, f)
    print("训练全过程完成！")


def start_training():
    global corpus_file_paths
    if not corpus_file_paths:
        status_label.config(text="请先选择文件！")
        return

    status_label.config(text="正在分词，请稍候...")
    root.update()

    ko_sents, zh_sents = read_corpus(corpus_file_paths)
    ko_sents = [clean_text(s) for s in ko_sents]
    zh_sents = [clean_text(s) for s in zh_sents]

    ko_tokens = tokenize(ko_sents, "ko")
    zh_tokens = tokenize(zh_sents, "zh")

    ko_vocab = build_vocab(ko_tokens)
    zh_vocab = build_vocab(zh_tokens)

    ko_train, ko_test, zh_train, zh_test = train_test_split(ko_tokens, zh_tokens, test_size=0.1, random_state=42)

    train_ds = TranslationDataset(ko_train, zh_train, ko_vocab, zh_vocab)
    test_ds = TranslationDataset(ko_test, zh_test, ko_vocab, zh_vocab)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_model(train_loader, test_loader, ko_vocab, zh_vocab, device)


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
    root.title("韩中翻译训练 V3.0.1 (Attention)")
    corpus_file_paths = []

    btn_select = tk.Button(root, text="1. 选择语料文件", command=select_files)
    btn_select.pack(pady=10)

    file_list_text = tk.Text(root, height=5, width=60)
    file_list_text.pack(pady=5)

    btn_train = tk.Button(root, text="2. 开始训练 V3.0.1 (Attention)", command=start_training, bg="green", fg="white")
    btn_train.pack(pady=10)

    status_label = tk.Label(root, text="等待操作...")
    status_label.pack(pady=5)

    root.mainloop()

