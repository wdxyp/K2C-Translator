import re
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from collections import Counter
import os
import pickle
import jieba
import openpyxl
from konlpy.tag import Okt
from datetime import datetime

# ========== 1. 环境与设备配置 ==========
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()
print(f"当前运行设备: {DEVICE}")

def mount_google_drive():
    """如果是 Colab 环境，检查或提示挂载 Google Drive"""
    if os.path.exists('/content/drive/MyDrive'):
        print("✅ 检测到 Google Drive 已挂载。")
        return True
        
    try:
        import sys
        # 判断是否在 Colab 内部（交互式环境）
        is_colab = 'google.colab' in sys.modules or os.path.exists('/content')
        
        if is_colab:
            print("提示: 在 Colab 中使用 !python 运行脚本时无法直接挂载网盘。")
            print("请确保您已在 Colab 的代码单元格中运行了以下代码：")
            print("    from google.colab import drive")
            print("    drive.mount('/content/drive')")
        return False
    except Exception:
        return False

def find_data_file(filename):
    """在当前目录和 Google Drive 中搜索数据文件"""
    # 预设可能的后缀名
    extensions = ['', '.xlsx', '.xls', '.csv']
    
    # 1. 首先在当前目录下搜索
    for ext in extensions:
        path = filename + ext
        if os.path.exists(path):
            return path

    # 2. 如果在 Colab 中，搜索 Google Drive
    drive_base = "/content/drive/MyDrive"
    if os.path.exists(drive_base):
        print(f"正在 Google Drive 中搜索: {filename} ...")
        # 优先匹配根目录下的文件
        for ext in extensions:
            path = os.path.join(drive_base, filename + ext)
            if os.path.exists(path):
                return path
        
        # 如果根目录没找到，进行全盘递归搜索
        for root, dirs, files in os.walk(drive_base):
            for f in files:
                if f.startswith(filename):
                    full_path = os.path.join(root, f)
                    print(f"在网盘路径中找到文件: {full_path}")
                    return full_path
    return None

# ========== 2. 数据预处理函数 ==========
def clean_text(sentence):
    sentence = re.sub(r'[^\w\s]', '', str(sentence))
    return sentence.strip()

def tokenize(sentences, lang):
    tokenized = []
    total = len(sentences)
    print(f"正在对 {lang} 语料进行分词，总计 {total} 条句子...")
    if lang == 'ko':
        try:
            okt = Okt()
            for i, sentence in enumerate(sentences):
                tokens = okt.morphs(sentence)
                tokenized.append(tokens)
                if (i + 1) % 1000 == 0:
                    print(f"  已完成 {i+1}/{total} ({(i+1)/total*100:.1f}%)", end='\r')
        except Exception as e:
            print(f"韩文分词器启动失败: {e}. 尝试空格分词...")
            for i, sentence in enumerate(sentences):
                tokenized.append(sentence.split())
    elif lang == 'zh':
        for i, sentence in enumerate(sentences):
            tokens = jieba.lcut(sentence)
            tokenized.append(tokens)
            if (i + 1) % 1000 == 0:
                print(f"  已完成 {i+1}/{total} ({(i+1)/total*100:.1f}%)", end='\r')
    print(f"\n{lang} 分词完成！")
    return tokenized

def build_vocab(sentences, min_freq=2, max_size=30000):
    counter = Counter()
    for sentence in sentences:
        for word in sentence:
            if isinstance(word, str) and word.strip():
                counter[word] += 1
    most_common = counter.most_common(max_size)
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    index = 4
    for word, freq in most_common:
        if freq >= min_freq and word not in vocab:
            vocab[word] = index
            index += 1
    return vocab

def text_to_tensor(sentences, vocab):
    tensors = []
    unk_idx = vocab.get('<unk>', 0)
    for sentence in sentences:
        tensor = [vocab['<sos>']] + [vocab.get(word, unk_idx) for word in sentence] + [vocab['<eos>']]
        tensors.append(torch.LongTensor(tensor))
    return tensors

# ========== 3. 模型定义 (Seq2Seq + LSTM) ==========
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_lens, batch_first=False, enforce_sorted=False)
        outputs, (hidden, cell) = self.rnn(packed)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src, src_lens)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

def collate_fn(batch, pad_idx):
    ko_batch, zh_batch = zip(*batch)
    ko_lens = [len(seq) for seq in ko_batch]
    # 转换为 list 以满足类型检查
    ko_padded = torch.nn.utils.rnn.pad_sequence(list(ko_batch), batch_first=False, padding_value=pad_idx)
    zh_padded = torch.nn.utils.rnn.pad_sequence(list(zh_batch), batch_first=False, padding_value=pad_idx)
    return ko_padded, ko_lens, zh_padded

# ========== 4. 训练核心逻辑 ==========
def train_on_cloud(corpus_path):
    print(f"开始云端训练任务，数据源: {corpus_path}")
    
    # 读取语料
    all_ko, all_zh = [], []
    wb = openpyxl.load_workbook(corpus_path, data_only=True)
    ws = wb.active
    if ws is None:
        raise ValueError("无法读取 Excel 表格内容")
        
    for row in ws.iter_rows(min_row=2, values_only=True):
        if len(row) >= 4 and row[1] and row[3]:
            all_ko.append(clean_text(row[1]))
            all_zh.append(clean_text(row[3]))
    
    # 分词与划分
    ko_tokens = tokenize(all_ko, 'ko')
    zh_tokens = tokenize(all_zh, 'zh')
    ko_train, ko_test, zh_train, zh_test = train_test_split(ko_tokens, zh_tokens, test_size=0.1, random_state=42)

    # 词汇表
    korean_vocab = build_vocab(ko_train)
    chinese_vocab = build_vocab(zh_train)
    
    # 转张量
    train_data = list(zip(text_to_tensor(ko_train, korean_vocab), text_to_tensor(zh_train, chinese_vocab)))
    test_data = list(zip(text_to_tensor(ko_test, korean_vocab), text_to_tensor(zh_test, chinese_vocab)))

    # 参数配置 (根据算力动态调整)
    is_accelerated = DEVICE.type in ['cuda', 'mps']
    HID_DIM = 512 if is_accelerated else 256
    EMB_DIM = 512 if is_accelerated else 256
    BATCH_SIZE = 128 if is_accelerated else 64
    N_EPOCHS = 100 if is_accelerated else 20
    
    # 初始化模型
    enc = Encoder(len(korean_vocab), EMB_DIM, HID_DIM, 2, 0.5)
    dec = Decoder(len(chinese_vocab), EMB_DIM, HID_DIM, 2, 0.5)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=korean_vocab['<pad>'])
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, 
                                               collate_fn=lambda b: collate_fn(b, korean_vocab['<pad>']))
    
    best_loss = float('inf')
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0
        for src, src_lens, trg in train_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            optimizer.zero_grad()
            output = model(src, src_lens, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{N_EPOCHS} | Loss: {avg_loss:.4f}")
        
        # 自动保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_dir = 'Translate_Model_Cloud'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
            with open(f'{save_dir}/vocabs.pkl', 'wb') as f:
                pickle.dump({'ko': korean_vocab, 'zh': chinese_vocab}, f)

if __name__ == "__main__":
    # 1. 尝试挂载 Google Drive (仅在 Colab 环境生效)
    mount_google_drive()
    
    # 2. 搜索数据文件
    target_filename = "Corpus(K2C)-2"
    data_path = find_data_file(target_filename)
    
    if data_path:
        print(f"成功定位数据文件: {data_path}")
        train_on_cloud(data_path)
    else:
        print(f"❌ 错误: 无法在当前目录或 Google Drive 中找到文件 '{target_filename}'")
        print("提示: 请确保已将文件上传到 Google Drive 的根目录，或者上传到 Colab 的 /content/ 文件夹中。")
