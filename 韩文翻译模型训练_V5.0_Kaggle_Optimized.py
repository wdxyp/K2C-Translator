import re
import numpy as np
import warnings

# 屏蔽来自第三方库（如 jieba）的 Python 3.12 语法警告
warnings.filterwarnings("ignore", category=SyntaxWarning)

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import os
import pickle
import jieba
import openpyxl
from konlpy.tag import Okt

# ========== 1. 环境与设备配置 (V5.5 深度内存优化版) ==========
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        print(f"✅ 检测到 {gpu_count} 个 NVIDIA GPU (T4)。")
        return device, gpu_count
    else:
        return torch.device('cpu'), 0

DEVICE, GPU_COUNT = get_device()

def train_test_split(x, y, test_size=0.1, random_state=42):
    if len(x) != len(y):
        raise ValueError("x 与 y 长度不一致")
    n = len(x)
    indices = list(range(n))
    rng = random.Random(random_state)
    rng.shuffle(indices)
    split = int(n * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    x_train = [x[i] for i in train_idx]
    x_test = [x[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    return x_train, x_test, y_train, y_test

def find_data_file(filename):
    """在 Kaggle 输入目录中搜索数据文件"""
    kaggle_input_base = "/kaggle/input"
    if os.path.exists(kaggle_input_base):
        print(f"🔍 正在 Kaggle Input 中搜索: {filename} ...")
        for root, _, files in os.walk(kaggle_input_base):
            for f in files:
                if filename.lower() in f.lower() and any(f.endswith(ext) for ext in ['.xlsx', '.xls', '.csv']):
                    full_path = os.path.join(root, f)
                    print(f"✅ 找到语料文件: {full_path}")
                    return full_path
    return None

# ========== 2. 数据处理 (全面对齐 batch_first) ==========
def clean_text(sentence):
    return re.sub(r'[^\w\s]', '', str(sentence)).strip()

def tokenize(sentences, lang, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"📦 加载 {lang} 分词缓存...")
        with open(cache_path, 'rb') as f: return pickle.load(f)
    
    tokenized = []
    print(f"正在对 {lang} 语料分词 (总计 {len(sentences)} 条)...")
    if lang == 'ko':
        okt = Okt()
        for i, s in enumerate(sentences):
            tokenized.append(okt.morphs(s))
            if (i+1)%5000==0: print(f"  ko: {i+1}/{len(sentences)}", end='\r')
    else:
        try:
            jieba.enable_parallel()
            for i, s in enumerate(sentences):
                tokenized.append(jieba.lcut(s))
                if (i+1)%10000==0: print(f"  zh: {i+1}/{len(sentences)}", end='\r')
            jieba.disable_parallel()
        except:
            for i, s in enumerate(sentences):
                tokenized.append(jieba.lcut(s))
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f: pickle.dump(tokenized, f)
    return tokenized

def build_vocab(sentences, max_size=30000):
    counter = Counter()
    for s in sentences:
        for w in s:
            if isinstance(w, str) and w.strip(): counter[w] += 1
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    for w, _ in counter.most_common(max_size):
        if w not in vocab: vocab[w] = len(vocab)
    return vocab

def text_to_tensor(sentences, vocab):
    tensors = []
    unk = vocab.get('<unk>', 3)
    for s in sentences:
        t = [vocab['<sos>']] + [vocab.get(w, unk) for w in s] + [vocab['<eos>']]
        tensors.append(torch.LongTensor(t))
    return tensors

# ========== 3. 模型定义 (DataParallel 深度适配) ==========
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        # src_lens 必须在 CPU 上用于 pack_padded_sequence
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_lens.to('cpu'), batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.rnn(packed)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1) # [batch, 1]
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens, trg, teacher_forcing_ratio=0.5):
        # DataParallel 会切分 batch_size 到各卡
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, self.decoder.output_dim).to(src.device)
        
        hidden, cell = self.encoder(src, src_lens)
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)
            
        return outputs

def collate_fn(batch, pad_idx):
    ko, zh = zip(*batch)
    ko_lens = torch.LongTensor([len(x) for x in ko])
    ko_padded = torch.nn.utils.rnn.pad_sequence(list(ko), batch_first=True, padding_value=pad_idx)
    zh_padded = torch.nn.utils.rnn.pad_sequence(list(zh), batch_first=True, padding_value=pad_idx)
    return ko_padded, ko_lens, zh_padded

class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

# ========== 4. 训练核心逻辑 ==========
def train_on_kaggle(corpus_path):
    print(f"🚀 启动 Kaggle 训练 (V5.3 终极版)...")
    
    # 读取数据
    all_ko, all_zh = [], []
    wb = openpyxl.load_workbook(corpus_path, data_only=True)
    ws = wb.active
    if ws is None:
        print("❌ 错误: Excel 表格中没有活动工作表。")
        return

    for row in ws.iter_rows(min_row=2, values_only=True):
        if len(row) >= 4 and row[1] and row[3]:
            all_ko.append(clean_text(row[1])); all_zh.append(clean_text(row[3]))
    
    # 参数配置 (V5.5 极致显存压缩)
    HID_DIM = 256  # 从 512 降至 256
    EMB_DIM = 256  # 从 512 降至 256
    BATCH_SIZE = 64 if GPU_COUNT > 1 else 32  # 从 128 降至 64
    N_EPOCHS = 100
    MAX_LEN = 100  # 限制最大长度，防止异常长句子撑爆显存
    
    # 清理显存碎片
    torch.cuda.empty_cache()
    
    # 过滤超长句子
    filtered_ko, filtered_zh = [], []
    for k, z in zip(all_ko, all_zh):
        if len(k) <= MAX_LEN and len(z) <= MAX_LEN:
            filtered_ko.append(k); filtered_zh.append(z)
    all_ko, all_zh = filtered_ko, filtered_zh
    
    # 分词
    cache_dir = "/kaggle/working/token_cache"
    ko_tokens = tokenize(all_ko, 'ko', f"{cache_dir}/ko.pkl")
    zh_tokens = tokenize(all_zh, 'zh', f"{cache_dir}/zh.pkl")
    
    # 划分数据集
    ko_train, ko_test, zh_train, zh_test = train_test_split(ko_tokens, zh_tokens, test_size=0.1, random_state=42)
    k_vocab = build_vocab(ko_train); c_vocab = build_vocab(zh_train)
    
    print(f"📊 统计: 训练集={len(ko_train)}, 韩语词={len(k_vocab)}, 中文词={len(c_vocab)}")
    
    # 初始化模型与多 GPU
    model = Seq2Seq(Encoder(len(k_vocab), EMB_DIM, HID_DIM, 2, 0.5), 
                    Decoder(len(c_vocab), EMB_DIM, HID_DIM, 2, 0.5), DEVICE).to(DEVICE)
    if GPU_COUNT > 1:
        print("🚀 开启 DataParallel (双 T4)...")
        model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=c_vocab['<pad>'])
    
    # 开启混合精度训练 (AMP) - 使用最新 torch.amp 接口
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    train_loader = DataLoader(
        PairDataset(list(zip(text_to_tensor(ko_train, k_vocab), text_to_tensor(zh_train, c_vocab)))),
        batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=lambda b: collate_fn(b, k_vocab['<pad>']))
    
    test_loader = DataLoader(
        PairDataset(list(zip(text_to_tensor(ko_test, k_vocab), text_to_tensor(zh_test, c_vocab)))),
        batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_fn(b, k_vocab['<pad>']))
    
    best_loss = float('inf')
    save_dir = '/kaggle/working/Translate_Model_Kaggle'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(N_EPOCHS):
        model.train(); epoch_loss = 0
        for i, (src, src_lens, trg) in enumerate(train_loader):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            optimizer.zero_grad()
            
            # 使用混合精度进行前向传播
            with autocast(enabled=torch.cuda.is_available()):
                output = model(src, src_lens, trg)
                output_dim = output.shape[-1]
                loss = criterion(output[:, 1:, :].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            
            # 缩放损失并进行反向传播
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            if (i+1)%20==0: print(f" Epoch:{epoch+1:02d} B:{i+1}/{len(train_loader)} Loss:{loss.item():.3f}", end='\r')
        
        # 验证
        model.eval(); test_loss = 0
        with torch.no_grad():
            for src, src_lens, trg in test_loader:
                src, trg = src.to(DEVICE), trg.to(DEVICE)
                with autocast(enabled=torch.cuda.is_available()):
                    output = model(src, src_lens, trg, 0)
                    test_loss += criterion(output[:, 1:, :].reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1)).item()
        
        avg_train = epoch_loss / len(train_loader)
        avg_test = test_loss / len(test_loader)
        print(f"\n[Summary] Epoch:{epoch+1:02d} | Train Loss:{avg_train:.3f} | Test Loss:{avg_test:.3f}")
        
        if avg_test < best_loss:
            best_loss = avg_test
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), f'{save_dir}/best_model.pth')
            
            # 分别保存韩语和中文词汇表
            with open(f'{save_dir}/kr_vocab.pkl', 'wb') as f:
                pickle.dump(k_vocab, f)
            with open(f'{save_dir}/zh_vocab.pkl', 'wb') as f:
                pickle.dump(c_vocab, f)
                
            print(f" ✨ 模型与词汇表已保存至 {save_dir}")

if __name__ == "__main__":
    path = find_data_file("Corpus(K2C)-2")
    if path:
        train_on_kaggle(path)
    else:
        print("❌ 错误: 未能找到语料文件。")
