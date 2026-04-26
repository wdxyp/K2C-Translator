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

# ========== 1. 环境与设备配置 (Kaggle 多 GPU 优化版) ==========
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        print(f"✅ 检测到 {gpu_count} 个 NVIDIA GPU (T4)。")
        return device, gpu_count
    else:
        return torch.device('cpu'), 0

DEVICE, GPU_COUNT = get_device()

def find_data_file(filename):
    """在 Kaggle 输入目录中搜索数据文件"""
    # 预设可能的后缀名
    extensions = ['', '.xlsx', '.xls', '.csv']
    
    # 1. 首先在当前目录下搜索
    for ext in extensions:
        path = filename + ext
        if os.path.exists(path):
            return path

    # 2. 搜索 Kaggle 输入目录 (/kaggle/input)
    kaggle_input_base = "/kaggle/input"
    if os.path.exists(kaggle_input_base):
        print(f"正在 Kaggle Input 中搜索: {filename} ...")
        for root, dirs, files in os.walk(kaggle_input_base):
            for f in files:
                # 模糊匹配文件名
                if filename.lower() in f.lower():
                    full_path = os.path.join(root, f)
                    # 检查是否是 Excel 或 CSV 文件
                    if any(full_path.endswith(ext) for ext in ['.xlsx', '.xls', '.csv']):
                        print(f"✅ 在 Kaggle 数据集中找到文件: {full_path}")
                        return full_path
    return None

# ========== 2. 数据预处理函数 ==========
def clean_text(sentence):
    sentence = re.sub(r'[^\w\s]', '', str(sentence))
    return sentence.strip()

def tokenize(sentences, lang, cache_path=None):
    # 1. 尝试从缓存读取 (Kaggle 缓存通常放在 /kaggle/working)
    if cache_path and os.path.exists(cache_path):
        print(f"📦 检测到 {lang} 分词缓存，正在加载...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    tokenized = []
    total = len(sentences)
    print(f"正在对 {lang} 语料进行分词，总计 {total} 条句子 (此过程仅在首次运行时较慢)...")
    
    if lang == 'ko':
        try:
            okt = Okt()
            for i, sentence in enumerate(sentences):
                tokens = okt.morphs(sentence)
                tokenized.append(tokens)
                if (i + 1) % 5000 == 0:
                    print(f"  已完成 {i+1}/{total} ({(i+1)/total*100:.1f}%)", end='\r')
        except Exception as e:
            print(f"韩文分词器启动失败: {e}. 尝试空格分词...")
            for i, sentence in enumerate(sentences):
                tokenized.append(sentence.split())
    elif lang == 'zh':
        # Kaggle Linux 环境支持并行分词
        try:
            jieba.enable_parallel()
            for i, sentence in enumerate(sentences):
                tokens = jieba.lcut(sentence)
                tokenized.append(tokens)
                if (i + 1) % 10000 == 0:
                    print(f"  已完成 {i+1}/{total} ({(i+1)/total*100:.1f}%)", end='\r')
            jieba.disable_parallel()
        except Exception:
            for i, sentence in enumerate(sentences):
                tokens = jieba.lcut(sentence)
                tokenized.append(tokens)
                
    print(f"\n{lang} 分词完成！")
    
    # 2. 保存到缓存
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(tokenized, f)
        print(f"💾 {lang} 分词结果已缓存至: {cache_path}")
        
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
    ko_padded = torch.nn.utils.rnn.pad_sequence(list(ko_batch), batch_first=False, padding_value=pad_idx)
    zh_padded = torch.nn.utils.rnn.pad_sequence(list(zh_batch), batch_first=False, padding_value=pad_idx)
    return ko_padded, ko_lens, zh_padded

# ========== 4. 训练核心逻辑 ==========
def train_on_kaggle(corpus_path):
    print(f"🚀 开始 Kaggle 训练任务，数据源: {corpus_path}")
    
    # 建立缓存路径 (Kaggle 输出目录 /kaggle/working/)
    file_mtime = os.path.getmtime(corpus_path)
    cache_dir = "/kaggle/working/token_cache"
    ko_cache = os.path.join(cache_dir, f"ko_{int(file_mtime)}.pkl")
    zh_cache = os.path.join(cache_dir, f"zh_{int(file_mtime)}.pkl")
    
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
    ko_tokens = tokenize(all_ko, 'ko', cache_path=ko_cache)
    zh_tokens = tokenize(all_zh, 'zh', cache_path=zh_cache)
    ko_train, ko_test, zh_train, zh_test = train_test_split(ko_tokens, zh_tokens, test_size=0.1, random_state=42)

    # 词汇表
    korean_vocab = build_vocab(ko_train)
    chinese_vocab = build_vocab(zh_train)
    
    # 打印统计信息
    print("\n" + "="*30)
    print(f"📊 训练统计结果:")
    print(f"   训练集句子数: {len(ko_train)}")
    print(f"   测试集句子数: {len(ko_test)}")
    print(f"   韩语词汇表大小: {len(korean_vocab)}")
    print(f"   中文词汇表大小: {len(chinese_vocab)}")
    print("="*30 + "\n")
    
    # 转张量
    train_data = list(zip(text_to_tensor(ko_train, korean_vocab), text_to_tensor(zh_train, chinese_vocab)))
    test_data = list(zip(text_to_tensor(ko_test, korean_vocab), text_to_tensor(zh_test, chinese_vocab)))

    # 参数配置 (Kaggle 2xT4 GPU 极大化优化)
    # T4 有 16GB 显存，双卡共有 32GB。
    # 增加 BATCH_SIZE 以充分利用双卡并行效率
    HID_DIM = 512
    EMB_DIM = 512
    BATCH_SIZE = 256 if GPU_COUNT > 1 else 128
    N_EPOCHS = 100
    
    # 初始化模型
    enc = Encoder(len(korean_vocab), EMB_DIM, HID_DIM, 2, 0.5)
    dec = Decoder(len(chinese_vocab), EMB_DIM, HID_DIM, 2, 0.5)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    # 如果有多个 GPU，使用 DataParallel 包装模型
    if GPU_COUNT > 1:
        print(f"🚀 正在开启多 GPU 并行训练模式 (DataParallel)...")
        model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=korean_vocab['<pad>'])
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, 
                                               collate_fn=lambda b: collate_fn(b, korean_vocab['<pad>']))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                                              collate_fn=lambda b: collate_fn(b, korean_vocab['<pad>']))
    
    best_test_loss = float('inf')
    save_dir = '/kaggle/working/Translate_Model_Kaggle'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(N_EPOCHS):
        # --- 训练阶段 ---
        model.train()
        epoch_train_loss = 0
        total_batches = len(train_loader)
        
        for i, (src, src_lens, trg) in enumerate(train_loader):
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
            
            epoch_train_loss += loss.item()
            
            if (i + 1) % 50 == 0 or (i + 1) == total_batches:
                progress = (i + 1) / total_batches * 100
                print(f" Epoch: {epoch+1:02d}, Batch: {i+1}/{total_batches}, Progress: {progress:6.2f}%, Train Loss: {loss.item():.3f}")
        
        avg_train_loss = epoch_train_loss / total_batches
        
        # --- 测试阶段 ---
        model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for src, src_lens, trg in test_loader:
                src, trg = src.to(DEVICE), trg.to(DEVICE)
                output = model(src, src_lens, trg, 0)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                loss = criterion(output, trg)
                epoch_test_loss += loss.item()
        
        avg_test_loss = epoch_test_loss / len(test_loader)
        print(f"\n[Summary] Epoch: {epoch+1:02d} | Train Loss: {avg_train_loss:.3f} | Test Loss: {avg_test_loss:.3f}")
        
        # 自动保存最优模型到 Kaggle 工作目录
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            # 如果是并行模型，保存时需要提取 .module
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), f'{save_dir}/best_model.pth')
            with open(f'{save_dir}/vocabs.pkl', 'wb') as f:
                pickle.dump({'ko': korean_vocab, 'zh': chinese_vocab}, f)
            print(f" ✨ 模型已保存至 {save_dir}/best_model.pth")

if __name__ == "__main__":
    # 搜索 Kaggle 数据集中的文件
    target_filename = "Corpus(K2C)-2"
    data_path = find_data_file(target_filename)
    
    if data_path:
        train_on_kaggle(data_path)
    else:
        print(f"❌ 错误: 无法在 Kaggle Input 中找到包含 '{target_filename}' 的 Excel 文件。")
        print("提示: 请确保您已通过 'Add Data' 按钮上传了数据集，并正确拼写了文件名。")
