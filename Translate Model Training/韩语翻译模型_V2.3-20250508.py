# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import random
import spacy
from konlpy.tag import Okt
from torchtext.data import Field, BucketIterator, TabularDataset

# ========== 1. 环境设置 ==========
# 安装依赖 (需在终端运行以下命令)
# !pip install torch torchtext spacy konlpy
# !python -m spacy download zh_core_web_sm
# !python -m spacy download ko_core_news_sm

# ========== 2. 数据预处理 ==========
# 韩语分词工具
okt = Okt()
# 中文分词工具
spacy_zh = spacy.load("zh_core_web_sm")

def tokenize_korean(text):
    """韩语句子分词"""
    return okt.morphs(text)

def tokenize_chinese(text):
    """中文按字符分词"""
    return [char for char in text]

# 定义字段处理器
SRC = Field(
    tokenize=tokenize_korean,
    init_token="<sos>",
    eos_token="<eos>",
    lower=True
)

TRG = Field(
    tokenize=tokenize_chinese,
    init_token="<sos>",
    eos_token="<eos>",
    lower=True
)

# 加载数据 (示例文件: ko-zh.txt)
data_fields = [('src', SRC), ('trg', TRG)]
train_data = TabularDataset(
    path="ko-zh.txt",
    format="tsv",
    fields=data_fields,
    skip_header=False
)

# 构建词表
SRC.build_vocab(train_data, min_freq=1)
TRG.build_vocab(train_data, min_freq=1)

# 创建数据迭代器
BATCH_SIZE = 32
train_iterator = BucketIterator(
    train_data,
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# ========== 3. 模型定义 ==========
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))  # [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = nn.Linear(hid_dim + emb_dim, hid_dim)
        self.rnn = nn.LSTM(hid_dim + emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]
        
        # 注意力计算
        attn_weights = torch.sum(encoder_outputs * hidden[-1], dim=2).t()  # [batch_size, src_len]
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(1)     # [batch_size, 1, src_len]
        
        weighted = torch.bmm(attn_weights, encoder_outputs.permute(1,0,2)) # [batch_size, 1, hid_dim]
        weighted = weighted.permute(1,0,2)                                 # [1, batch_size, hid_dim]
        
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [1, batch_size, emb_dim + hid_dim]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        encoder_outputs, (hidden, cell) = self.encoder(src)
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        input = trg[0, :]  # 初始输入为<sos>
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# ========== 4. 训练配置 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])

# ========== 5. 训练函数 ==========
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for _, batch in enumerate(iterator):
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# ========== 6. 训练执行 ==========
N_EPOCHS = 20
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}')

# ========== 7. 翻译函数 ==========
def translate(sentence, model, src_field, trg_field, device, max_len=50):
    model.eval()
    
    if isinstance(sentence, str):
        tokens = [token.lower() for token in src_field.preprocess(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
        
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor)
    
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(
                trg_tensor, hidden, cell, encoder_outputs
            )
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return ''.join(trg_tokens[1:-1])  # 中文输出无需空格

# ========== 8. 测试示例 ==========
test_sentences = [
    "안녕하세요",
    "오늘 날씨가 좋아요",
    "한국 음식이 맛있습니다"
]

for sentence in test_sentences:
    translation = translate(sentence, model, SRC, TRG, device)
    print(f"韩语输入: {sentence}")
    print(f"中文输出: {translation}\n")