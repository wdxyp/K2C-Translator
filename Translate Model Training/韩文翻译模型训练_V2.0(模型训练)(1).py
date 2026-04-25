import re
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
import tkinter as tk
from tkinter import filedialog
import csv
from datetime import datetime
import os
import chardet
import pickle

# 选择语料库文件
def select_corpus_files():
    root = tk.Tk()
    root.withdraw()

    print("请选择多个韩文语料库文件（可多选）")
    korean_file_paths = filedialog.askopenfilenames()
    print("请选择多个中文语料库文件（可多选）")
    chinese_file_paths = filedialog.askopenfilenames()

    return korean_file_paths, chinese_file_paths

# 假设韩语和中文句子分别存储在多个不同的txt文件中，一行一行扫描读取
def read_corpus(korean_file_paths, chinese_file_paths):
    all_korean_sentences = []
    all_chinese_sentences = []

    # 确保韩文和中文文件数量相同
    assert len(korean_file_paths) == len(chinese_file_paths), "韩语和中文文件数量不一致"

    for korean_file_path, chinese_file_path in zip(korean_file_paths, chinese_file_paths):
        try:
            # 检测韩文文件编码
            with open(korean_file_path, 'rb') as f_ko:
                raw_data_ko = f_ko.read()
                result_ko = chardet.detect(raw_data_ko)
                encoding_ko = result_ko['encoding']

            # 检测中文文件编码
            with open(chinese_file_path, 'rb') as f_zh:
                raw_data_zh = f_zh.read()
                result_zh = chardet.detect(raw_data_zh)
                encoding_zh = result_zh['encoding']

            with open(korean_file_path, 'r', encoding=encoding_ko) as f_ko, open(chinese_file_path, 'r', encoding=encoding_zh) as f_zh:
                korean_line = f_ko.readline()
                chinese_line = f_zh.readline()
                while korean_line and chinese_line:
                    all_korean_sentences.append(korean_line.strip())
                    all_chinese_sentences.append(chinese_line.strip())
                    korean_line = f_ko.readline()
                    chinese_line = f_zh.readline()
                # 检查两个文件是否同时到达末尾
                if korean_line or chinese_line:
                    raise ValueError(f"文件 {korean_file_path} 和 {chinese_file_path} 的行数不一致")
        except Exception as e:
            print(f"读取文件 {korean_file_path} 或 {chinese_file_path} 时出错: {e}")

    # 检查是否有句子被读取
    if len(all_korean_sentences) == 0 or len(all_chinese_sentences) == 0:
        print("未读取到任何句子，请检查文件内容。")

    return all_korean_sentences, all_chinese_sentences

# 清洗文本，去除特殊字符
def clean_text(sentence):
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence.strip()

# 分词（对于韩语和中文，可使用不同的分词工具）
def tokenize(sentences):
    tokenized = []
    for sentence in sentences:
        tokens = sentence.split()  # 简单示例，实际可能需要更复杂的分词方法
        tokenized.append(tokens)
    return tokenized

# 创建交互界面
def create_gui():
    root = tk.Tk()
    root.title("语料库文件选择和保存")

    # 全局变量用于存储文件路径
    global korean_file_paths, chinese_file_paths
    korean_file_paths = []
    chinese_file_paths = []

    # 显示文件路径的文本框
    korean_file_text = tk.Text(root, height=5, width=50)
    korean_file_text.pack(pady=5)
    chinese_file_text = tk.Text(root, height=5, width=50)
    chinese_file_text.pack(pady=5)

    # 选择语料库文件
    def select_files():
        # 修改为使用 global 关键字
        global korean_file_paths, chinese_file_paths
        korean_file_paths = filedialog.askopenfilenames()
        chinese_file_paths = filedialog.askopenfilenames()

        # 清空文本框
        korean_file_text.delete(1.0, tk.END)
        chinese_file_text.delete(1.0, tk.END)

        # 显示文件路径
        for path in korean_file_paths:
            korean_file_text.insert(tk.END, path + "\n")
        for path in chinese_file_paths:
            chinese_file_text.insert(tk.END, path + "\n")

        status_label.config(text=f"韩文文件: {len(korean_file_paths)} 个, 中文文件: {len(chinese_file_paths)} 个")

    # 保存文件
    def save_files():
        global korean_file_paths, chinese_file_paths
        if not korean_file_paths or not chinese_file_paths:
            status_label.config(text="请先选择语料库文件")
            return

        korean_sentences, chinese_sentences = read_corpus(korean_file_paths, chinese_file_paths)

        # 过滤掉空句子
        non_empty_indices = [i for i, (ko, zh) in enumerate(zip(korean_sentences, chinese_sentences)) if ko and zh]
        korean_sentences = [korean_sentences[i] for i in non_empty_indices]
        chinese_sentences = [chinese_sentences[i] for i in non_empty_indices]

        # 清洗文本
        korean_sentences = [clean_text(sent) for sent in korean_sentences]
        chinese_sentences = [clean_text(sent) for sent in chinese_sentences]

        # 分词
        korean_tokens = tokenize(korean_sentences)
        chinese_tokens = tokenize(chinese_sentences)

        # 划分训练集和测试集
        ko_train, ko_test, zh_train, zh_test = train_test_split(korean_tokens, chinese_tokens, test_size=0.2, random_state=42)

        # 将分词后的列表转换为字符串
        def tokens_to_string(tokens_list):
            return [' '.join(tokens) for tokens in tokens_list]

        ko_train_str = tokens_to_string(ko_train)
        ko_test_str = tokens_to_string(ko_test)
        zh_train_str = tokens_to_string(zh_train)
        zh_test_str = tokens_to_string(zh_test)

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # 定义保存文件夹名
        save_folder = f"Training Data"

        # 检查文件夹是否存在，如果不存在则创建
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        train_filename = os.path.join(save_folder, f"train_{timestamp}.csv")
        test_filename = os.path.join(save_folder, f"test_{timestamp}.csv")

        # 保存训练集到 CSV 文件
        with open(train_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['韩语', '中文'])
            for ko, zh in zip(ko_train_str, zh_train_str):
                if ko and zh:  # 确保行不为空
                    writer.writerow([ko, zh])

        # 保存测试集到 CSV 文件
        with open(test_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['韩语', '中文'])
            for ko, zh in zip(ko_test_str, zh_test_str):
                if ko and zh:  # 确保行不为空
                    writer.writerow([ko, zh])

        status_label.config(text=f"训练集已保存到 {train_filename}\n测试集已保存到 {test_filename}")
        # 训练模型
        train_model(train_filename, test_filename)

    # 开始按钮
    start_button = tk.Button(root, text="开始", command=save_files)
    start_button.pack(pady=10)

    # 停止按钮（目前只是占位，可根据需求添加停止逻辑）
    stop_button = tk.Button(root, text="停止", command=lambda: status_label.config(text="停止操作未实现"))
    stop_button.pack(pady=10)

    # 选择文件按钮
    select_button = tk.Button(root, text="选择语料库文件", command=select_files)
    select_button.pack(pady=10)

    status_label = tk.Label(root, text="")
    status_label.pack(pady=10)

    root.mainloop()

# 加载数据集
def load_dataset(file_path):
    korean_sentences = []
    chinese_sentences = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            korean_sentences.append(row[0].split())
            chinese_sentences.append(row[1].split())
    return korean_sentences, chinese_sentences

# 构建词汇表
def build_vocab(sentences):
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    index = 3
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = index
                index += 1
    return vocab

# 文本转张量
def text_to_tensor(sentences, vocab):
    tensors = []
    for sentence in sentences:
        tensor = [vocab['<sos>']] + [vocab.get(word, vocab['<pad>']) for word in sentence] + [vocab['<eos>']]
        tensors.append(torch.tensor(tensor, dtype=torch.long))
    return tensors

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

# 定义解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
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

# 定义Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# 训练模型
def train_model(train_file, test_file):
    # 加载数据集
    ko_train, zh_train = load_dataset(train_file)
    ko_test, zh_test = load_dataset(test_file)

    # 构建词汇表
    korean_vocab = build_vocab(ko_train)
    chinese_vocab = build_vocab(zh_train)

    # 构建反向词汇表
    vocab_reverse_chinese = {v: k for k, v in chinese_vocab.items()}

    # 文本转张量
    ko_train_tensor = text_to_tensor(ko_train, korean_vocab)
    zh_train_tensor = text_to_tensor(zh_train, chinese_vocab)
    ko_test_tensor = text_to_tensor(ko_test, korean_vocab)
    zh_test_tensor = text_to_tensor(zh_test, chinese_vocab)

    # 定义超参数
    INPUT_DIM = len(korean_vocab)
    OUTPUT_DIM = len(chinese_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    N_EPOCHS = 10
    CLIP = 1
    PAD_IDX = korean_vocab['<pad>']

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 训练模型
    train_losses = []
    test_losses = []
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0
        total_batches = len(ko_train_tensor)
        for i, (src, trg) in enumerate(zip(ko_train_tensor, zh_train_tensor)):
            src = src.unsqueeze(1).to(device)
            trg = trg.unsqueeze(1).to(device)
            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            # 输出当前进度
            progress = (i + 1) / total_batches * 100
            print(f'Epoch: {epoch+1:02}, Batch: {i+1}/{total_batches}, Progress: {progress:.2f}%, Train Loss: {loss.item():.3f}', end='\r')
        train_loss = epoch_loss / len(ko_train_tensor)
        train_losses.append(train_loss)
        print(f'\nEpoch: {epoch+1:02}, Train Loss: {train_loss:.3f}')

        # 计算测试集损失
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for src, trg in zip(ko_test_tensor, zh_test_tensor):
                src = src.unsqueeze(1).to(device)
                trg = trg.unsqueeze(1).to(device)
                output = model(src, trg)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                loss = criterion(output, trg)
                test_loss += loss.item()
        test_loss = test_loss / len(ko_test_tensor)
        test_losses.append(test_loss)
        print(f'Epoch: {epoch+1:02}, Test Loss: {test_loss:.3f}')

    # 评估模型
    model.eval()
    references = []
    candidates = []
    with torch.no_grad():
        for src, trg in zip(ko_test_tensor, zh_test_tensor):
            src = src.unsqueeze(1).to(device)
            trg = trg.unsqueeze(1).to(device)
            output = model(src, trg, teacher_forcing_ratio=0)  # 禁用教师强制
            output = output.argmax(2)
            output = output.squeeze(1).tolist()
            trg = trg.squeeze(1).tolist()
            reference = [[vocab_reverse_chinese[idx] for idx in trg if idx not in [chinese_vocab['<pad>'], chinese_vocab['<sos>'], chinese_vocab['<eos>']]]]
            candidate = [vocab_reverse_chinese[idx] for idx in output if idx not in [chinese_vocab['<pad>'], chinese_vocab['<sos>'], chinese_vocab['<eos>']]]
            references.append(reference)
            candidates.append(candidate)

    bleu_score = corpus_bleu(references, candidates)
    print(f'BLEU Score: {bleu_score:.4f}')

    # 检查并创建文件夹
    model_folder = 'Translate Model'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = os.path.join(model_folder, f'model_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    # 新增词汇表保存代码
    vocab_path = os.path.join(model_folder, f'korean_vocab_{timestamp}.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(korean_vocab, f)
    
    vocab_path = os.path.join(model_folder, f'chinese_vocab_{timestamp}.pkl') 
    with open(vocab_path, 'wb') as f:
        pickle.dump(chinese_vocab, f)

    # 打印训练集和测试集损失
    print('Train Losses:', train_losses)
    print('Test Losses:', test_losses)

# 移除原有的构建反向词汇表代码
# vocab_reverse_chinese = {v: k for k, v in chinese_vocab.items()}

# 运行交互界面
create_gui()