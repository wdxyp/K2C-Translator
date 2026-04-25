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

# 运行交互界面
create_gui()
