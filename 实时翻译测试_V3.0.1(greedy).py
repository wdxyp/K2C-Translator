import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import re
import unicodedata

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

    for ch in ["(", ")", "[", "]", "{", "}", "<", ">", ",", ":", ";", "?", "!", "/", "\\", "$", "#", "@", "~", "&", "*", "%", "+", "=", '"', "_", "-", "·"]:
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


def split_by_parentheses(text):
    if not isinstance(text, str) or not text:
        return [""]
    pattern = r"(\([^()]*\)|（[^（）]*）)"
    parts = re.split(pattern, text)
    return [p for p in parts if p is not None and p != ""]


def dedupe_repeated_cjk(text):
    if not isinstance(text, str) or not text:
        return text
    s = text
    for n in range(1, 7):
        s = re.sub(rf"([\u4e00-\u9fa5]{{{n}}})(?:\1)+", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def load_user_dict(md_path):
    token_overrides = {}
    direct_translations = {}
    replace_rules = {}
    glossary = {}
    model_only_terms = set()

    if not os.path.exists(md_path):
        return token_overrides, direct_translations, replace_rules, glossary, model_only_terms

    section = "glossary"
    with open(md_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("##"):
                header = line.lstrip("#").strip()
                if "分词" in header:
                    section = "tokenize"
                elif "直译" in header:
                    section = "translate"
                elif "替换" in header:
                    section = "replace"
                elif "术语" in header:
                    section = "glossary"
                else:
                    section = None
                continue
            if line.startswith("#"):
                header = line.lstrip("#").strip()
                if "术语" in header:
                    section = "glossary"
                continue
            if not line.startswith("- "):
                continue
            content = line[2:]
            sep_pos_ascii = content.find(":")
            sep_pos_full = content.find("：")
            sep_positions = [p for p in (sep_pos_ascii, sep_pos_full) if p != -1]
            if not sep_positions:
                only_ko = clean_text(content.strip())
                if only_ko:
                    model_only_terms.add(only_ko)
                    model_only_terms.add(content.strip())
                    if only_ko not in glossary:
                        glossary[only_ko] = ""
                    raw_ko = content.strip()
                    if raw_ko and raw_ko != only_ko and raw_ko not in glossary:
                        glossary[raw_ko] = ""
                continue

            sep_pos = min(sep_positions)
            left, right = content[:sep_pos], content[sep_pos + 1 :]
            left = left.strip()
            right = right.strip()
            if not left or not right:
                continue

            if section == "tokenize":
                token_overrides[clean_text(left)] = [t for t in right.split() if t]
            elif section == "translate":
                direct_translations[clean_text(left)] = right
            elif section == "replace":
                replace_rules[left] = right
            elif section == "glossary":
                parts = [p.strip() for p in re.split(r"[:：]", right) if p.strip()]
                if len(parts) >= 2:
                    wrong_zh = parts[0]
                    correct_zh = parts[1]
                else:
                    wrong_zh = parts[0]
                    correct_zh = parts[0]

                ko_term = left
                glossary[ko_term] = correct_zh
                ko_term_clean = clean_text(ko_term)
                if ko_term_clean and ko_term_clean != ko_term:
                    glossary[ko_term_clean] = correct_zh

                if wrong_zh:
                    replace_rules[wrong_zh] = correct_zh

    return token_overrides, direct_translations, replace_rules, glossary, model_only_terms


def apply_replacements(text, replace_rules):
    if not replace_rules:
        return text
    for src, dst in replace_rules.items():
        if src:
            text = text.replace(src, dst)
    return text


# --- 2. 模型定义 (必须与 V3.0 训练代码完全一致) ---
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

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_len)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = top1
        return outputs


# --- 3. 翻译函数 ---
def tokenize_for_display(text):
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = []
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9/_\-]*|[\uAC00-\uD7A3]+|\d+(?:\.\d+)?%?", text):
        tokens.append(m.group(0))
    return tokens


_okt = None


def tokenize_ko_for_model(text):
    s = clean_text(text)
    if not s:
        return []
    try:
        global _okt
        if _okt is None:
            from konlpy.tag import Okt

            _okt = Okt()
        return _okt.morphs(s)
    except Exception:
        return s.split()


def _decode_greedy(model, encoder_outputs, hidden, zh_vocab, device, max_len=50, repeat_penalty=1.2):
    inv_zh_vocab = {v: k for k, v in zh_vocab.items()}
    trg_indices = [zh_vocab["<sos>"]]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        logits = output.squeeze(0)
        if repeat_penalty and repeat_penalty > 1.0:
            for tid in set(trg_indices[1:]):
                logits[tid] = logits[tid] / repeat_penalty
        top1 = logits.argmax(0).item()
        trg_indices.append(top1)
        if top1 == zh_vocab["<eos>"]:
            break
    translated_tokens = [inv_zh_vocab.get(idx, "<unk>") for idx in trg_indices]
    translated_tokens = [t for t in translated_tokens if t not in ["<sos>", "<eos>", "<pad>"]]
    return translated_tokens


def translate_seq2seq_sentence_level(text, model, ko_vocab, zh_vocab, device, max_len=50, repeat_penalty=1.2):
    tokens = tokenize_ko_for_model(text)
    if not tokens:
        return "", 0.0
    unk_idx = ko_vocab.get("<unk>", 3)
    indices = [ko_vocab["<sos>"]] + [ko_vocab.get(t, unk_idx) for t in tokens] + [ko_vocab["<eos>"]]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(indices)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
    translated_tokens = _decode_greedy(
        model,
        encoder_outputs,
        hidden,
        zh_vocab,
        device,
        max_len=max_len,
        repeat_penalty=repeat_penalty,
    )
    if not translated_tokens:
        return "", 1.0
    unk_ratio = translated_tokens.count("<unk>") / max(1, len(translated_tokens))
    out = "".join([t for t in translated_tokens if t != "<unk>"]).strip()
    return out, unk_ratio


SIMPLE_PARTICLES = {"이", "의", "는", "를"}


def split_simple_particle(token):
    if not isinstance(token, str) or len(token) <= 1:
        return [token]
    if not re.fullmatch(r"[\uAC00-\uD7A3]+", token):
        return [token]
    last = token[-1]
    if last in SIMPLE_PARTICLES and len(token) > 1:
        return [token[:-1], last]
    return [token]


def split_by_user_terms(token, user_dict, max_pieces=4):
    if not isinstance(token, str) or len(token) <= 1:
        return [token]
    if not re.fullmatch(r"[\uAC00-\uD7A3]+", token):
        return [token]
    token_overrides, direct_translations, _, glossary, model_only_terms = user_dict
    known_terms = set()
    for k in glossary.keys():
        ck = clean_text(k)
        if ck:
            known_terms.add(ck)
    for k in direct_translations.keys():
        ck = clean_text(k)
        if ck:
            known_terms.add(ck)
    for k in token_overrides.keys():
        ck = clean_text(k)
        if ck:
            known_terms.add(ck)
    for k in model_only_terms:
        ck = clean_text(k)
        if ck:
            known_terms.add(ck)

    s = clean_text(token)
    if s in known_terms:
        return [token]

    best_prefix = None
    for j in range(len(s), 1, -1):
        cand = s[:j]
        if len(cand) >= 2 and cand in known_terms:
            best_prefix = cand
            break

    if not best_prefix:
        return [token]

    remainder = s[len(best_prefix) :]
    if not remainder:
        return [best_prefix]
    return [best_prefix, remainder]


def translate_korean_token(token, model, ko_vocab, zh_vocab, device, user_dict, cache, max_len=20, repeat_penalty=1.2):
    token_overrides, direct_translations, replace_rules, glossary, model_only_terms = user_dict
    key = clean_text(token)
    if not key:
        return ""
    if key in cache:
        return cache[key]

    if not (model_only_terms and key in model_only_terms) and key in direct_translations:
        out = apply_replacements(direct_translations[key], replace_rules)
        cache[key] = out
        return out
    if not (model_only_terms and key in model_only_terms) and key in glossary and glossary.get(key):
        out = apply_replacements(glossary[key], replace_rules)
        cache[key] = out
        return out

    if key in token_overrides:
        parts = token_overrides[key]
        out = "".join(
            [
                translate_korean_token(
                    p, model, ko_vocab, zh_vocab, device, user_dict, cache, max_len=max_len, repeat_penalty=repeat_penalty
                )
                for p in parts
            ]
        )
        cache[key] = out
        return out

    unk_idx = ko_vocab.get("<unk>", 3)
    indices = [ko_vocab["<sos>"], ko_vocab.get(key, unk_idx), ko_vocab["<eos>"]]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(indices)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    translated_tokens = _decode_greedy(
        model,
        encoder_outputs,
        hidden,
        zh_vocab,
        device,
        max_len=max_len,
        repeat_penalty=repeat_penalty,
    )
    out = "".join([t for t in translated_tokens if t not in ["<sos>", "<eos>", "<pad>"]])
    out = apply_replacements(out, replace_rules)
    if not out.strip():
        out = "<unk>"
    cache[key] = out
    return out


def translate_mixed_text_word_by_word(text, model, ko_vocab, zh_vocab, device, user_dict, cache):
    if not isinstance(text, str) or not text:
        return ""
    out = []
    for piece in re.findall(r"\s+|[A-Za-z][A-Za-z0-9/_\-\.\+]*|\d+(?:\.\d+)?%?|[\uAC00-\uD7A3]+|.", text):
        if not piece:
            continue
        if re.fullmatch(r"[\uAC00-\uD7A3]+", piece):
            for p1 in split_by_user_terms(piece, user_dict):
                for t in split_simple_particle(p1):
                    if t:
                        out.append(translate_korean_token(t, model, ko_vocab, zh_vocab, device, user_dict, cache))
        else:
            out.append(piece)
    return "".join(out)


def _extract_tokens_for_model(text):
    return tokenize_ko_for_model(text)


def translate_with_preserved_spans(text, model, ko_vocab, zh_vocab, device, user_dict, cache, max_len=50, repeat_penalty=1.2):
    if not isinstance(text, str) or not text:
        return ""
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    seq2seq_out, unk_ratio = translate_seq2seq_sentence_level(
        cleaned, model, ko_vocab, zh_vocab, device, max_len=max_len, repeat_penalty=repeat_penalty
    )
    if seq2seq_out and unk_ratio <= 0.5:
        return seq2seq_out
    return translate_mixed_text_word_by_word(cleaned, model, ko_vocab, zh_vocab, device, user_dict, cache)


def translate_sentence(sentence, model, ko_vocab, zh_vocab, device, user_dict, max_len=50, show_tokens=True):
    model.eval()
    token_overrides, direct_translations, replace_rules, glossary, model_only_terms = user_dict
    original_raw_sentence = sentence if isinstance(sentence, str) else ""
    raw_sentence = original_raw_sentence.strip()
    sentence_key = clean_text(raw_sentence)

    if show_tokens:
        print(f"分词结果(显示): {tokenize_for_display(raw_sentence)}")
        print(f"分词结果(送模型token): {_extract_tokens_for_model(raw_sentence)}")

    cache = {}

    translated_text = ""
    if not (model_only_terms and sentence_key in model_only_terms) and sentence_key in direct_translations:
        translated_text = direct_translations[sentence_key]
    elif not (model_only_terms and sentence_key in model_only_terms) and sentence_key in glossary and glossary.get(sentence_key):
        translated_text = glossary[sentence_key]
    else:
        parts = split_by_parentheses(raw_sentence)
        if isinstance(raw_sentence, str) and len(parts) > 1:
            out = []
            for p in parts:
                if p.startswith("(") and p.endswith(")"):
                    if re.fullmatch(r"\(\s*\d+\s*\)", p):
                        out.append(p)
                        continue
                    inner = p[1:-1]
                    inner_translated = translate_sentence(
                        inner, model, ko_vocab, zh_vocab, device, user_dict, max_len=max_len, show_tokens=False
                    )
                    out.append(f"({inner_translated})")
                    continue
                if p.startswith("（") and p.endswith("）"):
                    if re.fullmatch(r"（\s*\d+\s*）", p):
                        out.append(p)
                        continue
                    inner = p[1:-1]
                    inner_translated = translate_sentence(
                        inner, model, ko_vocab, zh_vocab, device, user_dict, max_len=max_len, show_tokens=False
                    )
                    out.append(f"（{inner_translated}）")
                    continue

                out.append(
                    translate_with_preserved_spans(
                        p, model, ko_vocab, zh_vocab, device, user_dict, cache, max_len=max_len, repeat_penalty=1.2
                    )
                )
            translated_text = "".join(out)
        else:
            translated_text = translate_with_preserved_spans(
                raw_sentence, model, ko_vocab, zh_vocab, device, user_dict, cache, max_len=max_len, repeat_penalty=1.2
            )

    translated_text = apply_replacements(translated_text, replace_rules)
    translated_text = dedupe_repeated_cjk(translated_text)
    return translated_text


# --- 4. 主程序 ---
if __name__ == "__main__":
    device = torch.device("cpu")
    model_dir = r"D:\PythonProject\Translate Model\V3.0(Attention)\20260527-best Model\epoch03-4.4666"
    model_dir = os.path.normpath(str(model_dir).strip().strip('"').strip("'"))
    user_dict_path = os.path.join(os.path.dirname(__file__), "user_dict.md")

    model_path = os.path.join(model_dir, "best_model_v3_0_1_attn.pth")
    ko_vocab_path = os.path.join(model_dir, "best_ko_vocab_v3_0_1_attn.pkl")
    zh_vocab_path = os.path.join(model_dir, "best_zh_vocab_v3_0_1_attn.pkl")

    import glob

    ko_vocabs = glob.glob(os.path.join(model_dir, "ko_vocab_v3_0_1_*.pkl"))
    zh_vocabs = glob.glob(os.path.join(model_dir, "zh_vocab_v3_0_1_*.pkl"))
    if ko_vocabs:
        ko_vocab_path = max(ko_vocabs, key=os.path.getctime)
    if zh_vocabs:
        zh_vocab_path = max(zh_vocabs, key=os.path.getctime)

    if not os.path.exists(model_path):
        fallback = os.path.join(model_dir, "best_model_v3_attn.pth")
        if os.path.exists(fallback):
            model_path = fallback
        else:
            print(f"找不到模型文件: {model_path}，请先运行 V3.0.1 训练脚本。")
    if not os.path.exists(ko_vocab_path) or not os.path.exists(zh_vocab_path):
        fallback_ko = os.path.join(model_dir, "best_ko_vocab_v3_attn.pkl")
        fallback_zh = os.path.join(model_dir, "best_zh_vocab_v3_attn.pkl")
        if os.path.exists(fallback_ko) and os.path.exists(fallback_zh):
            ko_vocab_path = fallback_ko
            zh_vocab_path = fallback_zh
        else:
            print(f"找不到词汇表文件: {ko_vocab_path} / {zh_vocab_path}，请先运行 V3.0.1 训练脚本。")
    else:
        with open(ko_vocab_path, "rb") as f:
            ko_vocab = pickle.load(f)
        with open(zh_vocab_path, "rb") as f:
            zh_vocab = pickle.load(f)

        INPUT_DIM = len(ko_vocab)
        OUTPUT_DIM = len(zh_vocab)
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        HID_DIM = 512
        N_LAYERS = 1

        attn = Attention(HID_DIM)
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, 0)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, 0, attn)
        model = Seq2Seq(enc, dec, device).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        print("V3.0.1 Attention 模型加载成功！")

        while True:
            sentence = input("\n请输入韩文 (输入 q 退出): ")
            if sentence.lower() == "q":
                break
            if not sentence.strip():
                continue

            user_dict = load_user_dict(user_dict_path)
            translation = translate_sentence(sentence, model, ko_vocab, zh_vocab, device, user_dict)
            print(f"中文翻译: {translation}")
