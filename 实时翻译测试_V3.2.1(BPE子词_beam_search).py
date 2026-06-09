import os
import re
import unicodedata
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def load_user_dict(md_path):
    token_overrides = {}
    direct_translations = {}
    replace_rules = {}
    glossary = {}

    if not os.path.exists(md_path):
        return token_overrides, direct_translations, replace_rules, glossary

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

    return token_overrides, direct_translations, replace_rules, glossary


def apply_replacements(text, replace_rules):
    if not replace_rules:
        return text
    for src, dst in replace_rules.items():
        if src:
            text = text.replace(src, dst)
    return text


def restore_raw_terms_in_output(text, glossary):
    if not isinstance(text, str) or not glossary:
        return text
    restored = text
    for raw_term in glossary.keys():
        if not isinstance(raw_term, str) or "/" not in raw_term:
            continue
        clean_term = clean_text(raw_term)
        if not clean_term or clean_term == raw_term:
            continue
        restored = restored.replace(clean_term, raw_term)
    return restored


def dedupe_repeated_ascii_runs(text):
    if not isinstance(text, str) or not text:
        return text
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9/]{2,})(?:\1)+")
    while True:
        new_text = pattern.sub(r"\1", text)
        if new_text == text:
            return text
        text = new_text


def dedupe_repeated_cjk_phrases(text):
    if not isinstance(text, str) or not text:
        return text
    pattern = re.compile(r"([\u4e00-\u9fa5]{1,4})(?:\1)+")
    while True:
        new_text = pattern.sub(r"\1", text)
        if new_text == text:
            return text
        text = new_text


def mark_untranslated_in_output(text):
    if not isinstance(text, str) or not text:
        return text
    marked = re.sub(r"[\uAC00-\uD7A3]+", "[?]", text)
    marked = re.sub(r"(?:\[\?\]){2,}", "[?]", marked)
    return marked


def apply_output_fallback(translated_text, original_text):
    if not isinstance(translated_text, str):
        return translated_text
    if translated_text.strip():
        return translated_text
    if isinstance(original_text, str) and re.search(r"[\uAC00-\uD7A3]", original_text):
        return "[?]"
    return translated_text


def strip_english_words_for_encode(text):
    if not isinstance(text, str) or not text:
        return text
    return re.sub(r"\s+", " ", text).strip()


def extract_english_terms(original_text):
    if not isinstance(original_text, str) or not original_text:
        return []
    src = original_text.replace("／", "/")
    terms = []
    for m in re.finditer(r"[A-Za-z0-9]+/[A-Za-z0-9]+", src):
        t = m.group(0)
        if t and t not in terms:
            terms.append(t)
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9]*", src):
        t = m.group(0)
        if t and t not in terms:
            terms.append(t)
    return terms


def split_by_parentheses(text):
    if not isinstance(text, str) or not text:
        return [text]
    pattern = re.compile(r"(\([^()]*\)|（[^（）]*）)")
    parts = pattern.split(text)
    return [p for p in parts if p is not None and p != ""]


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
        self.n_layers = int(n_layers)
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


def unwrap_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def infer_n_layers_from_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return 1
    max_layer = -1
    for k in state_dict.keys():
        m = re.search(r"\.rnn\.weight_ih_l(\d+)", k)
        if m:
            try:
                idx = int(m.group(1))
            except Exception:
                continue
            if idx > max_layer:
                max_layer = idx
    return max_layer + 1 if max_layer >= 0 else 1


def beam_search_decode_bpe(model, encoder_outputs, hidden, zh_sp, device, max_len=80, beam_size=5, length_penalty=0.7):
    sos_id = int(zh_sp.bos_id())
    eos_id = int(zh_sp.eos_id())
    no_repeat_ngram_size = 3
    repeat_token_penalty = 1.5

    def rank_score(total_logprob, seq_len):
        seq_len = max(1, seq_len)
        return total_logprob / (seq_len**length_penalty)

    def would_repeat_ngram(tokens, next_token_id):
        if no_repeat_ngram_size <= 0:
            return False
        if len(tokens) + 1 < no_repeat_ngram_size:
            return False
        n = no_repeat_ngram_size
        new_ngram = tuple(tokens[-(n - 1) :] + [next_token_id])
        existing = set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
        return new_ngram in existing

    beams = [{"tokens": [sos_id], "logprob": 0.0, "hidden": hidden, "ended": False}]

    for _ in range(max_len):
        candidates = []
        all_ended = True

        for beam in beams:
            if beam["ended"]:
                candidates.append(beam)
                continue

            all_ended = False
            last_token = beam["tokens"][-1]
            trg_tensor = torch.LongTensor([last_token]).to(device)
            with torch.no_grad():
                output, hidden_next = model.decoder(trg_tensor, beam["hidden"], encoder_outputs)

            log_probs = F.log_softmax(output, dim=1).squeeze(0)
            topk_log_probs, topk_ids = torch.topk(log_probs, k=min(beam_size, log_probs.numel()))

            for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                if tid == beam["tokens"][-1] and tid not in (sos_id, eos_id):
                    continue
                if len(beam["tokens"]) >= 3 and beam["tokens"][-1] == beam["tokens"][-2] and tid == beam["tokens"][-1]:
                    continue
                if would_repeat_ngram(beam["tokens"], tid):
                    continue
                if tid in beam["tokens"] and tid not in (sos_id, eos_id):
                    lp -= repeat_token_penalty

                new_tokens = beam["tokens"] + [tid]
                candidates.append(
                    {
                        "tokens": new_tokens,
                        "logprob": beam["logprob"] + lp,
                        "hidden": hidden_next,
                        "ended": tid == eos_id,
                    }
                )

        candidates.sort(key=lambda b: rank_score(b["logprob"], len(b["tokens"]) - 1), reverse=True)
        beams = candidates[:beam_size]

        if all_ended:
            break

    ended = [b for b in beams if b["ended"]]
    if ended:
        ended.sort(key=lambda b: rank_score(b["logprob"], len(b["tokens"]) - 1), reverse=True)
        return ended[0]["tokens"]

    beams.sort(key=lambda b: rank_score(b["logprob"], len(b["tokens"]) - 1), reverse=True)
    return beams[0]["tokens"]


def translate_sentence_core(sentence, model, ko_sp, zh_sp, device, user_dict, max_len=80):
    model.eval()
    token_overrides, direct_translations, replace_rules, glossary = user_dict

    original_raw_sentence = sentence.replace("／", "/") if isinstance(sentence, str) else ""
    raw_sentence = original_raw_sentence if original_raw_sentence else sentence

    def _can_split_progress(chunks, raw, delims):
        if not isinstance(raw, str):
            return False
        for ch in chunks:
            if not isinstance(ch, str):
                continue
            if ch in delims or not ch.strip():
                continue
            if len(ch) < len(raw):
                return True
        return False

    if isinstance(raw_sentence, str) and re.search(r"(?<!\d),(?!\d)|，", raw_sentence):
        chunks = re.split(r"((?<!\d),(?!\d)|，)", raw_sentence)
        if not _can_split_progress(chunks, raw_sentence, {",", "，"}):
            chunks = None
        if chunks is not None:
            out = []
            for ch in chunks:
                if ch in (",", "，"):
                    out.append(ch)
                    continue
                if ch == "" or not ch.strip():
                    out.append(ch)
                    continue
                out.append(translate_sentence(ch, model, ko_sp, zh_sp, device, user_dict, max_len=max_len))
            return "".join(out)

    if isinstance(raw_sentence, str) and re.search(r"(?<![A-Za-z0-9])/(?![A-Za-z0-9])", raw_sentence):
        chunks = re.split(r"((?<![A-Za-z0-9])/(?![A-Za-z0-9]))", raw_sentence)
        if not _can_split_progress(chunks, raw_sentence, {"/"}):
            chunks = None
        if chunks is not None:
            out = []
            for ch in chunks:
                if ch == "/":
                    out.append(ch)
                    continue
                if ch == "" or not ch.strip():
                    out.append(ch)
                    continue
                out.append(translate_sentence(ch, model, ko_sp, zh_sp, device, user_dict, max_len=max_len))
            return "".join(out)

    sentence = clean_text(raw_sentence)
    if not sentence:
        return ""

    if isinstance(sentence, str) and not re.search(r"[\uAC00-\uD7A3\u4e00-\u9fa5]", sentence):
        return raw_sentence.strip() if isinstance(raw_sentence, str) else sentence

    if sentence in direct_translations:
        translated_text = apply_replacements(direct_translations[sentence], replace_rules)
        translated_text = restore_raw_terms_in_output(translated_text, glossary)
        translated_text = dedupe_repeated_ascii_runs(translated_text)
        translated_text = dedupe_repeated_cjk_phrases(translated_text)
        return mark_untranslated_in_output(translated_text)

    if sentence in glossary:
        translated_text = apply_replacements(glossary[sentence], replace_rules)
        translated_text = restore_raw_terms_in_output(translated_text, glossary)
        translated_text = dedupe_repeated_ascii_runs(translated_text)
        translated_text = dedupe_repeated_cjk_phrases(translated_text)
        translated_text = apply_output_fallback(translated_text, original_raw_sentence)
        return mark_untranslated_in_output(translated_text)

    if isinstance(sentence, str) and " " in sentence:
        parts = [p for p in sentence.split() if p]
        if 1 < len(parts) <= 3 and any((p in glossary) or (p in direct_translations) for p in parts):
            merged = []
            for p in parts:
                if p in glossary:
                    merged.append(glossary[p])
                    continue
                if p in direct_translations:
                    merged.append(direct_translations[p])
                    continue
                merged.append(translate_sentence_core(p, model, ko_sp, zh_sp, device, user_dict, max_len=max_len))
            translated_text = " ".join([t for t in merged if isinstance(t, str) and t.strip()])
            translated_text = apply_replacements(translated_text, replace_rules)
            translated_text = restore_raw_terms_in_output(translated_text, glossary)
            translated_text = dedupe_repeated_ascii_runs(translated_text)
            translated_text = dedupe_repeated_cjk_phrases(translated_text)
            translated_text = apply_output_fallback(translated_text, original_raw_sentence)
            return mark_untranslated_in_output(translated_text)

    bos = int(ko_sp.bos_id())
    eos = int(ko_sp.eos_id())
    override_tokens = token_overrides.get(raw_sentence) or token_overrides.get(sentence)
    sentence_for_encode = " ".join(override_tokens) if override_tokens else sentence
    sentence_for_encode = strip_english_words_for_encode(sentence_for_encode)
    src_ids = [bos] + ko_sp.encode(sentence_for_encode, out_type=int) + [eos]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_ids)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    trg_ids = beam_search_decode_bpe(model, encoder_outputs, hidden, zh_sp, device, max_len=max_len, beam_size=5, length_penalty=0.7)
    zh_bos = int(zh_sp.bos_id())
    zh_eos = int(zh_sp.eos_id())
    zh_pad = int(zh_sp.pad_id())
    filtered = [i for i in trg_ids if i not in (zh_bos, zh_eos, zh_pad)]
    translated_text = zh_sp.decode(filtered) if filtered else ""

    translated_text = apply_replacements(translated_text, replace_rules)
    translated_text = restore_raw_terms_in_output(translated_text, glossary)
    translated_text = dedupe_repeated_ascii_runs(translated_text)
    translated_text = dedupe_repeated_cjk_phrases(translated_text)
    translated_text = apply_output_fallback(translated_text, original_raw_sentence)
    return mark_untranslated_in_output(translated_text)


def translate_sentence(sentence, model, ko_sp, zh_sp, device, user_dict, max_len=80):
    parts = split_by_parentheses(sentence if isinstance(sentence, str) else "")
    if isinstance(sentence, str) and len(parts) > 1:
        out = []
        for p in parts:
            if p.startswith("(") and p.endswith(")"):
                inner = p[1:-1]
                inner_translated = translate_sentence(inner, model, ko_sp, zh_sp, device, user_dict, max_len=max_len)
                out.append(f"({inner_translated})")
                continue
            if p.startswith("（") and p.endswith("）"):
                inner = p[1:-1]
                inner_translated = translate_sentence(inner, model, ko_sp, zh_sp, device, user_dict, max_len=max_len)
                out.append(f"（{inner_translated}）")
                continue
            out.append(translate_sentence_core(p, model, ko_sp, zh_sp, device, user_dict, max_len=max_len))
        return "".join(out)

    return translate_sentence_core(sentence, model, ko_sp, zh_sp, device, user_dict, max_len=max_len)


if __name__ == "__main__":
    if spm is None:
        print(f"无法导入 sentencepiece: {spm_import_error}")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = r"D:\PythonProject\Translate Model\kaggle\v3.2.1_subword\_1st\epoch20_Test Loss 5.9703"
    model_dir = os.path.normpath(str(model_dir).strip().strip('"').strip("'"))
    user_dict_path = os.path.join(os.path.dirname(__file__), "user_dict.md")

    if os.path.isfile(model_dir) and model_dir.lower().endswith(".pth"):
        model_path = model_dir
        model_dir = os.path.dirname(model_path)
    else:
        model_path = os.path.join(model_dir, "best_model_v3_2_1_bpe_attn.pth")

    ko_spm_path = os.path.join(model_dir, "spm_ko_v3_2_1.model")
    zh_spm_path = os.path.join(model_dir, "spm_zh_v3_2_1.model")

    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}")
        raise SystemExit(1)
    if not os.path.exists(ko_spm_path) or not os.path.exists(zh_spm_path):
        print(f"找不到 SentencePiece 模型: {ko_spm_path} / {zh_spm_path}")
        raise SystemExit(1)

    try:
        ko_sp = spm.SentencePieceProcessor(model_file=ko_spm_path)
        zh_sp = spm.SentencePieceProcessor(model_file=zh_spm_path)
    except OSError as e:
        print(f"SentencePiece 加载失败: {e}", flush=True)
        raise SystemExit(1)

    input_dim = ko_sp.get_piece_size()
    output_dim = zh_sp.get_piece_size()
    enc_emb_dim = 256
    dec_emb_dim = 256
    hid_dim = 512

    state_obj = torch.load(model_path, map_location=device)
    if isinstance(state_obj, dict) and "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
        state = state_obj["state_dict"]
    elif isinstance(state_obj, dict) and "model_state_dict" in state_obj and isinstance(state_obj["model_state_dict"], dict):
        state = state_obj["model_state_dict"]
    else:
        state = state_obj
    state = unwrap_state_dict(state)
    n_layers = infer_n_layers_from_state_dict(state)

    attn = Attention(hid_dim)
    enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, 0).to(device)
    dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, 0, attn).to(device)
    model = Seq2Seq(enc, dec, device).to(device)

    model.load_state_dict(state)
    model.eval()
    print("V3.2.1 BPE Attention 模型加载成功！")

    while True:
        sentence = input("\n请输入韩文 (输入 q 退出): ")
        if sentence.lower() == "q":
            break
        if not sentence.strip():
            continue

        user_dict = load_user_dict(user_dict_path)
        token_overrides, direct_translations, replace_rules, glossary = user_dict
        cleaned = clean_text(sentence.replace("／", "/"))
        override_tokens = token_overrides.get(sentence) or token_overrides.get(cleaned)
        bpe_text = " ".join(override_tokens) if override_tokens else cleaned
        english_terms = extract_english_terms(sentence)
        if english_terms:
            print(f"英文: {' '.join(english_terms)}")
        if bpe_text:
            try:
                display_text = strip_english_words_for_encode(bpe_text)
                if re.search(r"(?<![A-Za-z0-9])/(?![A-Za-z0-9])", cleaned):
                    chunks = re.split(r"((?<![A-Za-z0-9])/(?![A-Za-z0-9]))", cleaned)
                    out = []
                    for ch in chunks:
                        if ch == "/":
                            out.append("/")
                            continue
                        part = clean_text(ch)
                        part = strip_english_words_for_encode(part)
                        if not part:
                            continue
                        out.extend(ko_sp.encode(part, out_type=str))
                    print(f"分词(BPE): {' '.join(out)}")
                else:
                    bpe_pieces = ko_sp.encode(display_text, out_type=str) if display_text else []
                    print(f"分词(BPE): {' '.join(bpe_pieces)}")
            except Exception:
                pass

        translation = translate_sentence(sentence, model, ko_sp, zh_sp, device, user_dict)
        print(f"中文翻译: {translation}")
