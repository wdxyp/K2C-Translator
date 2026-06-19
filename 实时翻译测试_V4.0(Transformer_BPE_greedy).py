import argparse
import os
import re
import unicodedata
from contextlib import nullcontext

import torch
import torch.nn as nn

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
    s = s.replace("->", " -> ").replace("...", " ... ")
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


def clean_text(sentence: str) -> str:
    if not isinstance(sentence, str):
        return ""
    s = _normalize_text(sentence).replace("\n", " ")
    kept = []
    for ch in s:
        if ch.isspace() or ("0" <= ch <= "9") or ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            kept.append(ch)
            continue
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3 or 0x4E00 <= code <= 0x9FFF:
            kept.append(ch)
            continue
        cat = unicodedata.category(ch)
        if (cat and cat[0] == "P") or cat in ("Sm", "Sc"):
            kept.append(ch)
            continue
    s = "".join(kept)
    s = _separate_punct_boundaries(s)
    return re.sub(r"\s+", " ", s).strip()


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


def split_by_parentheses(text):
    if not isinstance(text, str) or not text:
        return [text]
    pattern = re.compile(r"(\([^()]*\)|（[^（）]*）)")
    parts = pattern.split(text)
    return [p for p in parts if p is not None and p != ""]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_src_vocab,
        n_trg_vocab,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.2,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.src_embedding = nn.Embedding(int(n_src_vocab), int(d_model))
        self.trg_embedding = nn.Embedding(int(n_trg_vocab), int(d_model))
        self.pos_encoder = PositionalEncoding(int(d_model), float(dropout))
        self.pos_decoder = PositionalEncoding(int(d_model), float(dropout))
        self.transformer = nn.Transformer(
            d_model=int(d_model),
            nhead=int(nhead),
            num_encoder_layers=int(num_encoder_layers),
            num_decoder_layers=int(num_decoder_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
        )
        self.fc_out = nn.Linear(int(d_model), int(n_trg_vocab))
        self.src_pad_idx = 0
        self.trg_pad_idx = 0

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, trg):
        src_mask = None
        trg_mask = self.generate_square_subsequent_mask(trg.size(0)).to(trg.device)
        src_key_padding_mask = (src == self.src_pad_idx).transpose(0, 1)
        trg_key_padding_mask = (trg == self.trg_pad_idx).transpose(0, 1)
        src_emb = self.pos_encoder(self.src_embedding(src) * (self.d_model**0.5))
        trg_emb = self.pos_decoder(self.trg_embedding(trg) * (self.d_model**0.5))
        output = self.transformer(
            src_emb,
            trg_emb,
            src_mask=src_mask,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(output)


def _unwrap_state_dict(state):
    if not isinstance(state, dict):
        return state
    if any(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
        return {k[7:]: v for k, v in state.items() if isinstance(k, str)}
    return state


def _infer_hparams_from_state_dict(state: dict) -> dict:
    d_model = None
    dim_ff = None
    enc_layers = None
    dec_layers = None
    if isinstance(state, dict):
        w = state.get("src_embedding.weight")
        if hasattr(w, "shape") and len(w.shape) == 2:
            d_model = int(w.shape[1])
        w2 = state.get("transformer.encoder.layers.0.linear1.weight")
        if hasattr(w2, "shape") and len(w2.shape) == 2:
            dim_ff = int(w2.shape[0])
        enc_max = -1
        dec_max = -1
        for k in state.keys():
            if not isinstance(k, str):
                continue
            m = re.match(r"^transformer\.encoder\.layers\.(\d+)\.", k)
            if m:
                enc_max = max(enc_max, int(m.group(1)))
            m = re.match(r"^transformer\.decoder\.layers\.(\d+)\.", k)
            if m:
                dec_max = max(dec_max, int(m.group(1)))
        if enc_max >= 0:
            enc_layers = enc_max + 1
        if dec_max >= 0:
            dec_layers = dec_max + 1

    d_model = 256 if d_model is None else int(d_model)
    dim_ff = 1024 if dim_ff is None else int(dim_ff)
    enc_layers = 6 if enc_layers is None else int(enc_layers)
    dec_layers = 6 if dec_layers is None else int(dec_layers)

    prefer = [8, 4, 16, 2, 1]
    nhead = None
    for h in prefer:
        if d_model % h == 0:
            nhead = int(h)
            break
    if nhead is None:
        nhead = 1

    return {"d_model": d_model, "dim_ff": dim_ff, "enc_layers": enc_layers, "dec_layers": dec_layers, "nhead": nhead}


def _resolve_v4_paths(model_dir: str) -> tuple[str, str, str]:
    model_dir = os.path.normpath(str(model_dir).strip().strip('"').strip("'"))
    best_model = os.path.join(model_dir, "best_model_v4_transformer.pth")
    ko_spm_path = os.path.join(model_dir, "spm_ko_v4.model")
    zh_spm_path = os.path.join(model_dir, "spm_zh_v4.model")

    if os.path.isfile(model_dir) and model_dir.lower().endswith(".pth"):
        best_model = model_dir
        model_dir = os.path.dirname(best_model)
        ko_spm_path = os.path.join(model_dir, "spm_ko_v4.model")
        zh_spm_path = os.path.join(model_dir, "spm_zh_v4.model")

    if (not os.path.exists(best_model)) and os.path.isdir(model_dir):
        cands = []
        for root, _, files in os.walk(model_dir):
            for fn in files:
                if fn == "best_model_v4_transformer.pth":
                    cands.append(os.path.join(root, fn))
        if cands:
            best_model = sorted(cands, key=lambda p: os.path.getmtime(p))[-1]
            model_dir = os.path.dirname(best_model)
            ko_spm_path = os.path.join(model_dir, "spm_ko_v4.model")
            zh_spm_path = os.path.join(model_dir, "spm_zh_v4.model")

    return best_model, ko_spm_path, zh_spm_path


def load_v4_transformer_from_dir(model_dir: str, device: torch.device):
    if spm is None:
        raise RuntimeError(f"无法导入 sentencepiece: {spm_import_error}")

    model_path, ko_spm_path, zh_spm_path = _resolve_v4_paths(model_dir)
    if not os.path.exists(model_path):
        raise RuntimeError(f"找不到模型文件: {model_path}")
    if not os.path.exists(ko_spm_path) or not os.path.exists(zh_spm_path):
        raise RuntimeError(f"找不到 SentencePiece 模型: {ko_spm_path} / {zh_spm_path}")

    ko_sp = spm.SentencePieceProcessor(model_file=ko_spm_path)
    zh_sp = spm.SentencePieceProcessor(model_file=zh_spm_path)

    state_obj = torch.load(model_path, map_location="cpu")
    if isinstance(state_obj, dict) and "model_state_dict" in state_obj and isinstance(state_obj["model_state_dict"], dict):
        state = state_obj["model_state_dict"]
    else:
        state = state_obj
    state = _unwrap_state_dict(state)

    hp = _infer_hparams_from_state_dict(state if isinstance(state, dict) else {})
    model = TransformerModel(
        n_src_vocab=int(ko_sp.get_piece_size()),
        n_trg_vocab=int(zh_sp.get_piece_size()),
        d_model=int(hp["d_model"]),
        nhead=int(hp["nhead"]),
        num_encoder_layers=int(hp["enc_layers"]),
        num_decoder_layers=int(hp["dec_layers"]),
        dim_feedforward=int(hp["dim_ff"]),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ko_sp, zh_sp, model_path


def _greedy_decode_transformer(model: nn.Module, src_ids: list[int], zh_sp, device: torch.device, max_len: int) -> list[int]:
    src = torch.LongTensor(src_ids).unsqueeze(1).to(device)
    bos_id = int(zh_sp.bos_id())
    eos_id = int(zh_sp.eos_id())
    trg = torch.LongTensor([bos_id]).unsqueeze(1).to(device)

    autocast_ctx = (
        torch.amp.autocast("cuda", enabled=(device.type == "cuda"))
        if hasattr(torch, "amp")
        else (torch.cuda.amp.autocast(enabled=(device.type == "cuda")) if device.type == "cuda" else nullcontext())
    )
    with torch.no_grad(), autocast_ctx:
        for _ in range(int(max_len)):
            out = model(src, trg)
            next_id = int(out[-1, 0].argmax(dim=-1).item())
            trg = torch.cat([trg, torch.LongTensor([[next_id]]).to(device)], dim=0)
            if next_id == eos_id:
                break
    return trg.squeeze(1).tolist()


def translate_sentence_core(sentence, model, ko_sp, zh_sp, device, user_dict, max_len=80):
    original_raw_sentence = sentence if isinstance(sentence, str) else ""
    sentence = original_raw_sentence.replace("／", "/")
    token_overrides, direct_translations, replace_rules, glossary = user_dict

    cleaned = clean_text(sentence)
    if not cleaned:
        return ""

    if cleaned in direct_translations:
        translated_text = direct_translations[cleaned]
        translated_text = apply_replacements(translated_text, replace_rules)
        translated_text = restore_raw_terms_in_output(translated_text, glossary)
        translated_text = dedupe_repeated_ascii_runs(translated_text)
        translated_text = dedupe_repeated_cjk_phrases(translated_text)
        translated_text = apply_output_fallback(translated_text, original_raw_sentence)
        return mark_untranslated_in_output(translated_text)

    override_tokens = token_overrides.get(original_raw_sentence) or token_overrides.get(cleaned)
    sentence_for_encode = " ".join(override_tokens) if override_tokens else cleaned

    bos = int(ko_sp.bos_id())
    eos = int(ko_sp.eos_id())
    src_ids = [bos] + ko_sp.encode(sentence_for_encode, out_type=int) + [eos]

    trg_ids = _greedy_decode_transformer(model, src_ids, zh_sp, device, max_len=max_len)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=r"D:\PythonProject\Translate Model\Google_colab\V4.0_Tranformer\_1st\Epoch30_test_loss 6.5368",
    )
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args, _unknown = parser.parse_known_args()

    if spm is None:
        print(f"无法导入 sentencepiece: {spm_import_error}")
        raise SystemExit(1)

    model_dir = os.path.normpath(str(args.model_dir).strip().strip('"').strip("'"))
    if not model_dir:
        model_dir = input("请输入模型目录(model_dir): ").strip()
    model_dir = os.path.normpath(str(model_dir).strip().strip('"').strip("'"))

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ko_sp, zh_sp, model_path = load_v4_transformer_from_dir(model_dir, device=device)
    print("V4 Transformer 模型加载成功！")
    print("model:", model_path)
    print("spm_ko:", os.path.join(os.path.dirname(model_path), "spm_ko_v4.model"))
    print("spm_zh:", os.path.join(os.path.dirname(model_path), "spm_zh_v4.model"))
    print("device:", str(device))

    user_dict_path = os.path.join(os.path.dirname(__file__), "user_dict.md")
    while True:
        sentence = input("\n请输入韩文 (输入 q 退出): ")
        if sentence.lower() == "q":
            break
        if not sentence.strip():
            continue
        user_dict = load_user_dict(user_dict_path)
        translation = translate_sentence(sentence, model, ko_sp, zh_sp, device, user_dict, max_len=int(args.max_len))
        print(f"中文翻译: {translation}")


if __name__ == "__main__":
    main()
