import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import re
from konlpy.tag import Okt
import jieba

# --- 1. 文本清洗 (必须与训练代码一致) ---
def clean_text(sentence):
    if not isinstance(sentence, str): return ""
    sentence = re.sub(r'[^\w\s\uAC00-\uD7A3\u4e00-\u9fa5]', '', sentence)
    return sentence.strip()

def load_user_dict(md_path):
    token_overrides = {}
    direct_translations = {}
    replace_rules = {}
    glossary = {}
    model_only_terms = set()

    if not os.path.exists(md_path):
        return token_overrides, direct_translations, replace_rules, glossary, model_only_terms

    section = 'glossary'
    with open(md_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('##'):
                header = line.lstrip('#').strip()
                if '分词' in header:
                    section = 'tokenize'
                elif '直译' in header:
                    section = 'translate'
                elif '替换' in header:
                    section = 'replace'
                elif '术语' in header:
                    section = 'glossary'
                else:
                    section = None
                continue
            if line.startswith('#'):
                header = line.lstrip('#').strip()
                if '术语' in header:
                    section = 'glossary'
                continue
            if not line.startswith('- '):
                continue
            content = line[2:]
            sep_pos_ascii = content.find(':')
            sep_pos_full = content.find('：')
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

            if section == 'tokenize':
                token_overrides[clean_text(left)] = [t for t in right.split() if t]
            elif section == 'translate':
                direct_translations[clean_text(left)] = right
            elif section == 'replace':
                replace_rules[left] = right
            elif section == 'glossary':
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


def load_revision_dict(md_path):
    replace_rules = {}
    if not os.path.exists(md_path):
        return replace_rules

    with open(md_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or not line.startswith("- "):
                continue
            content = line[2:]
            sep_pos_ascii = content.find(":")
            sep_pos_full = content.find("：")
            sep_positions = [p for p in (sep_pos_ascii, sep_pos_full) if p != -1]
            if not sep_positions:
                continue
            sep_pos = min(sep_positions)
            left, right = content[:sep_pos].strip(), content[sep_pos + 1 :].strip()
            if left and right:
                replace_rules[left] = right
    return replace_rules

def apply_replacements(text, replace_rules):
    if not replace_rules:
        return text
    for src, dst in replace_rules.items():
        if src:
            text = text.replace(src, dst)
    return text

def apply_glossary_merge(text, zh_terms):
    if not zh_terms:
        return text
    unique_terms = []
    for t in zh_terms:
        if t and t not in unique_terms:
            unique_terms.append(t)
    append_terms = [t for t in unique_terms if t not in text]
    if not append_terms:
        return text
    return f"{text}（术语：{'；'.join(append_terms)}）"

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
    pattern = re.compile(r"([\u4e00-\u9fa5]{1,8})(?:\1)+")
    while True:
        new_text = pattern.sub(r"\1", text)
        if new_text == text:
            return text
        text = new_text

def dedupe_shifou_pattern(text):
    if not isinstance(text, str) or not text:
        return text
    pattern = re.compile(r"([\u4e00-\u9fa5]{1,6})是否\1")
    while True:
        new_text = pattern.sub(r"是否\1", text)
        if new_text == text:
            return text
        text = new_text

def ensure_preserve_literals(translated_text, original_text):
    if not isinstance(translated_text, str) or not isinstance(original_text, str):
        return translated_text

    src = original_text.replace("／", "/")
    m_enum = re.match(r"\s*(\(\s*\d+\s*\)|（\s*\d+\s*）|\d+(?:\)|\.))", src)
    leading_enum = m_enum.group(1) if m_enum else None
    terms = []

    patterns = [
        r"\d+(?:\.\d+)?\s*℃",
        r"\(\s*\d+\s*\)",
        r"（\s*\d+\s*）",
        r"\d+\)",
        r"\d+\.",
        r"\d+(?:\.\d+)?\s*[~\-]\s*\d+(?:\.\d+)?\s*(?:mm|cm|m|℃|%|V|A|Hz|rpm)?",
        r"\d+(?:\.\d+)?\s*(?:mm|cm|m|kg|g|%|V|A|Hz|rpm|℃)",
        r"\d+(?:\.\d+)?\s*[eE][aA]",
        r"[:：~]",
    ]
    for pat in patterns:
        for m in re.finditer(pat, src):
            t = m.group(0)
            if t and t not in terms:
                terms.append(t)

    if not terms:
        return translated_text

    text = translated_text
    for t in terms:
        if re.fullmatch(r"\d+(?:\.\d+)?\s*ea", t, flags=re.IGNORECASE) and re.search(r"(?<![A-Za-z0-9])ea(?![A-Za-z0-9])", text, flags=re.IGNORECASE):
            text = re.sub(r"(?<![A-Za-z0-9])ea(?![A-Za-z0-9])", t, text, count=1, flags=re.IGNORECASE)
            continue
        if t in text:
            continue
        if text and not text.endswith((" ", "/", "（")):
            text += " "
        text += t

    if leading_enum:
        text = text.strip()
        if not text.startswith(leading_enum):
            text_wo_enum = re.sub(rf"\s*{re.escape(leading_enum)}\s*", " ", text).strip()
            text_wo_enum = re.sub(r"\s{2,}", " ", text_wo_enum)
            text = f"{leading_enum} {text_wo_enum}".strip()
    return text


def ensure_leading_enumeration(translated_text, original_text):
    if not isinstance(translated_text, str) or not isinstance(original_text, str):
        return translated_text
    m_enum = re.match(r"\s*(\(\s*\d+\s*\)|（\s*\d+\s*）|\d+(?:\)|\.))", original_text)
    leading_enum = m_enum.group(1) if m_enum else None
    if not leading_enum:
        return translated_text
    text = translated_text.strip()
    if text.startswith(leading_enum):
        return translated_text
    text_wo_enum = re.sub(rf"\s*{re.escape(leading_enum)}\s*", " ", text).strip()
    text_wo_enum = re.sub(r"\s{2,}", " ", text_wo_enum)
    return f"{leading_enum} {text_wo_enum}".strip()

def ensure_preserve_english_terms(translated_text, original_text):
    if not isinstance(translated_text, str) or not isinstance(original_text, str):
        return translated_text

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

    if not terms:
        return translated_text

    text = translated_text
    for t in terms:
        if t in text:
            continue
        m = re.search(r"(是否|有无|能否)", text)
        if m:
            insert_pos = m.start(1)
            before = text[:insert_pos].rstrip()
            after = text[insert_pos:].lstrip()
            if before and not before.endswith((" ", "/", "（", "(", "," , "，")):
                before += " "
            text = f"{before}{t} {after}".strip()
        else:
            if text and not text.endswith((" ", "/", "（")):
                text += " "
            text += t
    return text

def strip_english_terms_for_model(text):
    if not isinstance(text, str) or not text:
        return text
    s = text.replace("／", "/")
    s = re.sub(r"[A-Za-z0-9]+/[A-Za-z0-9]+", " ", s)
    s = re.sub(r"[A-Za-z][A-Za-z0-9]*", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _add_zh_token_or_chars(zh_vocab, allowed_ids, s):
    if not isinstance(s, str) or not s:
        return
    if s in zh_vocab:
        allowed_ids.add(zh_vocab[s])
        return
    for ch in s:
        if ch in zh_vocab:
            allowed_ids.add(zh_vocab[ch])

def mark_untranslated_in_output(text):
    if not isinstance(text, str) or not text:
        return text
    marked = re.sub(r"[\uAC00-\uD7A3]+", "[?]", text)
    marked = re.sub(r"(?:\[\?\]){2,}", "[?]", marked)
    return marked

COMMON_DECISION_APPEND = {
    "여부": {"zh": "与否", "blockers": ("与否", "是否", "有无", "能否")},
    "유무": {"zh": "有无", "blockers": ("有无", "是否", "能否")},
}

DECISION_REWRITE_CONFIG = [
    {
        "ko": "여부",
        "zh_append": "与否",
        "rewrite": "是否{pred}",
        "blockers": ("是否", "有无", "能否"),
        "pred_max_zh": 8,
    },
    {
        "ko": "유무",
        "zh_append": "有无",
        "rewrite": "是否有{pred}",
        "blockers": ("是否", "能否"),
        "pred_max_zh": 12,
    },
]

PREDICATE_LEXICON = [
    {"zh": "松动"},
    {"zh": "异常"},
    {"zh": "正常"},
    {"zh": "不良"},
    {"zh": "接触不良"},
    {"zh": "破损"},
    {"zh": "损坏"},
    {"zh": "磨损"},
    {"zh": "刮伤"},
    {"zh": "变形"},
    {"zh": "脱落"},
    {"zh": "缺失"},
    {"zh": "漏油"},
    {"zh": "漏气"},
    {"zh": "漏水"},
    {"zh": "渗漏"},
    {"zh": "堵塞"},
    {"zh": "卡滞"},
    {"zh": "断线"},
    {"zh": "断裂"},
    {"zh": "开路"},
    {"zh": "短路"},
    {"zh": "存在"},
    {"zh": "不存在"},
    {"zh": "有"},
    {"zh": "无"},
    {"zh": "安装", "ko": ["설치", "장착", "부착"]},
    {"zh": "组装", "ko": ["조립"]},
    {"zh": "紧固", "ko": ["체결"]},
    {"zh": "更换", "ko": ["교체"]},
    {"zh": "检查", "ko": ["점검"]},
    {"zh": "确认", "ko": ["확인"]},
    {"zh": "测量", "ko": ["측정"]},
    {"zh": "运行", "ko": ["작동", "구동", "운전"]},
]

PREDICATE_SUFFIXES = []
PREDICATE_KO_TO_ZH = {}
for _entry in PREDICATE_LEXICON:
    _zh = _entry.get("zh") if isinstance(_entry, dict) else None
    if isinstance(_zh, str) and _zh and _zh not in PREDICATE_SUFFIXES:
        PREDICATE_SUFFIXES.append(_zh)
    _kos = _entry.get("ko") if isinstance(_entry, dict) else None
    if isinstance(_kos, list) and isinstance(_zh, str) and _zh:
        for _ko in _kos:
            if isinstance(_ko, str) and _ko:
                PREDICATE_KO_TO_ZH[_ko] = _zh

DECISION_EQUIVALENT_FORMS = {
    "여부": ["여부", "여·부", "여/부", "여 부", "여  부"],
    "유무": ["유무", "유·무", "유/무", "유 무", "유  무"],
}

def normalize_decision_markers(text):
    if not isinstance(text, str) or not text:
        return text
    s = re.sub(r"\s+", " ", text)
    for canonical, variants in DECISION_EQUIVALENT_FORMS.items():
        if not isinstance(canonical, str) or not canonical:
            continue
        if not isinstance(variants, list):
            continue
        for v in variants:
            if isinstance(v, str) and v:
                s = s.replace(v, canonical)
    return s

def ensure_append_common_terms(translated_text, original_text):
    if not isinstance(translated_text, str) or not isinstance(original_text, str):
        return translated_text
    text = translated_text
    normalized_original = normalize_decision_markers(original_text)

    def _insert_before_tail(s, insert_text):
        m_tail = re.search(r"(\s*(?:\[\?\])+\s*)$", s)
        tail = m_tail.group(1) if m_tail else ""
        body = s[: -len(tail)] if tail else s
        body = body.rstrip()

        m_punct = re.search(r"([。！？!?；;,.，:：])\s*$", body)
        if m_punct:
            punct = m_punct.group(1)
            body_wo = body[: m_punct.start(1)].rstrip()
            return f"{body_wo} {insert_text}{punct}{tail}".strip()

        if not body:
            return f"{insert_text}{tail}".strip()
        if body.endswith((" ", "/", "（", "(", "→", "," , "，")):
            return f"{body}{insert_text}{tail}".strip()
        return f"{body} {insert_text}{tail}".strip()

    for ko, cfg in COMMON_DECISION_APPEND.items():
        if ko not in normalized_original:
            continue
        zh = cfg.get("zh")
        if not isinstance(zh, str) or not zh:
            continue
        if zh in text:
            continue
        blockers = cfg.get("blockers", ())
        if blockers and any(b in text for b in blockers):
            continue
        text = _insert_before_tail(text, zh)
    return text

def apply_decision_rewrite_rules(translated_text, original_text, tokens_display, glossary):
    if not isinstance(translated_text, str) or not isinstance(original_text, str):
        return translated_text, False
    normalized_original = normalize_decision_markers(original_text)

    def _insert_before_tail(body, insert_text):
        m_tail = re.search(r"(\s*(?:\[\?\])+\s*)$", body)
        tail = m_tail.group(1) if m_tail else ""
        core = body[: -len(tail)] if tail else body
        core = core.rstrip()

        m_punct = re.search(r"([。！？!?；;,.，:：])\s*$", core)
        if m_punct:
            punct = m_punct.group(1)
            core_wo = core[: m_punct.start(1)].rstrip()
            return f"{core_wo} {insert_text}{punct}{tail}".strip()

        if not core:
            return f"{insert_text}{tail}".strip()
        if core.endswith((" ", "/", "（", "(", "→", ",", "，")):
            return f"{core}{insert_text}{tail}".strip()
        return f"{core} {insert_text}{tail}".strip()

    def _find_predicate_ko(tokens, ko_suffix):
        if not isinstance(tokens, list) or not tokens:
            return None
        try:
            idx = len(tokens) - 1 - tokens[::-1].index(ko_suffix)
        except ValueError:
            idx = -1
        if idx > 0 and isinstance(tokens[idx - 1], str):
            return tokens[idx - 1]
        for tok in reversed(tokens):
            if isinstance(tok, str) and tok.endswith(ko_suffix) and tok != ko_suffix and len(tok) > len(ko_suffix):
                return tok[: -len(ko_suffix)]
        return None

    def _find_predicate_zh_from_text(text, zh_suffix, max_len):
        if not isinstance(text, str) or not text:
            return None
        if isinstance(zh_suffix, str) and zh_suffix:
            m = re.search(rf"([\u4e00-\u9fa5]{{1,{max_len}}})\s*{re.escape(zh_suffix)}", text)
            if m:
                return m.group(1)
        s = re.sub(rf"\s*{re.escape(zh_suffix)}\s*$", "", text).strip() if zh_suffix else text.strip()
        m = re.search(rf"([\u4e00-\u9fa5]{{1,{max_len}}})\s*$", s)
        return m.group(1) if m else None

    def _normalize_predicate_zh(pred_zh, max_len):
        if not isinstance(pred_zh, str) or not pred_zh:
            return pred_zh
        for suf in PREDICATE_SUFFIXES:
            if isinstance(suf, str) and suf and pred_zh.endswith(suf):
                return suf
        try:
            parts = [p.strip() for p in jieba.lcut(pred_zh) if isinstance(p, str) and p.strip()]
        except Exception:
            parts = []
        if parts:
            last = parts[-1]
            if last in PREDICATE_SUFFIXES:
                return last
            if 0 < len(last) <= max_len:
                return last
        if len(pred_zh) <= max_len:
            return pred_zh
        return pred_zh[-max_len:]

    for cfg in DECISION_REWRITE_CONFIG:
        ko = cfg.get("ko")
        zh_append = cfg.get("zh_append", "")
        rewrite_tpl = cfg.get("rewrite")
        blockers = cfg.get("blockers", ())
        pred_max_zh = int(cfg.get("pred_max_zh", 12))

        if not isinstance(ko, str) or not ko or ko not in normalized_original:
            continue
        if blockers and any(b in translated_text for b in blockers):
            continue
        if isinstance(zh_append, str) and zh_append and zh_append not in translated_text:
            continue
        if not isinstance(rewrite_tpl, str) or "{pred}" not in rewrite_tpl:
            continue

        pred_zh = None
        pred_ko = _find_predicate_ko(tokens_display, ko)
        if (
            pred_ko is not None
            and isinstance(glossary, dict)
            and isinstance(pred_ko, str)
            and pred_ko in glossary
            and isinstance(glossary[pred_ko], str)
            and glossary[pred_ko]
        ):
            pred_zh = glossary[pred_ko]
        if not pred_zh and isinstance(pred_ko, str) and pred_ko:
            mapped = PREDICATE_KO_TO_ZH.get(pred_ko)
            if isinstance(mapped, str) and mapped:
                pred_zh = mapped

        if not pred_zh:
            pred_zh = _find_predicate_zh_from_text(translated_text, zh_append, pred_max_zh)
        if not pred_zh:
            continue
        pred_zh = _normalize_predicate_zh(pred_zh, pred_max_zh)

        s = translated_text
        if isinstance(zh_append, str) and zh_append:
            s = re.sub(rf"\s*{re.escape(zh_append)}\s*", " ", s).strip()
        if s.endswith(pred_zh):
            s = s[: -len(pred_zh)].rstrip()

        decision_phrase = rewrite_tpl.format(pred=pred_zh)
        return _insert_before_tail(s, decision_phrase), True

    return translated_text, False

def _is_korean_token(token):
    return isinstance(token, str) and bool(re.search(r"[\uAC00-\uD7A3]", token))

def _is_ascii_or_number_token(token):
    return isinstance(token, str) and bool(re.fullmatch(r"[A-Za-z0-9]+", token))

def _is_unit_token(token):
    return isinstance(token, str) and token.lower() in {"mm", "cm", "m", "kg", "g", "v", "a", "hz", "rpm", "%", "°c"}

def merge_single_hangul_prefix_tokens(tokens, original_sentence):
    if not isinstance(original_sentence, str) or not tokens:
        return tokens
    sent = clean_text(original_sentence)
    if not sent:
        return tokens
    merged = []
    i = 0
    n = len(tokens)
    while i < n:
        if i + 1 < n:
            t1 = tokens[i]
            t2 = tokens[i + 1]
            if (
                isinstance(t1, str)
                and isinstance(t2, str)
                and re.fullmatch(r"[\uAC00-\uD7A3]", t1)
                and re.search(r"[\uAC00-\uD7A3]", t2)
            ):
                cand = f"{t1}{t2}"
                if cand in sent:
                    merged.append(cand)
                    i += 2
                    continue
        merged.append(tokens[i])
        i += 1
    return merged

def filter_model_tokens(tokens):
    filtered = []
    for t in tokens:
        if _is_korean_token(t):
            filtered.append(t)
            continue
        if _is_ascii_or_number_token(t):
            continue
        if _is_unit_token(t):
            continue
    return filtered

def merge_tokens_with_glossary(tokens, glossary, max_window=5):
    if not tokens or not glossary:
        return tokens, []

    terms = []
    for k in glossary.keys():
        if not isinstance(k, str) or not k:
            continue
        kc = clean_text(k)
        if kc and re.fullmatch(r"[\uAC00-\uD7A3]{2,}", kc):
            terms.append(kc)
    if not terms:
        return tokens, []

    term_set = set(terms)
    hit_terms = []
    merged = []

    i = 0
    n = len(tokens)
    while i < n:
        matched = False
        end_limit = min(n, i + max_window)
        for j in range(end_limit, i + 1, -1):
            cand = "".join(tokens[i:j])
            if cand in term_set:
                if cand not in hit_terms:
                    hit_terms.append(cand)
                merged.append(cand)
                i = j
                matched = True
                break
        if matched:
            continue
        merged.append(tokens[i])
        i += 1

    return merged, hit_terms

def force_glossary_zh_in_output(text, hit_terms, glossary):
    if not isinstance(text, str) or not text or not hit_terms or not glossary:
        return text
    out = text
    for t in hit_terms:
        zh = glossary.get(t)
        if not isinstance(zh, str) or not zh:
            continue
        if zh in out:
            continue
        if out.startswith(zh):
            continue
        out = f"{zh} {out}"
    return out.strip()

def split_by_parentheses(text):
    if not isinstance(text, str) or not text:
        return [text]
    pattern = re.compile(r"(\([^()]*\)|（[^（）]*）)")
    parts = pattern.split(text)
    return [p for p in parts if p is not None and p != ""]

def should_append_unknown_marker(tokens, glossary, unk_tokens_exist, translated_text):
    ko_tokens = [t for t in tokens if _is_korean_token(t)]
    ko_tokens_not_in_glossary = [t for t in ko_tokens if t not in glossary]
    if not ko_tokens_not_in_glossary:
        return False

    if unk_tokens_exist:
        return True

    if not isinstance(translated_text, str) or not translated_text.strip():
        return True

    zh_char_count = len(re.findall(r"[\u4e00-\u9fa5]", translated_text))
    return zh_char_count < len(ko_tokens)

def protect_glossary_terms(sentence, glossary):
    if not glossary:
        return sentence, {}, []

    marker_to_term = {}
    hit_zh_terms = []
    protected_sentence = sentence
    terms = sorted(glossary.keys(), key=len, reverse=True)
    for i, term in enumerate(terms):
        if not term:
            continue
        if term not in protected_sentence:
            continue
        marker = f"__TERM{i}__"
        protected_sentence = protected_sentence.replace(term, marker)
        marker_to_term[marker] = term
        hit_zh_terms.append(glossary[term])
    return protected_sentence, marker_to_term, hit_zh_terms

def beam_search_decode(
    model,
    encoder_outputs,
    hidden,
    zh_vocab,
    device,
    max_len=50,
    beam_size=5,
    length_penalty=0.7,
    allowed_token_ids=None,
    penalize_token_ids=None,
    penalize_value=0.0,
):
    sos_id = zh_vocab["<sos>"]
    eos_id = zh_vocab["<eos>"]
    no_repeat_ngram_size = 4
    repeat_token_penalty = 2.0

    allowed_tensor = None
    if allowed_token_ids:
        try:
            allowed_tensor = torch.LongTensor(list(set(allowed_token_ids))).to(device)
        except Exception:
            allowed_tensor = None

    penalize_tensor = None
    if penalize_token_ids and float(penalize_value) > 0:
        try:
            penalize_tensor = torch.LongTensor(list(set(penalize_token_ids))).to(device)
        except Exception:
            penalize_tensor = None

    def rank_score(total_logprob, seq_len):
        seq_len = max(1, seq_len)
        return total_logprob / (seq_len ** length_penalty)

    def would_repeat_ngram(tokens, next_token_id):
        if no_repeat_ngram_size <= 0:
            return False
        if len(tokens) + 1 < no_repeat_ngram_size:
            return False
        n = no_repeat_ngram_size
        new_ngram = tuple(tokens[-(n - 1):] + [next_token_id])
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
            if penalize_tensor is not None:
                log_probs[penalize_tensor] -= float(penalize_value)
            if allowed_tensor is not None:
                mask = torch.full_like(log_probs, -1e9)
                mask[allowed_tensor] = 0.0
                log_probs = log_probs + mask
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
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
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
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = top1
        return outputs

# --- 3. 翻译函数 ---
def translate_sentence(sentence, model, ko_vocab, zh_vocab, device, user_dict, max_len=50):
    parts = split_by_parentheses(sentence if isinstance(sentence, str) else "")
    if isinstance(sentence, str) and len(parts) > 1:
        out = []
        for p in parts:
            if p.startswith("(") and p.endswith(")"):
                if re.fullmatch(r"\(\s*\d+\s*\)", p):
                    out.append(p)
                    continue
                inner = p[1:-1]
                inner_translated = translate_sentence(inner, model, ko_vocab, zh_vocab, device, user_dict, max_len=max_len)
                out.append(f"({inner_translated})")
                continue
            if p.startswith("（") and p.endswith("）"):
                if re.fullmatch(r"（\s*\d+\s*）", p):
                    out.append(p)
                    continue
                inner = p[1:-1]
                inner_translated = translate_sentence(inner, model, ko_vocab, zh_vocab, device, user_dict, max_len=max_len)
                out.append(f"（{inner_translated}）")
                continue
            out.append(translate_sentence_core(p, model, ko_vocab, zh_vocab, device, user_dict, max_len=max_len))
        return "".join(out)

    return translate_sentence_core(sentence, model, ko_vocab, zh_vocab, device, user_dict, max_len=max_len)

def translate_sentence_core(sentence, model, ko_vocab, zh_vocab, device, user_dict, max_len=50):
    model.eval()
    token_overrides, direct_translations, replace_rules, glossary, model_only_terms = user_dict

    original_raw_sentence = sentence.replace("／", "/") if isinstance(sentence, str) else ""
    raw_sentence = original_raw_sentence if original_raw_sentence else sentence
    hit_zh_terms = []
    force_hit_terms = []
    if isinstance(raw_sentence, str) and glossary:
        clean_raw = clean_text(raw_sentence)
        if clean_raw:
            for term in sorted(glossary.keys(), key=lambda x: len(str(x)), reverse=True):
                if not isinstance(term, str) or not term:
                    continue
                t = clean_text(term)
                if not t or len(t) < 2:
                    continue
                if model_only_terms and (term in model_only_terms or t in model_only_terms):
                    continue
                if t in clean_raw and t not in force_hit_terms:
                    force_hit_terms.append(t)

    if isinstance(raw_sentence, str) and "→" in raw_sentence:
        chunks = re.split(r"(→)", raw_sentence)
        out = []
        for ch in chunks:
            if ch == "→":
                out.append(ch)
                continue
            if ch == "" or not ch.strip():
                out.append(ch)
                continue
            out.append(translate_sentence(ch, model, ko_vocab, zh_vocab, device, user_dict, max_len=max_len))
        return "".join(out)

    if isinstance(raw_sentence, str) and ("," in raw_sentence or "，" in raw_sentence):
        comma_pat = r"(?<!\d),(?!\d)|，"
        delims = re.findall(comma_pat, raw_sentence)
        if len(delims) == 1:
            m = re.search(comma_pat, raw_sentence)
            left = raw_sentence[: m.start()] if m else raw_sentence
            delim = m.group(0) if m else ","
            right = raw_sentence[m.end() :] if m else ""
            if re.search(r"[\uAC00-\uD7A3]", left) and re.search(r"[\uAC00-\uD7A3]", right):
                left_tr = translate_sentence(left, model, ko_vocab, zh_vocab, device, user_dict, max_len=max_len)
                right_tr = translate_sentence(right, model, ko_vocab, zh_vocab, device, user_dict, max_len=max_len)
                return f"{left_tr}{delim}{right_tr}"

        chunks = re.split(r"((?<!\d),(?!\d)|，)", raw_sentence)
        out = []
        for ch in chunks:
            if ch in (",", "，"):
                out.append(ch)
                continue
            if ch == "" or not ch.strip():
                out.append(ch)
                continue
            out.append(translate_sentence(ch, model, ko_vocab, zh_vocab, device, user_dict, max_len=max_len))
        return "".join(out)

    if isinstance(raw_sentence, str) and re.search(r"(?<![A-Za-z0-9])/(?![A-Za-z0-9])", raw_sentence):
        chunks = re.split(r"((?<![A-Za-z0-9])/(?![A-Za-z0-9]))", raw_sentence)
        out = []
        for ch in chunks:
            if ch == "/":
                out.append(ch)
                continue
            if ch == "" or not ch.strip():
                out.append(ch)
                continue
            out.append(translate_sentence(ch, model, ko_vocab, zh_vocab, device, user_dict, max_len=max_len))
        return "".join(out)

    raw_sentence_no_en = strip_english_terms_for_model(raw_sentence)
    sentence_full = clean_text(raw_sentence)
    sentence = clean_text(raw_sentence_no_en)

    if model_only_terms and (sentence_full in model_only_terms or sentence in model_only_terms):
        pass
    elif sentence_full in direct_translations:
        translated_text = apply_replacements(direct_translations[sentence_full], replace_rules)
        translated_text = restore_raw_terms_in_output(translated_text, glossary)
        translated_text = dedupe_repeated_ascii_runs(translated_text)
        translated_text = dedupe_repeated_cjk_phrases(translated_text)
        translated_text = ensure_preserve_english_terms(translated_text, original_raw_sentence)
        translated_text = ensure_preserve_literals(translated_text, original_raw_sentence)
        translated_text = force_glossary_zh_in_output(translated_text, [sentence_full] + force_hit_terms, glossary)
        translated_text = ensure_leading_enumeration(translated_text, original_raw_sentence)
        translated_text = ensure_append_common_terms(translated_text, original_raw_sentence)
        translated_text, _ = apply_decision_rewrite_rules(translated_text, original_raw_sentence, None, glossary)
        return mark_untranslated_in_output(apply_glossary_merge(translated_text, hit_zh_terms))

    if model_only_terms and (sentence_full in model_only_terms or sentence in model_only_terms):
        pass
    elif sentence in direct_translations:
        translated_text = apply_replacements(direct_translations[sentence], replace_rules)
        translated_text = restore_raw_terms_in_output(translated_text, glossary)
        translated_text = dedupe_repeated_ascii_runs(translated_text)
        translated_text = dedupe_repeated_cjk_phrases(translated_text)
        translated_text = ensure_preserve_english_terms(translated_text, original_raw_sentence)
        translated_text = ensure_preserve_literals(translated_text, original_raw_sentence)
        translated_text = force_glossary_zh_in_output(translated_text, [sentence] + force_hit_terms, glossary)
        translated_text = ensure_leading_enumeration(translated_text, original_raw_sentence)
        translated_text = ensure_append_common_terms(translated_text, original_raw_sentence)
        translated_text, _ = apply_decision_rewrite_rules(translated_text, original_raw_sentence, None, glossary)
        return mark_untranslated_in_output(apply_glossary_merge(translated_text, hit_zh_terms))

    if model_only_terms and (sentence_full in model_only_terms or sentence in model_only_terms):
        pass
    elif sentence_full in glossary:
        translated_text = apply_replacements(glossary[sentence_full], replace_rules)
        translated_text = restore_raw_terms_in_output(translated_text, glossary)
        translated_text = dedupe_repeated_ascii_runs(translated_text)
        translated_text = dedupe_repeated_cjk_phrases(translated_text)
        translated_text = ensure_preserve_english_terms(translated_text, original_raw_sentence)
        translated_text = ensure_preserve_literals(translated_text, original_raw_sentence)
        translated_text = force_glossary_zh_in_output(translated_text, [sentence_full] + force_hit_terms, glossary)
        translated_text = ensure_leading_enumeration(translated_text, original_raw_sentence)
        translated_text = ensure_append_common_terms(translated_text, original_raw_sentence)
        translated_text, _ = apply_decision_rewrite_rules(translated_text, original_raw_sentence, None, glossary)
        return mark_untranslated_in_output(apply_glossary_merge(translated_text, hit_zh_terms))

    if model_only_terms and (sentence_full in model_only_terms or sentence in model_only_terms):
        pass
    elif sentence in glossary:
        translated_text = apply_replacements(glossary[sentence], replace_rules)
        translated_text = restore_raw_terms_in_output(translated_text, glossary)
        translated_text = dedupe_repeated_ascii_runs(translated_text)
        translated_text = dedupe_repeated_cjk_phrases(translated_text)
        translated_text = ensure_preserve_english_terms(translated_text, original_raw_sentence)
        translated_text = ensure_preserve_literals(translated_text, original_raw_sentence)
        translated_text = force_glossary_zh_in_output(translated_text, [sentence] + force_hit_terms, glossary)
        translated_text = ensure_leading_enumeration(translated_text, original_raw_sentence)
        translated_text = ensure_append_common_terms(translated_text, original_raw_sentence)
        translated_text, _ = apply_decision_rewrite_rules(translated_text, original_raw_sentence, None, glossary)
        return mark_untranslated_in_output(apply_glossary_merge(translated_text, hit_zh_terms))
    
    # 分词
    if sentence_full in token_overrides:
        tokens_for_model = token_overrides[sentence_full]
        tokens_display = tokens_for_model
        hit_terms = []
    elif sentence in token_overrides:
        tokens_for_model = token_overrides[sentence]
        tokens_display = tokens_for_model
        hit_terms = []
    else:
        try:
            okt = Okt()
            tokens_for_model = okt.morphs(sentence)
        except Exception as e:
            print(f"Okt 分词失败: {e}，使用空格分词...")
            tokens_for_model = sentence.split()

        tokens_for_model = merge_single_hangul_prefix_tokens(tokens_for_model, sentence)
        tokens_display, hit_terms = merge_tokens_with_glossary(tokens_for_model, glossary)

    print(f"分词结果: {tokens_display}")
    ko_tokens = [t for t in tokens_display if _is_korean_token(t)]
    if ko_tokens and all((t in glossary and t not in model_only_terms) for t in ko_tokens):
        translated_text = "".join([glossary.get(t, t) for t in tokens_display])
        translated_text = apply_replacements(translated_text, replace_rules)
        translated_text = restore_raw_terms_in_output(translated_text, glossary)
        translated_text = dedupe_repeated_ascii_runs(translated_text)
        translated_text = dedupe_repeated_cjk_phrases(translated_text)
        translated_text = ensure_preserve_english_terms(translated_text, original_raw_sentence)
        translated_text = ensure_preserve_literals(translated_text, original_raw_sentence)
        translated_text = force_glossary_zh_in_output(translated_text, hit_terms, glossary)
        translated_text = ensure_append_common_terms(translated_text, original_raw_sentence)
        translated_text, _ = apply_decision_rewrite_rules(translated_text, original_raw_sentence, tokens_display, glossary)
        return mark_untranslated_in_output(translated_text)
    
    # 转索引
    unk_idx = ko_vocab.get('<unk>', 3)
    model_tokens = filter_model_tokens(tokens_for_model)
    if hit_terms:
        block_terms = [t for t in hit_terms if not (model_only_terms and t in model_only_terms)]
        if block_terms:
            model_tokens = [t for t in model_tokens if t not in block_terms]
    if not model_tokens:
        translated_text = ""
        translated_text = apply_replacements(translated_text, replace_rules)
        translated_text = restore_raw_terms_in_output(translated_text, glossary)
        translated_text = dedupe_repeated_ascii_runs(translated_text)
        translated_text = dedupe_repeated_cjk_phrases(translated_text)
        translated_text = ensure_preserve_english_terms(translated_text, original_raw_sentence)
        translated_text = ensure_preserve_literals(translated_text, original_raw_sentence)
        translated_text = ensure_leading_enumeration(translated_text, original_raw_sentence)
        translated_text = ensure_append_common_terms(translated_text, original_raw_sentence)
        translated_text, used_template = apply_decision_rewrite_rules(translated_text, original_raw_sentence, tokens_display, glossary)
        final_text = mark_untranslated_in_output(apply_glossary_merge(translated_text, hit_zh_terms))
        if not used_template and isinstance(final_text, str) and final_text and "[?]" not in final_text and re.search(r"[\uAC00-\uD7A3]", sentence):
            final_text = f"{final_text}[?]"
        return final_text

    unk_tokens_exist = any(t not in glossary and ko_vocab.get(t, unk_idx) == unk_idx for t in model_tokens)
    indices = [ko_vocab['<sos>']] + [ko_vocab.get(token, unk_idx) for token in model_tokens] + [ko_vocab['<eos>']]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(indices)])
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
    
    has_english = isinstance(original_raw_sentence, str) and re.search(r"[A-Za-z][A-Za-z0-9]*", original_raw_sentence)
    allowed_ids = None
    penalize_ids = None
    penalize_value = 0.0

    if has_english:
        can_constrain = False
        if isinstance(glossary, dict) and glossary:
            if hit_terms or force_hit_terms:
                can_constrain = True
            elif isinstance(tokens_display, list) and any((isinstance(t, str) and t in glossary) for t in tokens_display):
                can_constrain = True

        if can_constrain:
            allowed_ids = set()
            allowed_ids.add(zh_vocab.get("<eos>"))
            allowed_ids.add(zh_vocab.get("<sos>"))

            for w in ("是否", "有无", "能否", "与否"):
                _add_zh_token_or_chars(zh_vocab, allowed_ids, w)

            for _w in PREDICATE_SUFFIXES:
                _add_zh_token_or_chars(zh_vocab, allowed_ids, _w)

            for _t in (hit_terms + force_hit_terms):
                if isinstance(_t, str):
                    _zh = glossary.get(_t)
                    if isinstance(_zh, str) and _zh:
                        _add_zh_token_or_chars(zh_vocab, allowed_ids, _zh)

            allowed_ids = {i for i in allowed_ids if isinstance(i, int)}
            if len(allowed_ids) < 15:
                allowed_ids = None

        penalize_ids = set()
        for w in ("风扇", "电机", "马达", "泵", "阀", "气缸", "传感器", "开关", "连接器", "端子", "电缆"):
            if w in zh_vocab:
                penalize_ids.add(zh_vocab[w])
        penalize_value = 4.0

    trg_indices = beam_search_decode(
        model,
        encoder_outputs,
        hidden,
        zh_vocab,
        device,
        max_len=max_len,
        beam_size=5,
        length_penalty=0.7,
        allowed_token_ids=allowed_ids,
        penalize_token_ids=penalize_ids,
        penalize_value=penalize_value,
    )
    
    inv_zh_vocab = {v: k for k, v in zh_vocab.items()}
    translated_tokens = [inv_zh_vocab.get(idx, '<unk>') for idx in trg_indices]
    translated_text = "".join([t for t in translated_tokens if t not in ['<sos>', '<eos>', '<pad>', '<unk>']])
    translated_text = apply_replacements(translated_text, replace_rules)
    translated_text = restore_raw_terms_in_output(translated_text, glossary)
    translated_text = dedupe_repeated_ascii_runs(translated_text)
    translated_text = dedupe_repeated_cjk_phrases(translated_text)
    translated_text = dedupe_shifou_pattern(translated_text)
    translated_text = ensure_preserve_english_terms(translated_text, original_raw_sentence)
    translated_text = ensure_preserve_literals(translated_text, original_raw_sentence)
    translated_text = force_glossary_zh_in_output(translated_text, hit_terms + force_hit_terms, glossary)
    translated_text = ensure_leading_enumeration(translated_text, original_raw_sentence)
    translated_text = ensure_append_common_terms(translated_text, original_raw_sentence)
    translated_text, used_template = apply_decision_rewrite_rules(translated_text, original_raw_sentence, tokens_display, glossary)
    if isinstance(translated_text, str) and "配电盘" in translated_text:
        translated_text = translated_text.replace("电控盘", "")
    final_text = mark_untranslated_in_output(apply_glossary_merge(translated_text, hit_zh_terms))
    if (not used_template) and should_append_unknown_marker(tokens_display, glossary, unk_tokens_exist, final_text) and isinstance(final_text, str) and "[?]" not in final_text:
        final_text = f"{final_text}[?]"
    return final_text

# --- 4. 主程序 ---
if __name__ == "__main__":
    device = torch.device('cpu')
    model_dir = 'Translate Model'
    user_dict_path = os.path.join(os.path.dirname(__file__), 'user_dict.md')
    user_dict = load_user_dict(user_dict_path)
    
    model_path = os.path.join(model_dir, 'best_model_v3_attn.pth') 
    ko_vocab_path = os.path.join(model_dir, 'best_ko_vocab_v3_attn.pkl')
    zh_vocab_path = os.path.join(model_dir, 'best_zh_vocab_v3_attn.pkl')

    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}，请先运行 V3.0 训练脚本。")
    elif not os.path.exists(ko_vocab_path) or not os.path.exists(zh_vocab_path):
        print(f"找不到词汇表文件: {ko_vocab_path} / {zh_vocab_path}，请先运行 V3.0 训练脚本。")
    else:
        with open(ko_vocab_path, 'rb') as f: ko_vocab = pickle.load(f)
        with open(zh_vocab_path, 'rb') as f: zh_vocab = pickle.load(f)
        
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
        print("V3.0 Attention 模型加载成功！")
        
        while True:
            sentence = input("\n请输入韩文 (输入 q 退出): ")
            if sentence.lower() == 'q': break
            if not sentence.strip(): continue
            
            token_overrides, direct_translations, replace_rules, glossary, model_only_terms = load_user_dict(user_dict_path)
            user_dict = (token_overrides, direct_translations, replace_rules, glossary, model_only_terms)
            translation = translate_sentence(sentence, model, ko_vocab, zh_vocab, device, user_dict)
            print(f"中文翻译: {translation}")
