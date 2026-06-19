"""
从网页收集韩文句子，保存为Excel

用法：直接运行 python collect_korean_sentences.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, urlunparse

import openpyxl
import requests
from bs4 import BeautifulSoup

# Konlpy 导入（如果可用）
try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False
    print("⚠️  Konlpy 不可用，将使用基础切分方法")
    print("   安装命令：pip install konlpy JPype1-py3")


# ---------------------------------------------------------------------------
# Konlpy 分词器管理
# ---------------------------------------------------------------------------

class KoreanTokenizer:
    """韩文分词器封装"""
    
    _instance = None
    _okt = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_tokenizer()
        return cls._instance
    
    def _init_tokenizer(self):
        if KONLPY_AVAILABLE:
            try:
                print("🔧 初始化 Konlpy Okt 分词器...")
                self._okt = Okt()
                print("✅ Okt 分词器初始化成功！")
            except Exception as e:
                print(f"⚠️  Okt 初始化失败: {e}")
                self._okt = None
        else:
            self._okt = None
    
    def is_available(self) -> bool:
        return self._okt is not None
    
    def pos(self, text: str, stem: bool = False) -> list[tuple[str, str]]:
        """词性标注"""
        if not self._okt:
            return []
        try:
            return self._okt.pos(text, stem=stem)
        except:
            return []
    
    def nouns(self, text: str) -> list[str]:
        """提取名词"""
        if not self._okt:
            return []
        try:
            return self._okt.nouns(text)
        except:
            return []
    
    def morphs(self, text: str) -> list[str]:
        """分词"""
        if not self._okt:
            return []
        try:
            return self._okt.morphs(text)
        except:
            return []
    
    def analyze_quality(self, text: str) -> dict:
        """分析句子质量"""
        result = {
            'has_noun': False,
            'has_verb': False,
            'has_adjective': False,
            'noun_count': 0,
            'content_word_count': 0,
            'is_good_quality': False
        }
        
        if not self._okt:
            return result
        
        try:
            pos_tags = self._okt.pos(text)
            
            for word, tag in pos_tags:
                if tag in ['Noun', 'NNP', 'NNG']:
                    result['noun_count'] += 1
                    result['has_noun'] = True
                    result['content_word_count'] += 1
                elif tag in ['Verb', 'VV']:
                    result['has_verb'] = True
                    result['content_word_count'] += 1
                elif tag in ['Adjective', 'VA']:
                    result['has_adjective'] = True
                    result['content_word_count'] += 1
            
            # 质量判断：至少有1个名词和1个动词/形容词
            result['is_good_quality'] = (
                result['noun_count'] >= 1 and 
                (result['has_verb'] or result['has_adjective'])
            )
            
        except:
            pass
        
        return result

    def score_sentence(self, text: str) -> dict:
        """
        智能打分系统（满分100分）
        返回: {'score': int, 'details': str, 'score_length': int, 'score_hangul': int, 'score_noun': int, 'score_verb_adj': int, 'score_content_word': int, 'score_completeness': int}
        """
        score = 0
        details = []
        score_length = 0
        score_hangul = 0
        score_noun = 0
        score_verb_adj = 0
        score_content_word = 0
        score_completeness = 0
        
        # 1. 长度得分 (15分)
        text_len = len(text)
        if 50 <= text_len <= 80:
            score += 15
            score_length = 15
            details.append("长度: 15分")
        elif 30 <= text_len < 50 or 80 < text_len <= 100:
            score += 12
            score_length = 12
            details.append("长度: 12分")
        elif 15 <= text_len < 30 or 100 < text_len <= 128:
            score += 8
            score_length = 8
            details.append("长度: 8分")
        else:
            details.append("长度: 0分")
        
        # 2. 韩文比例 (20分)
        hangul_count = len(_HANGUL_RE.findall(text))
        hangul_ratio = hangul_count / text_len if text_len > 0 else 0
        if hangul_ratio >= 0.7:
            score += 20
            score_hangul = 20
            details.append("韩文比例: 20分")
        elif hangul_ratio >= 0.5:
            score += 15
            score_hangul = 15
            details.append("韩文比例: 15分")
        elif hangul_ratio >= 0.3:
            score += 10
            score_hangul = 10
            details.append("韩文比例: 10分")
        else:
            details.append("韩文比例: 0分")
        
        # 如果分词器不可用，就返回基础分
        if not self._okt:
            details.append("(分词器不可用，质量相关项默认处理)")
            return {
                'score': score,
                'details': ' | '.join(details),
                'score_length': score_length,
                'score_hangul': score_hangul,
                'score_noun': score_noun,
                'score_verb_adj': score_verb_adj,
                'score_content_word': score_content_word,
                'score_completeness': score_completeness
            }
        
        try:
            quality = self.analyze_quality(text)
            
            # 3. 名词数量 (20分)
            if quality['noun_count'] >= 3:
                score += 20
                score_noun = 20
                details.append(f"名词({quality['noun_count']}): 20分")
            elif quality['noun_count'] == 2:
                score += 15
                score_noun = 15
                details.append(f"名词({quality['noun_count']}): 15分")
            elif quality['noun_count'] == 1:
                score += 10
                score_noun = 10
                details.append(f"名词({quality['noun_count']}): 10分")
            else:
                details.append(f"名词({quality['noun_count']}): 0分")
            
            # 4. 动词/形容词 (20分)
            if quality['has_verb'] and quality['has_adjective']:
                score += 20
                score_verb_adj = 20
                details.append("动形齐全: 20分")
            elif quality['has_verb'] or quality['has_adjective']:
                score += 15
                score_verb_adj = 15
                details.append("有动/形: 15分")
            else:
                details.append("动形缺失: 0分")
            
            # 5. 内容词多样性 (15分)
            if quality['content_word_count'] >= 5:
                score += 15
                score_content_word = 15
                details.append(f"内容词({quality['content_word_count']}): 15分")
            elif quality['content_word_count'] >= 3:
                score += 10
                score_content_word = 10
                details.append(f"内容词({quality['content_word_count']}): 10分")
            elif quality['content_word_count'] >= 2:
                score += 5
                score_content_word = 5
                details.append(f"内容词({quality['content_word_count']}): 5分")
            else:
                details.append(f"内容词({quality['content_word_count']}): 0分")
            
            # 6. 完整性检查 (10分)
            if quality['is_good_quality']:
                score += 10
                score_completeness = 10
                details.append("完整性: 10分")
            else:
                details.append("完整性: 0分")
                
        except Exception as e:
            details.append(f"(分词异常: {str(e)[:20]})")
        
        # 确保分数不超过100
        score = min(score, 100)
        
        return {
            'score': score,
            'details': ' | '.join(details),
            'score_length': score_length,
            'score_hangul': score_hangul,
            'score_noun': score_noun,
            'score_verb_adj': score_verb_adj,
            'score_content_word': score_content_word,
            'score_completeness': score_completeness
        }


# 全局分词器实例
_tokenizer = None

def get_tokenizer() -> KoreanTokenizer:
    """获取全局分词器实例"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = KoreanTokenizer()
    return _tokenizer

# ---------------------------------------------------------------------------
# 文本规范化
# ---------------------------------------------------------------------------

_CIRCLED = {
    "⓪": 0, "①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5, "⑥": 6, "⑦": 7,
    "⑧": 8, "⑨": 9, "⑩": 10, "⑪": 11, "⑫": 12, "⑬": 13, "⑭": 14,
    "⑮": 15, "⑯": 16, "⑰": 17, "⑱": 18, "⑲": 19, "⑳": 20,
}


def _normalize_text(sentence: str) -> str:
    if not isinstance(sentence, str):
        return ""
    s = sentence.replace("\r\n", "\n").replace("\r", "\n")
    for ch, n in _CIRCLED.items():
        s = s.replace(ch, f"({n})")
    s = s.replace("…", "...")
    s = re.sub(r"-{1,4}>", "->", s)
    for arrow in ("→", "⇒", "➡", "⟶", "⟹", "➔", "➜", "➝", "➞", "➟", "➠"):
        s = s.replace(arrow, "->")
    for q in ("“", "”", "„", "‟", "＂"):
        s = s.replace(q, '"')
    return unicodedata.normalize("NFKC", s)


def clean_ko_sentence(sentence: str) -> str:
    """保留韩文、英文、数字、常见标点；去掉换行；清理开头结尾问题字符。"""
    if not isinstance(sentence, str):
        return ""
    s = _normalize_text(sentence).replace("\n", " ")
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
        cat = unicodedata.category(ch)
        if cat and cat[0] == "P":
            kept.append(ch)
            continue
        if cat in ("Sm", "Sc"):
            kept.append(ch)
            continue
    s = "".join(kept)
    s = re.sub(r"\s+", " ", s).strip()
    
    # 清理开头的问题字符
    while len(s) > 0:
        if s.startswith(">") or s.startswith("<"):
            s = s[1:].lstrip()
        elif s[0] in ("/", "\\", "-", "=", "_", "+", "*", "#", "@", "!", "?", ",", ".", "·", "•", "○", "~", "|", ";", ":"):
            s = s[1:].lstrip()
        else:
            break
    
    # 清理结尾的问题字符
    while len(s) > 0:
        if s[-1] in ("/", "\\", "-", "=", "_", "+", "*", "#", "@", "!", "?", ",", "·", "•", "○", "~", "|", ";", ":"):
            s = s[:-1].rstrip()
        else:
            break
    
    # 清理多余的标点
    s = re.sub(r"[.]{3,}", "…", s)  # 多个点转为省略号
    s = re.sub(r"[!]{2,}", "!", s)
    s = re.sub(r"[?]{2,}", "?", s)
    s = re.sub(r"[,]{2,}", ",", s)
    
    return s.strip()


# ---------------------------------------------------------------------------
# 句子切分与过滤
# ---------------------------------------------------------------------------

_HANGUL_RE = re.compile(r"[\uAC00-\uD7A3]")


@dataclass
class SentenceItem:
    ko: str
    source: str
    char_count: int = 0
    score: int = 0
    score_details: str = ""
    score_length: int = 0
    score_hangul: int = 0
    score_noun: int = 0
    score_verb_adj: int = 0
    score_content_word: int = 0
    score_completeness: int = 0
    zh_draft: str = ""
    zh_fix: str = ""
    confirmed: str = ""
    status: str = "已采集"


def ends_with_period(ko: str) -> bool:
    """句子必须以句号（或 ? ! …）结尾。"""
    if not ko:
        return False
    # 韩文常见句尾
    if re.search(r"(?:다|니다|습니다|ㅂ니다|요|함|임|됨|니까|네|예)\.$", ko):
        return True
    # 标准句末标点（排除数字后的小数点）
    return bool(re.search(r"(?<!\d)[.!?…]$", ko))


def is_brackets_balanced(text: str) -> bool:
    """检查括号是否平衡"""
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for c in text:
        if c in pairs.values():
            stack.append(c)
        elif c in pairs.keys():
            if not stack or stack.pop() != pairs[c]:
                return False
    return len(stack) == 0


def is_quotes_balanced(text: str) -> bool:
    """检查引号是否平衡（简单检查）"""
    # 双引号
    if text.count('"') % 2 != 0:
        return False
    # 单引号
    if text.count("'") % 2 != 0:
        return False
    return True


def is_valid_ko_sentence(ko: str, *, min_hangul: int = 4, min_chars: int = 15, max_chars: int = 128, use_konlpy_check: bool = True) -> tuple[bool, dict]:
    """验证韩文句子有效性
    
    返回: (是否有效, 详细信息字典)
    """
    info = {
        'valid': False,
        'reason': '',
        'konlpy_available': KONLPY_AVAILABLE,
        'quality': None
    }
    
    if not ko:
        info['reason'] = '空句子'
        return False, info
    
    # 长度检查
    if len(ko) < min_chars:
        info['reason'] = f'太短 ({len(ko)} < {min_chars})'
        return False, info
    if len(ko) > max_chars:
        info['reason'] = f'太长 ({len(ko)} > {max_chars})'
        return False, info
    
    # 必须以正确的结尾结束
    if not ends_with_period(ko):
        info['reason'] = '句末标点不正确'
        return False, info
    
    # 括号平衡
    if not is_brackets_balanced(ko):
        info['reason'] = '括号不匹配'
        return False, info
    
    # 引号平衡
    if not is_quotes_balanced(ko):
        info['reason'] = '引号不匹配'
        return False, info
    
    # 韩文数量检查
    hangul_count = len(_HANGUL_RE.findall(ko))
    if hangul_count < min_hangul:
        info['reason'] = f'韩文太少 ({hangul_count} < {min_hangul})'
        return False, info
    
    # 使用 Konlpy 进行质量检查（如果可用）
    if use_konlpy_check and KONLPY_AVAILABLE:
        tokenizer = get_tokenizer()
        if tokenizer.is_available():
            quality = tokenizer.analyze_quality(ko)
            info['quality'] = quality
            
            if not quality['is_good_quality']:
                # 如果 Konlpy 分析认为质量不好，我们仍然可以通过基础检查
                # 但标记一下
                info['reason'] = f'质量检查未通过 (名词:{quality["noun_count"]}, 动词:{quality["has_verb"]})'
                # 我们不严格拒绝，只是记录
    
    info['valid'] = True
    info['reason'] = '通过'
    return True, info


def split_sentences(text: str) -> list[str]:
    """更智能的韩文句子切分
    
    改进点：
    1. 优先按标准句末标点切分
    2. 考虑括号、引号内的内容不随意切分
    3. 合并过短的片段
    """
    # 第一步：预处理文本
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace('\n', ' ').replace('\r', ' ')
    if not text:
        return []
    
    sentences = []
    start = 0
    n = len(text)
    
    # 括号和引号状态跟踪
    in_brackets = 0  # 括号深度
    in_quotes = False
    
    i = 0
    while i < n:
        c = text[i]
        
        # 跟踪括号状态
        if c in '([{':
            in_brackets += 1
        elif c in ')]}':
            in_brackets = max(0, in_brackets - 1)
        # 跟踪引号状态
        elif c in '"\'':
            in_quotes = not in_quotes
        
        # 只有不在括号和引号内时才考虑切分
        if in_brackets == 0 and not in_quotes:
            # 检查是否是句末标点
            if c in '.!?…' and i > 0:
                # 检查是否是数字后的小数点
                if c == '.' and i > 0 and text[i-1].isdigit():
                    i += 1
                    continue
                
                # 检查韩文特殊结尾
                is_korean_end = False
                for end in ['다', '니다', '습니다', 'ㅂ니다', '요', '함', '임', '됨', '니까', '네', '예']:
                    end_len = len(end)
                    if i >= end_len and text[i-end_len:i] == end:
                        is_korean_end = True
                        break
                
                # 确认切分
                if is_korean_end or c in '!?…' or (c == '.' and not is_korean_end):
                    # 提取句子
                    sentence = text[start:i+1].strip()
                    if sentence:
                        sentences.append(sentence)
                    start = i + 1
        
        i += 1
    
    # 处理最后一个可能的句子
    if start < n:
        last_sentence = text[start:].strip()
        if last_sentence and ends_with_period(last_sentence):
            sentences.append(last_sentence)
    
    # 过滤和清理
    result = []
    for s in sentences:
        s = s.strip()
        if s and ends_with_period(s):
            result.append(s)
    
    return result


def _sentence_key(ko: str) -> str:
    return hashlib.md5(ko.encode("utf-8")).hexdigest()


def normalize_article_url(url: str) -> str:
    """统一文章 URL / 本地路径，用于去重登记。"""
    raw = url.strip()
    if not raw:
        return raw
    if raw.startswith("http://") or raw.startswith("https://"):
        p = urlparse(raw)
        scheme = (p.scheme or "https").lower()
        netloc = p.netloc.lower()
        path = p.path or "/"
        if path != "/":
            path = path.rstrip("/")
        return urlunparse((scheme, netloc, path, "", p.query, ""))
    return str(Path(raw).resolve())


class CorpusHistory:
    """记录已爬文章与已收集句子，保证多次采集不重复。"""

    VERSION = 3  # 版本更新
    DEFAULT_PATH = Path("corpus_trusted/crawl_history.json")

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else self.DEFAULT_PATH
        self.crawled_urls: set[str] = set()
        self.crawled_titles: set[str] = set()
        self.sentence_hashes: set[str] = set()
        self.url_page_progress: dict[str, int] = {}  # 记录每个URL的翻页进度
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"警告: 无法读取历史文件 {self.path} ({e})，将从头开始。")
            return
        if data.get("version") != self.VERSION:
            print(f"警告: 历史文件版本不匹配，将从头开始。")
            return
        self.crawled_urls = set(data.get("crawled_urls", []))
        self.crawled_titles = set(data.get("crawled_titles", []))
        self.sentence_hashes = set(data.get("sentence_hashes", []))
        self.url_page_progress = data.get("url_page_progress", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "crawled_urls": sorted(self.crawled_urls),
            "crawled_titles": sorted(self.crawled_titles),
            "sentence_hashes": sorted(self.sentence_hashes),
            "url_page_progress": self.url_page_progress,
            "counts": {
                "articles": len(self.crawled_urls),
                "sentences": len(self.sentence_hashes),
            },
        }
        print(f"[调试] 正在保存历史记录: {len(self.crawled_urls)} 篇文章, {len(self.sentence_hashes)} 个句子")
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[调试] 历史记录已保存到: {self.path}")

    def clear(self) -> None:
        """清零历史记录并立即写回空文件。"""
        self.crawled_urls.clear()
        self.crawled_titles.clear()
        self.sentence_hashes.clear()
        self.url_page_progress.clear()
        self.save()

    def has_article(self, url: str, title: str = "") -> tuple[bool, str]:
        """检查URL或标题是否已存在于历史记录中。
        返回 (是否重复, 原因)
        """
        norm_url = normalize_article_url(url)
        if norm_url in self.crawled_urls:
            return True, "URL已存在"
        
        # 只有在成功提取到有效标题时才检查标题重复
        # 避免因为通用标题导致误判
        if title and len(title) >= 10:
            if title in self.crawled_titles:
                return True, "标题已存在"
        
        return False, ""

    def has_sentence(self, ko: str) -> bool:
        return _sentence_key(ko) in self.sentence_hashes

    def mark_article(self, url: str, title: str = "") -> None:
        """记录文章URL和标题。"""
        self.crawled_urls.add(normalize_article_url(url))
        # 只记录有效标题
        if title and len(title) >= 10:
            self.crawled_titles.add(title)

    def mark_sentence(self, ko: str) -> None:
        self.sentence_hashes.add(_sentence_key(ko))

    def register_sentences(self, items: Iterable[SentenceItem]) -> None:
        """登记本轮写入 xlsx 的句子（文章 URL 在爬取成功时已登记）。"""
        for it in items:
            self.mark_sentence(it.ko)

    def get_url_page(self, url: str) -> int:
        """获取URL的上次翻页进度。"""
        norm_url = normalize_article_url(url)
        return self.url_page_progress.get(norm_url, 1)

    def set_url_page(self, url: str, page: int) -> None:
        """设置URL的翻页进度。"""
        norm_url = normalize_article_url(url)
        self.url_page_progress[norm_url] = page

    def summary(self) -> str:
        return (
            f"历史记录 ({self.path}): "
            f"已爬文章 {len(self.crawled_urls)} 篇, "
            f"已收集句子 {len(self.sentence_hashes)} 条"
        )


# ---------------------------------------------------------------------------
# 网页采集
# ---------------------------------------------------------------------------

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
}

ARTICLE_SELECTORS = [
    "#dic_area",  # Naver news
    "#newsct_article",
    "#article-view-content-div",  # epnc 等
    ".newsct_article",
    ".article-body",
    ".article_view",
    "#articleBody",
    "article",
    ".view_con",
    ".entry-content",
]

LIST_LINK_HINTS = ("articleView", "article/view", "/news/", "/article/")


def _list_page_url(base_url: str, page: int) -> str:
    if page <= 1:
        return base_url
    if re.search(r"([?&])page=\d+", base_url):
        return re.sub(r"([?&])page=\d+", rf"\1page={page}", base_url)
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}page={page}"


class WebCollector:
    def __init__(
        self,
        delay_sec: float = 1.5,
        *,
        history: CorpusHistory | None = None,
        list_max_pages: int = 15,
    ):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.delay_sec = delay_sec
        self.history = history
        self.list_max_pages = list_max_pages
        self.stats = {
            "articles_skipped": 0,
            "articles_fetched": 0,
            "sentences_skipped_dup": 0,
            "sentences_skipped_long": 0,
        }

    def fetch_html(self, url: str) -> str:
        time.sleep(self.delay_sec)
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text

    def extract_article_text(self, html: str, url: str) -> tuple[str, str]:
        """提取文章正文和标题，返回 (title, text) 元组。"""
        # 提取标题时用一个独立的soup对象，避免后面修改影响
        title_soup = BeautifulSoup(html, "lxml")
        title = ""
        # 尝试多个常见的标题选择器（按优先级）
        title_selectors = [
            "h1",  # 优先 h1
            ".article-title",
            ".news-title",
            "#articleTitle",
            ".entry-title",
            ".view_tit",
            ".tit_article",
            "[class*='article-tit']",
            "[class*='news-tit']",
        ]
        for sel in title_selectors:
            try:
                node = title_soup.select_one(sel)
                if node:
                    text = node.get_text(strip=True)
                    # 更严格的标题检查：至少10个字符，不能太短太通用
                    if len(text) >= 10 and len(text) < 150:
                        title = text
                        break
            except:
                continue
        
        # 如果没找到，尝试title标签但要清理
        if not title:
            try:
                title_node = title_soup.select_one("title")
                if title_node:
                    text = title_node.get_text(strip=True)
                    # 清理常见的网站后缀
                    text = text.split("|")[0].strip()
                    text = text.split("-")[0].strip()
                    text = text.split(":")[0].strip()
                    if len(text) >= 10 and len(text) < 150:
                        title = text
            except:
                pass
        
        # 提取正文用另一个soup对象
        soup = BeautifulSoup(html, "lxml")
        # 移除不需要的标签
        for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside"]):
            tag.decompose()
        # 移除标题相关标签（正文提取时）
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            tag.decompose()
        # 移除元数据和导航
        for tag in soup.find_all(["meta", "link"]):
            tag.decompose()
        
        body_text = ""
        # 优先使用文章选择器
        for sel in ARTICLE_SELECTORS:
            node = soup.select_one(sel)
            if node:
                # 在这个节点内部，进一步移除可能的标题
                for sub_tag in node.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                    sub_tag.decompose()
                text = node.get_text(" ", strip=True)
                if len(text) >= 80:
                    body_text = text
                    break
        # fallback: 收集所有段落
        if not body_text:
            paragraphs = []
            for p in soup.find_all("p"):
                text = p.get_text(" ", strip=True)
                if len(text) > 30:
                    paragraphs.append(text)
            if paragraphs:
                body_text = " ".join(paragraphs)
        
        return title, body_text

    def discover_article_urls(
        self,
        list_url: str,
        limit: int = 30,
        *,
        exclude: set[str] | None = None,
        start_page: int = 1,
    ) -> tuple[list[str], int]:
        """从列表页发现未爬过的文章链接（支持翻页和保存进度）。
        
        返回 (found_urls, last_page)
        """
        import random
        exclude = exclude or set()
        base = f"{urlparse(list_url).scheme}://{urlparse(list_url).netloc}"
        found: list[str] = []
        seen_on_site: set[str] = set()
        last_page = start_page

        for page in range(start_page, start_page + self.list_max_pages):
            if page > 10:  # 最多总共10页
                break
            last_page = page
            page_url = _list_page_url(list_url, page)
            try:
                html = self.fetch_html(page_url)
            except Exception as e:
                if page == 1:
                    raise
                print(f"    第 {page} 页获取失败，停止翻页")
                break
            soup = BeautifulSoup(html, "lxml")
            
            # 收集当前页所有可能的文章链接
            page_candidates: list[str] = []
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not any(x in href for x in LIST_LINK_HINTS):
                    continue
                full = normalize_article_url(urljoin(base, href))
                # 排除列表页自己！只保留文章页
                if "articlelist" in full.lower() or "list.html" in full.lower() or "list.php" in full.lower() or "section" in full.lower():
                    continue
                if full in seen_on_site:
                    continue
                seen_on_site.add(full)
                if full in exclude:
                    continue
                page_candidates.append(full)
            
            # 随机打乱顺序
            random.shuffle(page_candidates)
            
            # 添加到结果
            page_new = 0
            for url in page_candidates:
                found.append(url)
                page_new += 1
                if len(found) >= limit:
                    print(f"    翻到第 {page} 页，已收集到 {len(found)} 篇目标文章，停止")
                    return found, page
            
            if page_new == 0:
                print(f"    第 {page} 页未发现新文章，停止翻页")
                break
            print(f"    第 {page} 页，新增 {page_new} 篇，累计 {len(found)} 篇")
        
        print(f"    翻到第 {last_page} 页，共发现 {len(found)} 篇新文章")
        return found, last_page

    def sentences_from_url(
        self,
        url: str,
        *,
        discover: bool = True,
        articles_per_list: int = 25,
    ) -> list[SentenceItem]:
        items: list[SentenceItem] = []
        urls = [url]
        hist = self.history

        if discover and self._looks_like_list_page(url):
            try:
                exclude = hist.crawled_urls if hist else set()
                start_page = hist.get_url_page(url) if hist else 1
                urls, last_page = self.discover_article_urls(
                    url,
                    limit=articles_per_list,
                    exclude=exclude,
                    start_page=start_page,
                )
                if hist:
                    hist.set_url_page(url, last_page + 1)  # 下次从下一页开始
                print(f"  列表页发现 {len(urls)} 篇新文章: {url}")
            except Exception as e:
                print(f"  列表页解析失败，改抓当前页: {e}")
                urls = [url]
        else:
            # 对于单个文章URL，先粗略检查（不检查标题，因为还没提取）
            norm_url = normalize_article_url(url)
            if hist and norm_url in hist.crawled_urls:
                print(f"  跳过已爬文章: {url}")
                self.stats["articles_skipped"] += 1
                return items
            urls = [url]

        # 本轮内记录已处理的URL，避免同一轮内重复
        round_processed_urls: set[str] = set()
        # 计数器：每成功处理5篇文章保存一次历史
        articles_since_save = 0
        
        for i, article_url in enumerate(urls, 1):
            try:
                norm_url = normalize_article_url(article_url)
                print(f"  [{i}/{len(urls)}] 处理文章: {article_url[:60]}...")
                
                # 调试：检查本轮是否已处理过
                if norm_url in round_processed_urls:
                    print(f"    [调试] 本轮已处理过此URL，跳过")
                    continue
                round_processed_urls.add(norm_url)
                
                # 再次检查：如果这看起来是列表页，直接跳过
                if self._looks_like_list_page(article_url):
                    print(f"    [调试] 这看起来是列表页，跳过")
                    self.stats["articles_skipped"] += 1
                    continue
                
                html = self.fetch_html(article_url)
                title, text = self.extract_article_text(html, article_url)
                
                # 检查URL或标题是否已存在
                if hist:
                    is_dup, reason = hist.has_article(article_url, title)
                    if is_dup:
                        msg = f"    跳过 (原因: {reason})"
                        if reason == "标题已存在" and title:
                            msg += f" | 标题: {title[:60]}"
                        print(msg)
                        self.stats["articles_skipped"] += 1
                        continue
                
                if hist:
                    hist.mark_article(article_url, title)
                self.stats["articles_fetched"] += 1
                articles_since_save += 1
                
                sentences_before = len(items)
                long_before = self.stats["sentences_skipped_long"]
                short_before = self.stats.get("sentences_skipped_short", 0)
                invalid_before = self.stats.get("sentences_skipped_invalid", 0)
                
                # 调试：打印提取到的标题
                if title:
                    print(f"    [调试] 提取到标题: {title[:60]}")
                
                for raw in split_sentences(text):
                    ko = clean_ko_sentence(raw)
                    is_valid, valid_info = is_valid_ko_sentence(
                        ko, 
                        min_chars=MIN_SENTENCE_CHARS, 
                        max_chars=MAX_SENTENCE_CHARS
                    )
                    if not is_valid:
                        if len(ko) > MAX_SENTENCE_CHARS:
                            self.stats["sentences_skipped_long"] += 1
                        elif len(ko) < MIN_SENTENCE_CHARS:
                            self.stats["sentences_skipped_short"] = self.stats.get("sentences_skipped_short", 0) + 1
                        else:
                            self.stats["sentences_skipped_invalid"] = self.stats.get("sentences_skipped_invalid", 0) + 1
                        continue
                    if hist and hist.has_sentence(ko):
                        self.stats["sentences_skipped_dup"] += 1
                        continue
                    
                    # 打分
                    tokenizer = get_tokenizer()
                    score_result = tokenizer.score_sentence(ko)
                    
                    items.append(SentenceItem(
                        ko=ko,
                        source=article_url,
                        char_count=len(ko),
                        score=score_result['score'],
                        score_details=score_result['details'],
                        score_length=score_result['score_length'],
                        score_hangul=score_result['score_hangul'],
                        score_noun=score_result['score_noun'],
                        score_verb_adj=score_result['score_verb_adj'],
                        score_content_word=score_result['score_content_word'],
                        score_completeness=score_result['score_completeness']
                    ))
                    if hist:
                        hist.mark_sentence(ko)
                sentences_after = len(items)
                long_in_article = self.stats["sentences_skipped_long"] - long_before
                short_in_article = self.stats.get("sentences_skipped_short", 0) - short_before
                invalid_in_article = self.stats.get("sentences_skipped_invalid", 0) - invalid_before
                
                if sentences_after > sentences_before:
                    msg = f"    ✓ 从该文章提取 {sentences_after - sentences_before} 句"
                    if title:
                        msg += f" | 标题: {title[:40]}"
                    filtered = []
                    if long_in_article > 0:
                        filtered.append(f"长句{long_in_article}")
                    if short_in_article > 0:
                        filtered.append(f"短句{short_in_article}")
                    if invalid_in_article > 0:
                        filtered.append(f"无效{invalid_in_article}")
                    if filtered:
                        msg += f" (过滤 {', '.join(filtered)})"
                    print(msg)
                
                # 每成功处理5篇文章就保存一次历史
                if articles_since_save >= 5 and hist:
                    print("    [调试] 每5篇文章保存一次历史...")
                    hist.save()
                    articles_since_save = 0
                    
            except Exception as e:
                print(f"    ✗ 跳过: {str(e)[:80]}")
        
        # 处理完这个URL后也保存一下
        if hist and (self.stats["articles_fetched"] > 0 or self.stats["articles_skipped"] > 0):
            print("  [调试] 处理完此URL，保存历史...")
            hist.save()
            
        return items

    @staticmethod
    def _looks_like_list_page(url: str) -> bool:
        low = url.lower()
        return any(x in low for x in ("articlelist", "articleList", "list.html", "list.php", "section"))


def dedupe_items(items: list[SentenceItem]) -> list[SentenceItem]:
    seen: set[str] = set()
    out: list[SentenceItem] = []
    for it in items:
        key = _sentence_key(it.ko)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# ---------------------------------------------------------------------------
debug_chinese.py# Excel 读写
# ---------------------------------------------------------------------------

HEADERS = ("NO", "KO", "ZH", "ZH修正", "字符数","状态", "分数", "长度分", "韩文比例分", "名词分", "动形分", "内容词分", "完整性分")


def resolve_output_path(path: Path | str, *, add_timestamp: bool = True) -> Path:
    """保存前解析输出路径；默认在文件名中插入时间戳。

    例: ko_corpus_10000.xlsx -> ko_corpus_10000_20260527_143052.xlsx
    """
    p = Path(path)
    if not add_timestamp:
        return p
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return p.with_name(f"{p.stem}_{ts}{p.suffix}")


def clear_crawl_history_file(path: Path | None = None) -> Path:
    """清零 crawl_history.json 文件。"""
    history_path = Path(path) if path else CorpusHistory.DEFAULT_PATH
    history = CorpusHistory(history_path)
    history.clear()
    print(f"已清零历史记录文件: {history_path}")
    return history_path


def write_review_xlsx(path: Path, items: list[SentenceItem]) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Corpus"
    for col, h in enumerate(HEADERS, start=1):
        ws.cell(1, col).value = h
    for row, it in enumerate(items, start=2):
        ws.cell(row, 1).value = it.confirmed
        ws.cell(row, 2).value = it.ko
        ws.cell(row, 3).value = it.zh_draft
        ws.cell(row, 4).value = it.zh_fix or it.zh_draft
        ws.cell(row, 5).value = it.char_count
        ws.cell(row, 6).value = it.status
        ws.cell(row, 7).value = it.score
        ws.cell(row, 8).value = it.score_length
        ws.cell(row, 9).value = it.score_hangul
        ws.cell(row, 10).value = it.score_noun
        ws.cell(row, 11).value = it.score_verb_adj
        ws.cell(row, 12).value = it.score_content_word
        ws.cell(row, 13).value = it.score_completeness
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    wb.close()


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
URLs_FILE = Path("corpus_sources/urls_example.txt")
MAX_SENTENCES = 10000
MAX_ROUNDS = 3  # 最多尝试3轮
OUTPUT_FILE = "corpus_trusted/ko_corpus_10000.xlsx"
DELAY_SEC = 1.5
ARTICLES_PER_LIST = 50  # 每个列表页最多收集文章数
LIST_MAX_PAGES = 10  # 每个URL最多翻页层数
MAX_SENTENCE_CHARS = 128  # 句子最大字符数
MIN_SENTENCE_CHARS = 15  # 句子最小字符数
RESET_CRAWL_HISTORY = True  # 设为 True 时，程序运行前先清零 crawl_history.json

# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("韩文句子收集工具")
    print("=" * 60)
    print(f"目标: {MAX_SENTENCES} 句，最多尝试 {MAX_ROUNDS} 轮")
    print(f"每个URL翻页: 最多 {LIST_MAX_PAGES} 页")
    print(f"每个列表页: 最多收集 {ARTICLES_PER_LIST} 篇新文章")
    print(f"句子长度限制: {MIN_SENTENCE_CHARS}-{MAX_SENTENCE_CHARS} 字符")
    print("智能切分: 括号/引号内不切分，避免句子被截断")
    print("1轮定义 = 完整处理 urls_example.txt 中所有URL一次")
    print("历史记录 = 保存文章URL和标题，URL或标题相同即跳过")
    print("=" * 60)
    
    # 记录开始时间
    import time
    start_time = time.time()

    if RESET_CRAWL_HISTORY:
        print("\n[启动前清理] 已启用 crawl_history.json 清零")
        clear_crawl_history_file()
    
    # 初始化 Konlpy 分词器（如果可用）
    if KONLPY_AVAILABLE:
        print("\n🔧 正在初始化韩文分词器...")
        tokenizer = get_tokenizer()
        if tokenizer.is_available():
            print("✅ 分词器初始化成功！将使用 Konlpy Okt 进行质量分析")
        else:
            print("⚠️  分词器初始化失败，将使用基础验证方法")

    # 读取URL列表
    if not URLs_FILE.exists():
        print(f"错误: URL文件不存在: {URLs_FILE}")
        return

    urls = [
        ln.strip()
        for ln in URLs_FILE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    if not urls:
        print("错误: URL文件为空或只有注释")
        return

    print(f"\n加载了 {len(urls)} 个URL")

    # 初始化历史记录
    history = CorpusHistory()
    print(f"\n初始 {history.summary()}")

    # 初始化采集器
    collector = WebCollector(
        delay_sec=DELAY_SEC,
        history=history,
        list_max_pages=LIST_MAX_PAGES,
    )

    all_items: list[SentenceItem] = []
    seen_keys: set[str] = set(history.sentence_hashes)
    total_articles_fetched = 0
    total_articles_skipped = 0
    total_sentences_skipped_dup = 0
    total_sentences_skipped_long = 0
    total_sentences_skipped_short = 0
    total_sentences_skipped_invalid = 0
    total_sentences_quality_checked = 0

    # 开始多轮采集
    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"第 {round_num}/{MAX_ROUNDS} 轮开始")
        print(f"{'='*60}")
        
        round_start_count = len(all_items)
        
        # 本轮采集所有URL
        for url in urls:
            # 先去重一次，确保我们知道真实的进度
            all_items = dedupe_items(all_items)
            if len(all_items) >= MAX_SENTENCES:
                break
            
            print(f"\n[轮次 {round_num}] 采集: {url}")
            batch = collector.sentences_from_url(
                url,
                discover=True,
                articles_per_list=ARTICLES_PER_LIST,
            )
            for it in batch:
                key = _sentence_key(it.ko)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_items.append(it)
                
                # 不断去重和检查
                all_items = dedupe_items(all_items)
                if len(all_items) >= MAX_SENTENCES:
                    break
        
        # 统计本轮结果
        round_new_count = len(all_items) - round_start_count
        print(f"\n第 {round_num} 轮结束，本轮新增 {round_new_count} 句，累计 {len(all_items)} 句")
        
        # 累计统计
        total_articles_fetched += collector.stats['articles_fetched']
        total_articles_skipped += collector.stats['articles_skipped']
        total_sentences_skipped_dup += collector.stats['sentences_skipped_dup']
        total_sentences_skipped_long += collector.stats['sentences_skipped_long']
        total_sentences_skipped_short += collector.stats.get('sentences_skipped_short', 0)
        total_sentences_skipped_invalid += collector.stats.get('sentences_skipped_invalid', 0)
        
        # 重置本轮统计
        collector.stats['articles_fetched'] = 0
        collector.stats['articles_skipped'] = 0
        collector.stats['sentences_skipped_dup'] = 0
        collector.stats['sentences_skipped_long'] = 0
        collector.stats['sentences_skipped_short'] = 0
        collector.stats['sentences_skipped_invalid'] = 0
        
        # 检查是否达到目标
        if len(all_items) >= MAX_SENTENCES:
            print(f"\n✅ 第 {round_num} 轮已达到目标 {MAX_SENTENCES} 句，提前结束！")
            break
        
        # 检查本轮是否有新增，没有新增则提前结束
        if round_new_count == 0:
            print(f"\n⚠️  第 {round_num} 轮未收集到新句子，提前结束！")
            break
        
        # 每轮结束后保存一次历史
        history.save()

    # 最终去重和截取
    all_items = dedupe_items(all_items)
    
    # 如果还没达到目标，继续尝试（无限循环直到达到目标）
    extra_round = MAX_ROUNDS + 1
    while len(all_items) < MAX_SENTENCES:
        print(f"\n{'='*60}")
        print(f"补充轮 {extra_round} - 继续收集直到达到 {MAX_SENTENCES} 句")
        print(f"{'='*60}")
        print(f"当前进度: {len(all_items)}/{MAX_SENTENCES} 句")
        
        round_start_count = len(all_items)
        
        # 继续采集
        for url in urls:
            all_items = dedupe_items(all_items)
            if len(all_items) >= MAX_SENTENCES:
                break
            
            print(f"\n[补充轮 {extra_round}] 采集: {url}")
            batch = collector.sentences_from_url(
                url,
                discover=True,
                articles_per_list=ARTICLES_PER_LIST,
            )
            for it in batch:
                key = _sentence_key(it.ko)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_items.append(it)
                
                all_items = dedupe_items(all_items)
                if len(all_items) >= MAX_SENTENCES:
                    break
        
        # 统计本轮结果
        round_new_count = len(all_items) - round_start_count
        print(f"\n补充轮 {extra_round} 结束，本轮新增 {round_new_count} 句，累计 {len(all_items)} 句")
        
        # 累计统计
        total_articles_fetched += collector.stats['articles_fetched']
        total_articles_skipped += collector.stats['articles_skipped']
        total_sentences_skipped_dup += collector.stats['sentences_skipped_dup']
        total_sentences_skipped_long += collector.stats['sentences_skipped_long']
        total_sentences_skipped_short += collector.stats.get('sentences_skipped_short', 0)
        total_sentences_skipped_invalid += collector.stats.get('sentences_skipped_invalid', 0)
        
        # 重置
        collector.stats['articles_fetched'] = 0
        collector.stats['articles_skipped'] = 0
        collector.stats['sentences_skipped_dup'] = 0
        collector.stats['sentences_skipped_long'] = 0
        collector.stats['sentences_skipped_short'] = 0
        collector.stats['sentences_skipped_invalid'] = 0
        
        # 检查是否有新增
        if round_new_count == 0:
            print(f"\n⚠️  无法收集到更多句子，停止！")
            break
        
        history.save()
        extra_round += 1
    
    # 最终截取到目标数量
    all_items = all_items[:MAX_SENTENCES]

    # 设置状态
    for it in all_items:
        it.status = "已采集"

    # 保存历史
    history.register_sentences(all_items)
    history.save()
    print(f"\n最终 {history.summary()}")

    # 保存Excel
    out = resolve_output_path(OUTPUT_FILE)
    write_review_xlsx(out, all_items)

    # 计算总用时
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    print("\n" + "=" * 60)
    print(f"完成: {len(all_items)} 句 → {out}")
    print(f"  总用时: {hours}小时 {minutes}分 {seconds}秒")
    print(f"  累计新爬文章 {total_articles_fetched} 篇")
    print(f"  累计跳过已爬文章 {total_articles_skipped} 篇")
    print(f"  累计跳过重复句子 {total_sentences_skipped_dup} 条")
    print(f"  累计过滤超长句子 {total_sentences_skipped_long} 条 (>{MAX_SENTENCE_CHARS}字符)")
    print(f"  累计过滤过短句子 {total_sentences_skipped_short} 条 (<15字符)")
    print(f"  累计过滤无效句子 {total_sentences_skipped_invalid} 条 (括号/引号不匹配等)")
    if KONLPY_AVAILABLE:
        print(f"  ✅ 使用 Konlpy Okt 分词器进行质量验证")
    
    # 提示是否达到目标
    if len(all_items) < MAX_SENTENCES:
        print(f"\n⚠️  警告: 未达到目标 {MAX_SENTENCES} 句，仅收集到 {len(all_items)} 句")
        print("   可能原因:")
        print("   - 可用的新文章不足")
        print("   - 大部分文章已在历史记录中")
        print("   - 可尝试: 添加更多URL到 urls_example.txt 或清空历史记录重新运行")
    else:
        print(f"\n✅ 成功达到目标 {MAX_SENTENCES} 句！")
    print("=" * 60)


if __name__ == "__main__":
    main()
