import argparse
import csv
import dataclasses
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date


DATE_PATTERNS = [
    re.compile(
        r"(?P<y>20\d{2})\s*[\-/\.年]\s*(?P<m>\d{1,2})\s*(?:[\-/\.月]\s*(?P<d>\d{1,2}))?",
        re.IGNORECASE,
    ),
    re.compile(r"(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})"),
]


@dataclasses.dataclass(frozen=True)
class MemoItem:
    raw_date: str
    y: int | None
    m: int | None
    d: int | None
    category: str
    text: str

    @property
    def year_month(self) -> str:
        if self.y is None or self.m is None:
            return "unknown"
        return f"{self.y:04d}-{self.m:02d}"

    @property
    def year(self) -> str:
        if self.y is None:
            return "unknown"
        return f"{self.y:04d}"


def read_text_file(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "cp936", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def load_categories(path: str) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, list[str]] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, list):
            continue
        out[k] = [str(x) for x in v if str(x).strip()]
    return out


def normalize_text(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_date_from_line(line: str) -> tuple[int | None, int | None, int | None, str | None]:
    for pat in DATE_PATTERNS:
        m = pat.search(line)
        if not m:
            continue
        y = int(m.group("y"))
        mo = int(m.group("m"))
        d = m.groupdict().get("d")
        day = int(d) if d else None
        raw = m.group(0)
        return y, mo, day, raw
    return None, None, None, None


def split_into_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[list[str]] = []
    cur: list[str] = []

    for line in lines:
        if not line.strip():
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(line.rstrip("\n"))

    if cur:
        blocks.append(cur)

    out: list[str] = []
    for b in blocks:
        joined = "\n".join(b).strip()
        if joined:
            out.append(joined)
    return out


def classify(text: str, categories: dict[str, list[str]]) -> str:
    t = text.lower()
    scores: dict[str, int] = {}
    for cat, kws in categories.items():
        score = 0
        for kw in kws:
            if not kw:
                continue
            score += t.count(kw.lower())
        scores[cat] = score
    best_cat = "其他"
    best = 0
    for cat, score in scores.items():
        if score > best:
            best = score
            best_cat = cat
    return best_cat


def build_items(text: str, categories: dict[str, list[str]]) -> list[MemoItem]:
    blocks = split_into_blocks(text)
    items: list[MemoItem] = []

    cur_y: int | None = None
    cur_m: int | None = None
    cur_d: int | None = None
    cur_raw: str = ""

    for b in blocks:
        first_line = b.splitlines()[0]
        y, m, d, raw = extract_date_from_line(first_line)
        if y is None:
            y2, m2, d2, raw2 = extract_date_from_line(b)
            y, m, d, raw = y2, m2, d2, raw2

        raw_date = ""
        body = b

        if y is not None:
            cur_y, cur_m, cur_d = y, m, d
            cur_raw = raw or ""
            raw_date = cur_raw
            if raw:
                body = body.replace(raw, " ", 1)
        else:
            raw_date = cur_raw

        body = normalize_text(body)
        if not body:
            continue
        cat = classify(body, categories)
        items.append(
            MemoItem(
                raw_date=raw_date,
                y=cur_y,
                m=cur_m,
                d=cur_d,
                category=cat,
                text=body,
            )
        )

    return items


def top_keywords(texts: list[str], topk: int = 10) -> list[tuple[str, int]]:
    stop = {
        "今天",
        "这个",
        "那个",
        "一个",
        "一下",
        "然后",
        "但是",
        "因为",
        "所以",
        "还是",
        "需要",
        "准备",
        "计划",
        "完成",
        "感觉",
        "事情",
        "问题",
        "记录",
    }

    c = Counter()
    for t in texts:
        t = t.lower()

        for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}", t):
            if len(w) >= 3:
                c[w] += 1

        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", t):
            if chunk in stop:
                continue
            if len(chunk) <= 6:
                c[chunk] += 1
                continue
            for n in (2, 3, 4):
                for i in range(0, len(chunk) - n + 1):
                    gram = chunk[i : i + n]
                    if gram in stop:
                        continue
                    c[gram] += 1

    for k in list(c.keys()):
        if k.isdigit() or k in stop:
            del c[k]
    return c.most_common(topk)


def render_summary(items: list[MemoItem]) -> str:
    now = date.today().isoformat()
    total = len(items)
    years = sorted({it.year for it in items}, key=lambda x: (x == "unknown", x))

    by_ym: dict[str, list[MemoItem]] = defaultdict(list)
    for it in items:
        by_ym[it.year_month].append(it)

    by_year: dict[str, list[MemoItem]] = defaultdict(list)
    for it in items:
        by_year[it.year].append(it)

    lines: list[str] = []
    lines.append(f"# 备忘录年度/月度统计与总结")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 总条目数：{total}")
    lines.append("")

    lines.append("## 总览")
    lines.append("")
    overall_cat = Counter([it.category for it in items]).most_common()
    for cat, cnt in overall_cat:
        lines.append(f"- {cat}：{cnt}")
    lines.append("")

    lines.append("## 按年")
    lines.append("")
    for y in years:
        its = by_year[y]
        lines.append(f"### {y}")
        lines.append("")
        lines.append(f"- 条目数：{len(its)}")
        cats = Counter([it.category for it in its]).most_common(5)
        if cats:
            lines.append("- Top分类：" + "，".join([f"{c}({n})" for c, n in cats]))
        kws = top_keywords([it.text for it in its], topk=8)
        if kws:
            lines.append("- Top关键词：" + "，".join([f"{w}({n})" for w, n in kws]))
        lines.append("")

        yms = sorted(
            [k for k in by_ym.keys() if k.startswith(f"{y}-")],
            key=lambda s: (s == "unknown", s),
        )
        for ym in yms:
            its_m = by_ym[ym]
            lines.append(f"#### {ym}")
            lines.append("")
            lines.append(f"- 条目数：{len(its_m)}")
            cats_m = Counter([it.category for it in its_m]).most_common(5)
            if cats_m:
                lines.append("- Top分类：" + "，".join([f"{c}({n})" for c, n in cats_m]))
            kws_m = top_keywords([it.text for it in its_m], topk=8)
            if kws_m:
                lines.append("- Top关键词：" + "，".join([f"{w}({n})" for w, n in kws_m]))
            lines.append("")

    unknown_ym = by_ym.get("unknown")
    if unknown_ym:
        lines.append("## 无法识别日期的条目")
        lines.append("")
        lines.append(f"- 条目数：{len(unknown_ym)}")
        lines.append("- 建议：确保每条记录所在块（或首行）包含日期，如 `2024-03-21` 或 `2024年3月21日`。")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_items_csv(items: list[MemoItem], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "month", "day", "year_month", "category", "raw_date", "text"])
        for it in items:
            w.writerow(
                [
                    it.y or "",
                    it.m or "",
                    it.d or "",
                    it.year_month,
                    it.category,
                    it.raw_date,
                    it.text,
                ]
            )


def write_month_stats_csv(items: list[MemoItem], out_path: str) -> None:
    by_ym: dict[str, list[MemoItem]] = defaultdict(list)
    for it in items:
        by_ym[it.year_month].append(it)

    rows: list[tuple[str, int, str]] = []
    for ym, its in by_ym.items():
        cats = Counter([it.category for it in its]).most_common(5)
        cat_str = "|".join([f"{c}:{n}" for c, n in cats])
        rows.append((ym, len(its), cat_str))

    rows.sort(key=lambda r: (r[0] == "unknown", r[0]))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year_month", "count", "top_categories"])
        for ym, cnt, cat_str in rows:
            w.writerow([ym, cnt, cat_str])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", help="备忘录文件路径（.txt/.md等）")
    p.add_argument(
        "--categories",
        default=os.path.join(os.path.dirname(__file__), "memo_categories.json"),
        help="分类关键词配置JSON路径",
    )
    p.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "memo_output"),
        help="输出目录",
    )
    args = p.parse_args()

    categories = load_categories(args.categories)
    text = read_text_file(args.input)
    items = build_items(text, categories)

    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, "memo_summary.md")
    items_path = os.path.join(args.outdir, "memo_items.csv")
    month_path = os.path.join(args.outdir, "memo_month_stats.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(render_summary(items))

    write_items_csv(items, items_path)
    write_month_stats_csv(items, month_path)
    print(summary_path)
    print(items_path)
    print(month_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
