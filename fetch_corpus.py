# -*- coding: utf-8 -*-
"""Корпус для семинара 2: полные тексты обзорных страниц Википедии про 2026 год.

Запуск: python fetch_corpus.py
Результат: data/corpus_2026.jsonl (страница, ссылка, текст) и data/fresh_2026.jsonl (вопросы первой домашки,
у которых цитата-эталон по-прежнему находится в корпусе; поле evidence нужно для честного recall@k).
"""
import json, re, sys, time
from pathlib import Path
import requests

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
QUESTIONS = HERE.parent / "seminar01" / "data" / "fresh_2026.jsonl"
WIKI = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "agents-course-seminar02/1.0 (https://postypashki.ru; educational project)"}


def page_text(title):
    for attempt in range(4):
        try:
            r = requests.get(WIKI, timeout=40, headers=HEADERS, params={"action": "query", "prop": "extracts", "explaintext": 1,
                                                                        "titles": title, "redirects": 1, "format": "json"})
            if r.status_code == 200:
                page = next(iter(r.json()["query"]["pages"].values()))
                return page.get("title", title), page.get("extract", "")
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return title, ""


def norm(t):
    return " ".join(re.sub(r"[^\w\s]", " ", str(t).lower()).split())


if __name__ == "__main__":
    tasks = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    titles = sorted({t["page"] for t in tasks})
    corpus = []
    for title in titles:
        real, text = page_text(title)
        if text:
            corpus.append({"page": real, "url": "https://en.wikipedia.org/wiki/" + real.replace(" ", "_"), "text": text})
        print(f"{real:42s} {len(text):8d} символов")
    with (DATA / "corpus_2026.jsonl").open("w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    texts = {c["page"]: norm(c["text"]) for c in corpus}
    kept = [t for t in tasks if norm(t["evidence"])[:120] in texts.get(t["page"], "")]
    with (DATA / "fresh_2026.jsonl").open("w", encoding="utf-8") as f:
        for n, t in enumerate(kept):
            f.write(json.dumps({"id": f"fresh-{n}", "source": "fresh", "question": t["question"], "answer": t["answer"],
                                "evidence": t["evidence"], "page": t["page"], "url": t["url"],
                                "agent_ok": bool(t.get("agent_ok"))}, ensure_ascii=False) + "\n")
    lines = sum(len([l for l in c["text"].splitlines() if len(l.strip()) >= 40]) for c in corpus)
    print(f"\nстраниц: {len(corpus)}, символов: {sum(len(c['text']) for c in corpus)}, содержательных строк: {lines}")
    print(f"вопросов с цитатой, найденной в корпусе: {len(kept)} из {len(tasks)}")
