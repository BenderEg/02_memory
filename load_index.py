import os, re, json, time, hashlib, threading, logging, unicodedata
from functools import cache
from typing import Literal
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pymilvus import MilvusClient, DataType, Function, FunctionType

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL, DIM = "openai/text-embedding-3-small", 512
DATA = ROOT / "data"
CACHE_PATH = ROOT / "emb_cache.npz"
LEDGER_LOG = ROOT / "logs" / "ledger.jsonl"
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
RECALL_PNG = ROOT / "img" / "recall.png"
SEARCH_PNG = ROOT / "img" / "search_chars.png"
COLORS = {"lines": "#5436A3", "chars": "#F09000"}
SEARCH_COLORS = {"vector only": "#5436A3", "keyword only": "#1A9E8F", "hybrid RRF": "#F09000"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("load_index")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


@dataclass
class Ledger:
    calls: list = field(default_factory=list)
    local: threading.local = field(default_factory=threading.local)

    def add(self, tag, model, usage, seconds=0.0):
        cost = usage.get("cost") or 0.0
        self.calls.append({"tag": tag, "model": model.split("/")[-1], "prompt": usage.get("prompt_tokens", 0),
                           "completion": usage.get("completion_tokens", 0), "cost": cost, "seconds": round(seconds, 2)})
        self.local.spent = self.mine + cost
        log.info("ledger %s %s prompt=%s cost=$%.6f", tag, model, usage.get("prompt_tokens", 0), cost)
        return cost

    @property
    def total(self):
        return sum(c["cost"] for c in self.calls)

    @property
    def mine(self):
        return getattr(self.local, "spent", 0.0)

    def table(self):
        frame = pd.DataFrame(self.calls, columns=["tag", "model", "prompt", "completion", "cost", "seconds"])
        return frame.groupby(["tag", "model"]).agg(calls=("cost", "size"), prompt=("prompt", "sum"),
                                                   completion=("completion", "sum"), cost=("cost", "sum")).round(5)

    def save(self, path=LEDGER_LOG):
        if not self.calls:
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        run = datetime.now().isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as f:
            for call in self.calls:
                f.write(json.dumps({"run": run, **call}, ensure_ascii=False) + "\n")
        return len(self.calls)


ledger = Ledger()


def headers():
    # ключ нужен только для запроса к API: импорт, тесты и прогоны по кэшу и сохранённым ответам работают без него
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("нет ключа: положите OPENROUTER_API_KEY в .env рядом со скриптом")
    return {"Authorization": f"Bearer {key}", "X-Title": "agents-course-seminar02"}


def post_with_retry(url, body, attempts=5):
    problem = "нет ответа"
    auth = headers()
    for attempt in range(attempts):
        try:
            r = requests.post(url, json=body, headers=auth, timeout=120)
            if r.status_code == 200:
                return r.json()
            problem = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            problem = type(e).__name__
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(problem)


def load_jsonl(name):
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(name, rows):
    with (DATA / name).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_lines(row):
    chunks, pending = [], []
    for line in row["text"].splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        if len(line) <= 40:
            pending.append(line)
            continue
        chunks.append({"stage": row["stage"], "match": row["match"], "url": row["url"], "text": " ".join(pending + [line])})
        pending = []
    if pending:  # короткие строки в самом конце: приклеиваем к последнему куску
        if chunks:
            chunks[-1]["text"] += " " + " ".join(pending)
        else:
            chunks.append({"stage": row["stage"], "match": row["match"], "url": row["url"], "text": " ".join(pending)})
    return chunks


def chunk_chars(row, size=400):
    text = " ".join(row["text"].split())
    return [{"stage": row["stage"], "match": row["match"], "url": row["url"], "text": text[i:i + size]}
            for i in range(0, len(text), size)]


CHUNKERS = {"lines": chunk_lines, "chars": chunk_chars}

EMB = {}


def load_cache():
    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH)
        EMB.update({str(k): v.astype(np.float32) for k, v in zip(z["keys"], z["vectors"])})
    return len(EMB)


def save_cache():
    if EMB:
        np.savez_compressed(CACHE_PATH, keys=np.array(list(EMB)), vectors=np.stack(list(EMB.values())).astype(np.float16))


def embed_batch(texts, tag="embed"):
    started = time.perf_counter()
    data = post_with_retry(EMBED_URL, {"model": EMBED_MODEL, "input": texts, "dimensions": DIM})
    ledger.add(tag, EMBED_MODEL, data.get("usage") or {}, time.perf_counter() - started)
    return [row["embedding"] for row in data["data"]]


def key_of(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_cached(texts, tag="embed"):
    missing = list(dict.fromkeys(t for t in texts if key_of(t) not in EMB))  # set нельзя т.к. нет порядка
    for i in range(0, len(missing), 64):
        batch = missing[i:i + 64]
        for text, vector in zip(batch, embed_batch(batch, tag)):
            EMB[key_of(text)] = np.asarray(vector, dtype=np.float32)

    vectors = np.stack([EMB[key_of(t)] for t in texts])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def emb_text(chunk):
    return f"{chunk['stage']} | {chunk['match']}: {chunk['text']}"


def build(mode, rows):
    chunks = [c for row in rows for c in CHUNKERS[mode](row)]
    vectors = embed_cached([emb_text(c) for c in chunks], tag=f"embed:{mode}")
    save_jsonl(f"chunks_{mode}.jsonl", chunks)
    return chunks, vectors


def index_to_milvus(client, name, chunks, vectors):
    if client.has_collection(name):
        client.drop_collection(name)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)  # stage/match/url/text лягут динамическими полями
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vectors.shape[1])
    # по словам ищем в той же строке, что эмбеддим: у кусков chars из середины текста иначе нет названий команд
    schema.add_field("doc", DataType.VARCHAR, max_length=65535, enable_analyzer=True, analyzer_params={"type": "english"})
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function("bm25", FunctionType.BM25, input_field_names="doc", output_field_names="sparse"))
    index_params = client.prepare_index_params()
    index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(name, schema=schema, index_params=index_params)
    rows = [{"id": i, "vector": v.tolist(), "doc": emb_text(c), **c} for i, (c, v) in enumerate(zip(chunks, vectors))]
    for i in range(0, len(rows), 1000):
        client.insert(name, rows[i:i + 1000])
    client.flush(name)
    return client.get_collection_stats(name)


NUMBERS = {word: str(n) for n, word in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                                 "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUMBERS |= {word: str(n) for n, word in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
                                                  "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth "
                                                  "twentieth".split(), 1)}


def norm(text):
    text = "".join(c for c in unicodedata.normalize("NFKD", str(text).lower()) if not unicodedata.combining(c))  # Souček -> soucek
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 64,478 -> 64478
    text = re.sub(r"(?<=\d)['’]", " minute", text)  # 88' -> 88 minute, как 88th minute
    words = re.sub(r"\b(a|an|the)\b|[^\w\s]", " ", text).split()
    words = [re.sub(r"^(\d+)(st|nd|rd|th)$", r"\1", w) for w in words]  # 1000th -> 1000
    return " ".join(NUMBERS.get(w, w) for w in words)


def is_gold(chunk, task):
    # evidence в квизе пересказан LLM, а не процитирован, поэтому совпадение по словам не требуем:
    # хватает нужного матча (url) и ответа в тексте куска
    return chunk["url"] == task["url"] and norm(task["answer"]) in norm(chunk["text"])


def search_vector(client, name, q_vecs, limit=20):
    results = client.search(name, data=q_vecs.tolist(), anns_field="vector", limit=limit)
    return [[h["id"] for h in hits] for hits in results]


def search_keyword(client, name, questions, limit=20):
    # BM25 внутри Milvus: сырой текст вопроса, токенизирует и считает сама база
    results = client.search(name, data=questions, anns_field="sparse", limit=limit, search_params={"metric_type": "BM25"})
    return [[h["id"] for h in hits] for hits in results]


def rrf(rankings, k=5, K=60):
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1 / (K + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:k]


def search_hybrid(by_vector, by_keyword, limit=20):
    return [rrf([v, w], limit) for v, w in zip(by_vector, by_keyword)]


def recall_at(ranked, chunks, tasks, ks):
    return {k: float(np.mean([any(is_gold(chunks[i], t) for i in r[:k]) for r, t in zip(ranked, tasks)])) for k in ks}


def recall_curve(client, name, chunks, q_vecs, tasks, ks=(1, 3, 4, 5, 10, 20)):
    return recall_at(search_vector(client, name, q_vecs, max(ks)), chunks, tasks, ks)


def compare_search(client, name, chunks, q_vecs, tasks, ks=(1, 3, 5), limit=20):
    by_vector = search_vector(client, name, q_vecs, limit)
    by_keyword = search_keyword(client, name, [t["question"] for t in tasks], limit)
    rankings = {"vector only": by_vector, "keyword only": by_keyword, "hybrid RRF": search_hybrid(by_vector, by_keyword, limit)}
    return pd.DataFrame({method: {f"recall@{k}": v for k, v in recall_at(r, chunks, tasks, ks).items()}
                         for method, r in rankings.items()}).T


@cache
def milvus():
    return MilvusClient(uri=MILVUS_URI, token=os.getenv("MILVUS_TOKEN", ""))


def search_milvus(name, query, k=5, flt=""):
    q_vec = embed_cached([query], tag="embed:kb")
    hits = milvus().search(name, data=q_vec.tolist(), anns_field="vector", limit=k, filter=flt,
                           output_fields=["stage", "match", "url", "text"])[0]
    return [h["entity"] for h in hits]


STAGES = tuple(f"2026 FIFA World Cup Group {g}" for g in "ABCDEFGHIJKL") + ("2026 FIFA World Cup round of 32",
                                                                          "2026 FIFA World Cup knockout stage")


class KBArgs(BaseModel):
    query: str = Field(description="short search query in English: teams, players, stadium, referee or other key terms of the match")
    # enum вместо свободной строки: агент выдумывал '2026 FIFA World Cup Group Stage' и получал пустую выдачу
    stage: Literal[("",) + STAGES] = Field(default="", description="optional stage to search within; leave empty unless you are sure "
                                                                   "which group or round the match belongs to")


def kb_hits(query, stage=""):
    # vector по нарезке lines: лучший recall@k, гибрид с BM25 его не поднимает
    return search_milvus("chunks_lines", query, 5, f"stage == {json.dumps(stage)}" if stage else "")


def format_hits(hits):
    if not hits:
        return "nothing found"
    return "\n".join(f"[{h['stage']} / {h['match']}] {h['text']} ({h['url']})" for h in hits)


def knowledge_base(query: str, stage: str = "") -> str:
    return format_hits(kb_hits(query, stage))


KB_SCHEMA = {"type": "function", "function": {
    "name": "knowledge_base",
    "description": "Searches the local knowledge base of 2026 FIFA World Cup match reports and returns the five closest "
                   "passages with their stage, match and source link",
    "parameters": KBArgs.model_json_schema()}}


def plot_recall(curves, path=RECALL_PNG):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for mode, curve in curves.items():
        ax.plot([str(k) for k in curve], [v * 100 for v in curve.values()], marker="o", lw=2.2, color=COLORS[mode], label=mode)
    ax.set_xlabel("k, сколько кусков берём"); ax.set_ylabel("recall@k, %"); ax.set_ylim(0, 105); ax.legend(frameon=False); ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_search(table, path=SEARCH_PNG):
    ax = table.T.plot.bar(figsize=(7, 3.6), color=[SEARCH_COLORS[m] for m in table.index], rot=0)
    ax.set_xlabel("chars, k"); ax.set_ylabel("recall@k"); ax.set_ylim(0, 1.05); ax.legend(frameon=False); ax.grid(alpha=0.3, axis="y")
    path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return path


def main():
    rows = load_jsonl("match_descriptions.jsonl")
    tasks = load_jsonl("match_quiz_evaluated.jsonl")
    log.info("кэш эмбеддингов: %d векторов", load_cache())
    client = milvus()
    curves = {}
    try:
        q_vecs = embed_cached([t["question"] for t in tasks], tag="embed:questions")
        for mode in CHUNKERS:
            chunks, vectors = build(mode, rows)
            stats = index_to_milvus(client, f"chunks_{mode}", chunks, vectors)
            print(f"{mode}: {len(chunks)} кусков, векторы {vectors.shape}, в Milvus chunks_{mode}: {stats['row_count']}")
            curves[mode] = recall_curve(client, f"chunks_{mode}", chunks, q_vecs, tasks)
        print(pd.DataFrame(curves).round(2).to_string())
        print(f"график: {plot_recall(curves)}")
        search = compare_search(client, "chunks_chars", load_jsonl("chunks_chars.jsonl"), q_vecs, tasks)
        print(search.round(2).to_string())
        print(f"график: {plot_search(search)}")
    finally:
        save_cache()
        log.info("в лог леджера записано %d вызовов", ledger.save())
    if ledger.calls:
        print(ledger.table().to_string())
    print(f"итого потрачено ${ledger.total:.6f}")


if __name__ == "__main__":
    main()
