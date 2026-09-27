"""Шаг 4 домашки: агент из main.py с эпизодической и семантической памятью, демонстрация сессий и график токенов.

Эпизодическая память: журнал реплик logs/episodes.jsonl с сессией и временем, в контекст не идёт.
Семантическая: факты о пользователе в коллекции Milvus facts, ключ факта (city, favorite_team, ...) это primary key,
поэтому новый факт с тем же ключом заменяет старый через upsert, а не ложится рядом.

Нужен поднятый Milvus с коллекцией chunks_lines (python load_index.py). Запуск: python memory.py
"""
import re, json
from datetime import datetime
from functools import partial

import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from pymilvus import DataType

from load_index import DIM, ROOT, embed_cached, ledger, load_cache, load_jsonl, log, milvus, save_cache
from main import MODELS, agent, chat

JOURNAL = ROOT / "logs" / "episodes.jsonl"
TOKENS_PNG = ROOT / "img" / "memory_tokens.png"
WINDOW, FULL = "окно плюс факты", "вся история"
MODE_COLORS = {WINDOW: "#5436A3", FULL: "#F09000"}

CHAT_SYSTEM = ("You are an assistant for questions about 2026 FIFA World Cup matches. Reply in the user's language and briefly. "
               "For any question about a match look facts up with the knowledge_base tool (queries in English), do not repeat "
               "the same call and never answer match facts from memory. If the knowledge base has no answer, say so. "
               "Greetings and statements about the user need no search: just reply. Use what you know about the user "
               "to resolve words like 'my team' or 'my city'.")
FACTS_SYSTEM = ("Ты ведёшь память ассистента о пользователе. Верни один JSON-объект вида "
                "{\"facts\": [{\"key\": \"...\", \"text\": \"...\"}]}: полный обновлённый список устойчивых фактов о пользователе. "
                "key это короткий слот на английском в snake_case: name, city, favorite_team, answer_style и т.п., "
                "text это факт одной фразой на русском в третьем лице: «Живёт в Москве», «Просит отвечать коротко». Если новый факт отменяет старый (переехал, сменил команду, передумал), "
                "оставь key старого факта и замени text. Если пользователь отказался от факта без замены, не включай его. "
                "Разовые вопросы о матчах и мелочи не записывай.")


def remember_episode(session, role, text, path=JOURNAL):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"session": session, "time": datetime.now().isoformat(timespec="seconds"), "role": role, "text": text},
                           ensure_ascii=False) + "\n")


def episodes(session, path=JOURNAL):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["session"] == session]


class FactMemory:
    """Факты о пользователе в своей коллекции Milvus: key -> text, вектор считается по text."""

    def __init__(self, name="facts", embed=partial(embed_cached, tag="embed:memory")):
        self.name, self.embed = name, embed

    def reset(self):
        if milvus().has_collection(self.name):
            milvus().drop_collection(self.name)

    def _collection(self):
        client = milvus()
        if not client.has_collection(self.name):
            schema = client.create_schema(auto_id=False)
            schema.add_field("key", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
            schema.add_field("text", DataType.VARCHAR, max_length=1000)
            index_params = client.prepare_index_params()
            index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
            # Strong: поиск сразу после upsert видит новую версию факта
            client.create_collection(self.name, schema=schema, index_params=index_params, consistency_level="Strong")
        return client

    def all(self):
        if not milvus().has_collection(self.name):
            return {}
        rows = milvus().query(self.name, filter='key != ""', output_fields=["key", "text"])
        return {r["key"]: r["text"] for r in rows}

    def upsert(self, facts):
        if not facts:
            return
        vectors = self.embed(list(facts.values()))
        self._collection().upsert(self.name, [{"key": k, "vector": v.tolist(), "text": t}
                                              for (k, t), v in zip(facts.items(), vectors)])

    def sync(self, facts):
        # facts это полный новый список: чего в нём нет, пользователь отменил
        gone = sorted(set(self.all()) - set(facts))
        if gone:
            milvus().delete(self.name, ids=gone)
        self.upsert(facts)

    def recall(self, query, k=3):
        if not milvus().has_collection(self.name):
            return []
        hits = milvus().search(self.name, data=self.embed([query]).tolist(), anns_field="vector", limit=k,
                               output_fields=["text"])[0]
        return [h["entity"]["text"] for h in hits]


class Fact(BaseModel):
    key: str = Field(description="слот факта в snake_case: name, city, favorite_team")
    text: str = Field(description="факт о пользователе одной фразой")


class Facts(BaseModel):
    facts: list[Fact]


def json_from(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    cleaned = re.sub(r'(\\["\\/bfnrtu])|\\', lambda e: e.group(1) or "\\\\", m.group(0) if m else "{}")
    return json.loads(cleaned, strict=False)


def extract_facts(dialog, known):
    prompt = f"Известные факты: {json.dumps(known, ensure_ascii=False)}\n\nДиалог:\n{dialog}"
    msg = chat([{"role": "system", "content": FACTS_SYSTEM}, {"role": "user", "content": prompt}], MODELS["cheap"], tag="memory")
    # пустой или битый ответ падает на валидации, а не стирает память через sync
    return {f.key: f.text for f in Facts.model_validate(json_from(msg.get("content"))).facts}


def talk(session, history, text, mem, mode=WINDOW, model=MODELS["cheap"]):
    remember_episode(session, "user", text)
    facts = mem.recall(text) if mode == WINDOW else []
    system = CHAT_SYSTEM + ("\nWhat you know about the user: " + "; ".join(facts) if facts else "")
    window = history[-4:] if mode == WINDOW else history
    out = agent(text, model, system=system, history=window, tag=f"dialog: {mode}")
    if mode == WINDOW:
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": out["answer"]}]
    else:
        history[:] = out["messages"][1:]  # вся переписка агента, с вызовами knowledge_base и выдачей
    remember_episode(session, "assistant", out["answer"])
    return out["answer"], facts


def end_session(session, mem):
    dialog = "\n".join(f"{e['role']}: {e['text'][:400]}" for e in episodes(session))
    mem.sync(extract_facts(dialog, mem.all()))
    return mem.all()


def token_bench(questions, mem):
    rows = []
    for mode in (FULL, WINDOW):
        history = []
        for turn, question in enumerate(questions, 1):
            start = len(ledger.calls)
            talk(f"bench/{mode}", history, question, mem, mode)
            calls = ledger.calls[start:]
            rows.append({"mode": mode, "turn": turn, "calls": sum(c["tag"].startswith("dialog") for c in calls),
                         "prompt": sum(c["prompt"] for c in calls if c["tag"].startswith("dialog")),
                         "cost": sum(c["cost"] for c in calls)})
    return pd.DataFrame(rows)


def plot_tokens(bench, path=TOKENS_PNG):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for mode, part in bench.groupby("mode", sort=False):
        ax.plot(part["turn"], part["prompt"], marker="o", lw=2.2, color=MODE_COLORS[mode],
                label=f"{mode}: ${part['cost'].sum():.4f} за диалог")
    ax.set_xlabel("реплика диалога"); ax.set_ylabel("токенов на входе за реплику"); ax.set_xticks(bench["turn"].unique())
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def say(session, history, mem, text):
    answer, facts = talk(session, history, text, mem)
    print(f"\n[{session}] пользователь: {text}\n  факты из памяти: {facts or 'нет'}\n  агент: {' '.join(answer.split())}")


def print_facts(title, facts):
    print(f"\n{title}:\n" + "\n".join(f"  {k}: {v}" for k, v in facts.items()))


# реплики диалога для замера: вопросы квиза о матчах Марокко и Испании, по шесть
BENCH_IDS = ["fresh-13", "fresh-15", "fresh-18", "fresh-75", "fresh-88", "fresh-96",
             "fresh-42", "fresh-44", "fresh-47", "fresh-82", "fresh-92", "fresh-100"]


def main():
    log.info("кэш эмбеддингов: %d векторов", load_cache())
    mem, run = FactMemory(), datetime.now().strftime("%Y%m%d-%H%M%S")
    mem.reset()
    try:
        s1, first = f"{run}/s1", []
        say(s1, first, mem, "Привет! Меня зовут Женя, я из Москвы и болею за сборную Японии. Отвечай коротко.")
        say(s1, first, mem, "Какой юбилей турнира случился в матче Тунис — Япония?")
        print_facts("факты после сессии 1", end_session(s1, mem))

        s2, second = f"{run}/s2", []
        say(s2, second, mem, "Кто забил за мою сборную в матче с Нидерландами?")
        say(s2, second, mem, "Японию я больше не смотрю, теперь болею за сборную Марокко.")
        print_facts("факты после сессии 2", end_session(s2, mem))

        s3, third = f"{run}/s3", []
        say(s3, third, mem, "Кто забил автогол в матче моей сборной с Гаити?")

        print("\nжурнал этого запуска (в контекст не идёт):")
        for e in episodes(s1) + episodes(s2) + episodes(s3):
            print(f"  {e['time']} {e['session']} {e['role']}: {' '.join(e['text'].split())[:90]}")

        by_id = {t["id"]: t for t in load_jsonl("match_quiz_evaluated.jsonl")}
        bench = token_bench([by_id[i]["question"] for i in BENCH_IDS], mem)
        print("\n" + bench.pivot(index="turn", columns="mode", values="prompt").to_string())
        print(bench.groupby("mode", sort=False).agg(calls=("calls", "sum"), prompt=("prompt", "sum"), cost=("cost", "sum"))
              .round(6).to_string())
        print(f"график: {plot_tokens(bench)}")
    finally:
        save_cache()
        log.info("в лог леджера записано %d вызовов", ledger.save())
    if ledger.calls:
        print(ledger.table().to_string())
    print(f"итого потрачено ${ledger.total:.6f}")


if __name__ == "__main__":
    main()
