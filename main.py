"""Шаг 3 домашки: RAG всегда против агента с knowledge_base, таблица report, картинка money_chart, отказы и разбор провалов.

Нужен поднятый Milvus с коллекцией chunks_lines (python load_index.py). Запуск: python main.py [--rerun [rag-cheap agent-cheap rag-mid]]
Без --rerun ответы моделей берутся из data/eval_results.jsonl и только пересчитываются проверкой, повторный запуск не платит.
--rerun без имён спрашивает модели заново во всех конфигурациях, с именами только в перечисленных.
"""
import re, time, argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import matplotlib.pyplot as plt

from load_index import (DATA, ROOT, KBArgs, KB_SCHEMA, embed_cached, format_hits, is_gold, kb_hits, ledger, load_cache,
                        load_jsonl, log, milvus, norm, post_with_retry, save_cache, search_milvus)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = {"cheap": "openai/gpt-4o-mini", "mid": "anthropic/claude-haiku-4.5"}
MODEL_COLORS = {"gpt-4o-mini": "#5436A3", "claude-haiku-4.5": "#1A9E8F", "claude-sonnet-4.6": "#F09000"}
WEEK1_MARKER = "s"  # квадрат: прошлая неделя, круг: эта
RESULTS = DATA / "eval_results.jsonl"
WEEK1 = DATA / "week1_baseline.jsonl"  # сильная модель без инструментов и живой поиск из 01_ReAct, собирает import_week1.py
MONEY_PNG = ROOT / "img" / "money_chart.png"

RAG_SYSTEM = ("Answer the question about 2026 FIFA World Cup matches using only the numbered sources. Cite the source number like [2]. "
              "If the sources do not contain the answer, reply exactly NOT_FOUND. The last line must be FINAL: <short answer>.")
AGENT_SYSTEM = ("You answer questions about 2026 FIFA World Cup matches. Never answer from memory: look facts up with the knowledge_base tool, "
                "do not repeat the same call. When done, write the last line as FINAL: <short answer>, "
                "or NOT_FOUND if the knowledge base has no answer.")


def chat(messages, model, tools=None, tag="chat"):
    body = {"model": model, "messages": messages, "temperature": 0, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
    started = time.perf_counter()
    data = post_with_retry(CHAT_URL, body)
    ledger.add(tag, model, data.get("usage") or {}, time.perf_counter() - started)
    return data["choices"][0]["message"]


def rag_answer(question, model, k=5):
    hits = search_milvus("chunks_lines", question, k)
    sources = "\n".join(f"[{i + 1}] ({h['stage']} / {h['match']}) {h['text']}" for i, h in enumerate(hits))
    before = ledger.mine
    msg = chat([{"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"}], model, tag="rag")
    return {"answer": msg.get("content") or "", "hits": hits, "cost": ledger.mine - before, "searches": 1}


def run_tool(call, hits):
    try:
        args = KBArgs.model_validate_json(call["function"]["arguments"])
        found = kb_hits(args.query, args.stage)
        hits += found
        result = format_hits(found)
    except Exception as e:
        log.warning("инструмент %s упал: %s", call["function"]["name"], e)
        result = f"tool error {call['function']['name']}: {e}"
    return {"role": "tool", "tool_call_id": call["id"], "content": result[:3000]}


def agent(question, model, max_steps=6, system=AGENT_SYSTEM, history=(), tag="agent"):
    # history: прошлые реплики диалога между системным промптом и вопросом, их даёт память (memory.py)
    messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": question}]
    seen, hits, before = set(), [], ledger.mine
    for step in range(max_steps):
        msg = chat(messages, model, tools=[KB_SCHEMA], tag=tag)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        keys = {(c["function"]["name"], c["function"]["arguments"]) for c in calls}
        if not calls or keys & seen or step == max_steps - 1:
            break
        seen |= keys
        messages += [run_tool(c, hits) for c in calls]
    if calls:
        messages += [{"role": "tool", "tool_call_id": c["id"], "content": "call rejected: answer from what you already have"} for c in calls]
        msg = chat(messages, model, tag=tag)
        messages.append(msg)
    searches = sum(m.get("role") == "tool" for m in messages[len(history) + 2:] if isinstance(m, dict))
    unique = list({(h["url"], h["text"]): h for h in hits}.values())
    return {"answer": msg.get("content") or "", "hits": unique, "cost": ledger.mine - before, "searches": searches,
            "messages": messages}


def final_answer(text):
    m = re.findall(r"FINAL:\s*(.+)", text or "")
    return m[-1].strip() if m else (text or "").strip()


# отказ ищем только в строке FINAL: у прошлой недели в промпте был FINAL: unknown, sonnet отказывалась ещё и по-русски,
# а Haiku пишет отказ словами: "FINAL: The sources do not contain information about ..."
REFUSAL = re.compile(r"NOT_FOUND|^unknown\b|unable to|could not|couldn't|cannot|do(es)? not contain|no information|"
                     r"недоступн|недостаточно|не удалось", re.I)
# слова без содержания: короткий ответ из одних них ("World Cup") не засчитывается за эталон "1994 FIFA World Cup"
FILLER = {"of", "in", "on", "at", "by", "to", "for", "and", "match", "world", "cup", "fifa", "2026"}


def is_refusal(answer):
    return bool(REFUSAL.search(final_answer(answer)))


def answer_words(text):
    # U.S. -> united states, goals -> goal (только слова от 4 букв: "is", "gus" не трогаем)
    text = re.sub(r"\bU\.?S\.?A?\b", "United States", str(text))
    return [re.sub(r"(?<=\w{3})s$", "", w) for w in norm(text).split()]


def is_correct(task, answer):
    # верно, если в ответе все слова эталона ("1,000th FIFA World Cup match" для "1,000th World Cup match")
    # или ответ короче эталона, но все его содержательные слова из эталона ("10" для "10 goals",
    # "Lincoln Financial Field" для "Lincoln Financial Field, Philadelphia"); порядок не важен
    if is_refusal(answer):
        return False
    gold, got = answer_words(task["answer"]), set(answer_words(final_answer(answer)))
    core = got - FILLER
    return bool(gold) and (all(w in got for w in gold) or bool(core) and core <= set(gold))


def score(results, tasks):
    by_id = {t["id"]: t for t in tasks}
    results["final"] = results["answer"].map(final_answer)
    results["correct"] = [is_correct(by_id[i], a) for i, a in zip(results["id"], results["answer"])]
    results["found"] = [any(is_gold(h, by_id[i]) for h in hits) for i, hits in zip(results["id"], results["hits"])]
    results["refused"] = results["answer"].map(is_refusal)  # по строке FINAL: "NOT_FOUND ... FINAL: <ответ>" не отказ
    return results


def evaluate(key, tasks, workers=8):
    config, model, fn = CONFIGS[key]

    def one(task):
        try:
            out = fn(task["question"], model)
        except Exception as e:
            log.exception("%s / %s упал на %s", config, model, task["id"])
            out = {"answer": f"ошибка: {e}", "hits": [], "cost": 0.0, "searches": 0}
        return {"key": key, "config": config, "model": model.split("/")[-1], "id": task["id"], "source": task["source"],
                "question": task["question"], "gold": task["answer"], "evidence": task["evidence"], "answer": out["answer"],
                "cost": out["cost"], "searches": out["searches"],
                "hits": [{f: h[f] for f in ("stage", "match", "url", "text")} for h in out["hits"]]}
    with ThreadPoolExecutor(workers) as pool:
        return score(pd.DataFrame(list(pool.map(one, tasks))), tasks)


def report(results):
    answerable = results[results["source"] != "unanswerable"]
    g = answerable.groupby(["config", "model"], sort=False)
    table = g.agg(n=("correct", "size"), accuracy=("correct", "mean"), cost_per_task=("cost", "mean"),
                  searches=("searches", "mean")).reset_index()
    table["cost_per_correct"] = table["cost_per_task"] / table["accuracy"].replace(0, float("nan"))
    return table.round(6)


def refusals(results):
    unanswerable = results["source"] == "unanswerable"
    rows = []
    for (config, model), part in results.groupby(["config", "model"], sort=False):
        honest, wrong = part[unanswerable.loc[part.index]], part[~unanswerable.loc[part.index]]
        # на прошлой неделе вопросов без ответа не было
        rows.append({"config": config, "model": model,
                     "честно отклонил": f"{int(honest['refused'].sum())} / {len(honest)}" if len(honest) else "—",
                     "отклонил зря": f"{int(wrong['refused'].sum())} / {len(wrong)}"})
    return pd.DataFrame(rows)


def money_chart(table, path=MONEY_PNG):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for _, r in table.iterrows():
        ax.scatter(r["cost_per_task"] * 100, r["accuracy"] * 100, s=110, color=MODEL_COLORS[r["model"]],
                   marker=WEEK1_MARKER if "неделя 1" in r["config"] else "o")
        ax.annotate(f"{r['config']}\n{r['model']}", (r["cost_per_task"] * 100, r["accuracy"] * 100), fontsize=8,
                    xytext=(7, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("цена вопроса, центы (логарифмическая шкала)"); ax.set_ylabel("доля верных, %")
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1], ["0.01", "0.03", "0.1", "0.3", "1"]); ax.minorticks_off(); ax.set_xlim(0.008, 1.5)
    ax.set_ylim(0, 105); ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def piles_of(results, config, model):
    wrong = results[(results["config"] == config) & (results["model"] == model)
                    & (results["source"] != "unanswerable") & ~results["correct"]]
    return {"поиск не нашёл": wrong[~wrong["found"]], "нашёл, но ответ неверный": wrong[wrong["found"]]}


def diagnose(results, table):
    # table отсортирована от лучшей конфигурации; если у лучшей кучка пуста, пример берём у следующей
    # у прошлой недели нет выдачи поиска в трейсах, на кучки её не разложить
    table = table[~table["config"].str.contains("неделя 1")]
    ranked = list(zip(table["config"], table["model"]))
    counts = pd.DataFrame({f"{c} / {m}": {name: len(p) for name, p in piles_of(results, c, m).items()} for c, m in ranked}).T
    print(f"\nразбор провалов, лучшая конфигурация: {ranked[0][0]} / {ranked[0][1]}\n{counts.to_string()}")
    for name in counts.columns:
        source = next(((c, m, piles_of(results, c, m)[name]) for c, m in ranked if not piles_of(results, c, m)[name].empty), None)
        if source is None:
            print(f"\n=== {name}: ни одной ошибки ни в одной конфигурации")
            continue
        config, model, pile = source
        r = pile.iloc[0]
        print(f"\n=== {name} ({config} / {model}): {r['id']}\nвопрос: {r['question']}\nэталон: {r['gold']}\nцитата: {r['evidence']}\n"
              f"ответ модели: {' '.join(r['answer'].split())}\nконтекст:\n{format_hits(r['hits'])}")


CONFIGS = {"rag-cheap": ("RAG всегда", MODELS["cheap"], rag_answer),
           "agent-cheap": ("агент с knowledge_base", MODELS["cheap"], agent),
           "rag-mid": ("RAG всегда", MODELS["mid"], rag_answer)}


def collect(tasks, rerun):
    # rerun: None — всё из файла, [] — все конфигурации заново, [ключи] — только эти; чего нет в файле, считается всегда
    old = pd.read_json(RESULTS, lines=True, dtype={"id": str}) if RESULTS.exists() else pd.DataFrame(columns=["key"])
    todo = [k for k in CONFIGS if rerun == [] or k in (rerun or []) or k not in set(old["key"])]
    if not todo:
        log.info("беру готовые ответы из %s, для нового прогона запустите с --rerun", RESULTS)
        return score(old, tasks)
    log.info("спрашиваю модели заново: %s", ", ".join(todo))
    embed_cached([t["question"] for t in tasks], tag="embed:questions")  # одной пачкой до потоков
    milvus()
    fresh = {k: evaluate(k, tasks) for k in todo}
    results = pd.concat([fresh[k] if k in fresh else old[old["key"] == k] for k in CONFIGS], ignore_index=True)
    results.to_json(RESULTS, orient="records", lines=True, force_ascii=False)
    return score(results, tasks)


def with_week1(results, tasks):
    # строки прошлой недели не перезапускаются, только проверяются той же is_correct; в таблице они первыми
    week1 = pd.read_json(WEEK1, lines=True, dtype={"id": str})
    return score(pd.concat([week1, results], ignore_index=True), tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", nargs="*", choices=list(CONFIGS), help="без имён заново все конфигурации, с именами только эти")
    args = parser.parse_args()
    answerable, unanswerable = load_jsonl("match_quiz_evaluated.jsonl"), load_jsonl("match_quiz_unanswerable.jsonl")
    log.info("кэш эмбеддингов: %d векторов", load_cache())
    try:
        tasks = answerable + unanswerable
        results = with_week1(collect(tasks, args.rerun), tasks)
        table = report(results)
        print(table.to_string(index=False))
        print(refusals(results).to_string(index=False))
        print(f"график: {money_chart(table)}")
        diagnose(results, table.sort_values(["accuracy", "cost_per_task"], ascending=[False, True]))
    finally:
        save_cache()
        log.info("в лог леджера записано %d вызовов", ledger.save())
    if ledger.calls:
        print(ledger.table().to_string())
    print(f"итого потрачено ${ledger.total:.6f}")


if __name__ == "__main__":
    main()
