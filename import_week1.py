"""Две строки прошлой недели для таблицы report: трейсы первой домашки -> data/week1_baseline.jsonl.

Разовый скрипт, результат закоммичен, поэтому main.py читает только data/ и работает без ../01_ReAct.
Сильная модель без инструментов: traces/без инструментов/claude-sonnet-4.6.
Дешёвая модель с живым поиском: traces/повторные запуски/поиск и чтение страницы/gpt-4o-mini, а чего там нет
(вопросы, верные с первого раза), из traces/поиск и чтение страницы/gpt-4o-mini. Так же собран итог 01_ReAct
«после правок»: $0.00094 за вопрос, 0.44 верных.
Запуск: python import_week1.py [путь к 01_ReAct]
"""
import sys, csv, json
from functools import cache
from pathlib import Path

from load_index import DATA, ROOT, load_jsonl, save_jsonl

REACT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "01_ReAct"
# ключ: (config, model, [(папка трейсов, таблица с числом шагов)]), повторный запуск раньше первого
RUNS = {"week1-sonnet": ("без инструментов, неделя 1", "claude-sonnet-4.6",
                         [("traces/без инструментов/claude-sonnet-4.6", "results/без инструментов__strong.csv")]),
        "week1-search": ("живой поиск, неделя 1", "gpt-4o-mini",
                         [("traces/повторные запуски/поиск и чтение страницы/gpt-4o-mini",
                           "results/повторные запуски/поиск и чтение страницы__cheap.csv"),
                          ("traces/поиск и чтение страницы/gpt-4o-mini", "results/поиск и чтение страницы__cheap.csv")])}


@cache
def steps_of(table):
    # шаг в 01_ReAct — итерация цикла агента, в трейс он не пишется, только в таблицу results
    with (REACT / table).open(encoding="utf-8") as f:
        return {row["id"]: int(row["steps"]) for row in csv.DictReader(f)}


def trace(sources, task_id):
    # первая папка, где есть трейс вопроса: повторный запуск важнее первого
    folder, table = next((d, t) for d, t in sources if (REACT / d / f"{task_id}.json").exists())
    return json.loads((REACT / folder / f"{task_id}.json").read_text(encoding="utf-8")), steps_of(table)[task_id]


def main():
    tasks = load_jsonl("match_quiz_evaluated.jsonl")
    rows = []
    for key, (config, model, sources) in RUNS.items():
        for task in tasks:
            t, steps = trace(sources, task["id"])
            assert t["task"]["question"] == task["question"], task["id"]
            rows.append({"key": key, "config": config, "model": model, "id": task["id"], "source": task["source"],
                         "question": task["question"], "gold": task["answer"], "evidence": task["evidence"],
                         "answer": t["answer"], "cost": t["cost"],
                         "searches": sum(m.get("role") == "tool" for m in t["messages"]),
                         "steps": steps, "hits": []})
    save_jsonl("week1_baseline.jsonl", rows)
    print(f"{len(rows)} строк -> {DATA / 'week1_baseline.jsonl'}")


if __name__ == "__main__":
    main()
