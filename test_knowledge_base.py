"""Три случая для инструмента knowledge_base: ответ нашёлся, фильтр по стадии, пустая выдача.

Нужен поднятый Milvus с коллекцией chunks_lines (python load_index.py). Запуск: python test_knowledge_base.py
"""
from pydantic import ValidationError

from load_index import KBArgs, KB_SCHEMA, knowledge_base, load_cache, save_cache

STAGE = "2026 FIFA World Cup Group A"


def test_finds_answer():
    out = knowledge_base("all-female refereeing trio Czech Republic vs South Africa")
    assert "Tori Penso" in out and "United States" in out, out


def test_stage_filter():
    out = knowledge_base("referee", STAGE)
    lines = out.splitlines()
    assert len(lines) == 5 and all(line.startswith(f"[{STAGE} / ") for line in lines), out


def test_unknown_stage():
    assert knowledge_base("final score", "2026 FIFA World Cup Group Z") == "nothing found"


def test_schema_matches_args():
    call = KBArgs.model_validate_json('{"query": "Mercedes-Benz Stadium attendance"}')
    assert call.stage == "" and KB_SCHEMA["function"]["parameters"]["required"] == ["query"]


def test_schema_rejects_made_up_stage():
    try:
        KBArgs.model_validate_json('{"query": "offside", "stage": "2026 FIFA World Cup Group Stage"}')
    except ValidationError:
        return
    raise AssertionError("выдуманная стадия прошла валидацию")


if __name__ == "__main__":
    load_cache()
    try:
        for test in (test_finds_answer, test_stage_filter, test_unknown_stage, test_schema_matches_args,
                     test_schema_rejects_made_up_stage):
            test()
            print("ok", test.__name__)
    finally:
        save_cache()
