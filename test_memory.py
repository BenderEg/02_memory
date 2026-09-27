"""Запись и поиск фактов без LLM и без API эмбеддингов: пустая память, найденный факт, заменённый факт, забытый факт.

Вместо эмбеддингов хэш символьных триграмм, коллекция facts_test. Нужен поднятый Milvus. Запуск: python test_memory.py
"""
import re, hashlib

import numpy as np

from load_index import DIM, ledger, milvus
from memory import FactMemory


def fake_embed(texts):
    vectors = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in zip(vectors, texts):
        text = " " + " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split()) + " "
        for i in range(len(text) - 2):
            row[int(hashlib.md5(text[i:i + 3].encode()).hexdigest(), 16) % DIM] += 1
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def fresh_memory():
    mem = FactMemory("facts_test", embed=fake_embed)
    mem.reset()
    return mem


def test_empty_memory():
    mem = fresh_memory()
    assert mem.recall("где я живу?") == [] and mem.all() == {}


def test_finds_fact():
    mem = fresh_memory()
    mem.upsert({"name": "Пользователя зовут Женя", "city": "Живёт в Москве", "favorite_team": "Болеет за сборную Японии"})
    assert mem.recall("За какую сборную я болею?", k=1) == ["Болеет за сборную Японии"], mem.recall("За какую сборную я болею?")


def test_replaced_fact():
    mem = fresh_memory()
    mem.upsert({"name": "Пользователя зовут Женя", "favorite_team": "Болеет за сборную Японии"})
    mem.upsert({"favorite_team": "Болеет за сборную Марокко"})
    found = mem.recall("За какую сборную я болею?", k=3)
    assert mem.all() == {"name": "Пользователя зовут Женя", "favorite_team": "Болеет за сборную Марокко"}, mem.all()
    assert "Болеет за сборную Марокко" in found and "Болеет за сборную Японии" not in found, found


def test_forgotten_fact():
    mem = fresh_memory()
    mem.upsert({"name": "Пользователя зовут Женя", "favorite_team": "Болеет за сборную Японии"})
    mem.sync({"name": "Пользователя зовут Женя"})  # полный новый список без сборной
    assert mem.all() == {"name": "Пользователя зовут Женя"} and mem.recall("сборная", k=3) == ["Пользователя зовут Женя"]


if __name__ == "__main__":
    try:
        for test in (test_empty_memory, test_finds_fact, test_replaced_fact, test_forgotten_fact):
            test()
            print("ok", test.__name__)
    finally:
        milvus().drop_collection("facts_test")
    assert not ledger.calls, ledger.calls
    print("вызовов модели и эмбеддингов: 0")
