"""Tokenisation for lexical search over API docs (camelCase / snake_case / paths aware)."""

from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

_stemmer = snowballstemmer.stemmer("english")

STOPWORDS = frozenset("""
a an the and or of to for in on at by with from into about as is are was were be been being it its this that these those
i me my we our you your he she they them their what which who whom how when where why do does did done can could should
would will shall may might must have has had not no yes please just all any some also than then there here up out so if
over under again more most such only own same too very s t via per using use used e g eg
""".split())

_split_camel = re.compile(r"([a-z0-9])([A-Z])")
_non_alnum = re.compile(r"[^a-z0-9]+")


@lru_cache(maxsize=50_000)
def _stem(word: str) -> str:
    return _stemmer.stemWord(word)


def tokenize(text: str) -> list[str]:
    text = _split_camel.sub(r"\1 \2", text or "").lower()
    out = []
    for w in _non_alnum.split(text):
        if not w or w in STOPWORDS or len(w) == 1:
            continue
        out.append(_stem(w))
    return out
