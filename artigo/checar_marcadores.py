#!/usr/bin/env python3
"""Procura no artigo.md os marcadores lexicais de texto de LLM usados na pesquisa do autor
(pesquisa-uso-de-IA/scripts/marker_lexicon_v2.py: núcleo, secundário e exploratório).

Não é um detector: aponta palavras e construções que a pesquisa associou a texto gerado ou
reescrito por LLM, para o autor decidir se troca. Referências e comentários ficam de fora.
Inclui a correção do singular de "crucial"/"primordial" registrada na errata do relatório 07.

Uso: .venv/bin/python artigo/checar_marcadores.py
"""
import importlib.util, re, unicodedata
from pathlib import Path

LEX = Path.home() / "Documents/pesquisa-uso-de-IA/scripts/marker_lexicon_v2.py"
spec = importlib.util.spec_from_file_location("lex", LEX); lex = importlib.util.module_from_spec(spec); spec.loader.exec_module(lex)
MARC = [m for m in lex.PT if m[2] != "placebo"] + [
    ("crucial_singular", lex.W + r"crucial" + lex.E, "nucleo", "errata do relatório 07"),
    ("primordial_singular", lex.W + r"primordial" + lex.E, "nucleo", "errata do relatório 07"),
]

md = (Path(__file__).parent / "artigo.md").read_text(encoding="utf-8")
corpo = md.split("\n---\n", 1)[1].split("# REFERÊNCIAS")[0]
corpo = "\n".join(l for l in corpo.splitlines() if not l.strip().startswith(("<!--", "[[")))
corpo = corpo.split("## ABSTRACT")[0] + corpo.split("**Keywords:**", 1)[-1] if "## ABSTRACT" in corpo else corpo
txt = " ".join(unicodedata.normalize("NFC", corpo).lower().split())
n_pal = len(re.findall(r"\w+", txt))
total = 0
for nome, rx, nivel, _ in MARC:
    achados = [m for m in re.finditer(rx, txt)]
    if not achados: continue
    total += len(achados)
    print(f"[{nivel:>26}] {nome}: {len(achados)}")
    for m in achados[:4]:
        print("      …" + txt[max(0, m.start() - 50):m.end() + 40] + "…")
print(f"\n{total} ocorrências em {n_pal} palavras ({1000 * total / max(n_pal, 1):.2f} por mil); "
      "núcleo: zero é a meta; secundário e exploratório: avaliar caso a caso.")
