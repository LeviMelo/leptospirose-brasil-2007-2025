# -*- coding: utf-8 -*-
"""Confere o documento montado contra os limites do modelo da RESS.

O modelo em ``docs/literature/journals/RESS_submission_rules.txt`` impoe cinco
limites que so se descobre ter estourado depois que o texto ja cresceu:

===========================  ======================================
resumo                       250 palavras
introducao ate a discussao   3.500 palavras (artigo original)
referencias                  30 (artigo original)
ilustracoes                  5 (artigo original)
palavras-chave               5 descritores
===========================  ======================================

A contagem e feita sobre o ``.docx`` construido, e nao sobre o codigo que o
constroi, porque e o arquivo que a revista recebe. Titulos de secao, legendas
de ilustracao, tabelas e material final ficam fora da contagem do corpo: o
limite e "da introducao ate o final da discussao".

Uso:
    python conferir_limites.py [arquivo.docx] [-v]

Sai com codigo 1 se algum limite for excedido, para poder entrar num gancho de
verificacao antes da submissao.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import docx

AQUI = Path(__file__).resolve().parent
PADRAO = AQUI / "RESS-2026-1462_v2_ajustado.docx"

LIMITE_RESUMO = 250
LIMITE_CORPO = 3500
LIMITE_REFERENCIAS = 30
LIMITE_ILUSTRACOES = 5
DESCRITORES = 5

INICIO_CORPO = "Introdução"
FIM_CORPO = ("Conflitos de interesse", "Disponibilidade de dados")
INICIO_RESUMO = "Resumo"
FIM_RESUMO = ("Palavras-chave",)

# Titulos de secao e de subsecao nao sao texto corrido e nao entram na contagem.
TITULOS = {
    "Métodos", "Resultados", "Discussão", "Delineamento e período",
    "Cenário e fontes de dados", "Variáveis", "Vieses", "Tamanho do estudo",
    "Métodos estatísticos",
}


def palavras(texto: str) -> int:
    return len([t for t in re.split(r"\s+", texto.strip()) if t])


def _faixa(paragrafos: list[str], inicio: str, fins: tuple[str, ...]) -> list[str]:
    dentro, acc = False, []
    for t in paragrafos:
        if not t:
            continue
        if t == inicio:
            dentro = True
            continue
        if dentro and any(t.startswith(f) for f in fins):
            break
        if dentro:
            acc.append(t)
    return acc


def main(caminho: Path, detalhado: bool = False) -> int:
    d = docx.Document(str(caminho))
    paras = [p.text.strip() for p in d.paragraphs]

    corpo = [t for t in _faixa(paras, INICIO_CORPO, FIM_CORPO) if t not in TITULOS]
    resumo = _faixa(paras, INICIO_RESUMO, FIM_RESUMO)

    n_corpo = sum(palavras(t) for t in corpo)
    n_resumo = sum(palavras(t) for t in resumo)
    n_refs = sum(1 for t in paras if re.match(r"^\d+\.\s", t))
    n_tabelas = len(d.tables)
    n_figuras = sum(1 for r in d.part.rels.values() if "image" in r.reltype)
    chave = next((t for t in paras if t.startswith("Palavras-chave")), "")
    n_chave = len([x for x in chave.split(":", 1)[-1].split(";") if x.strip()])

    linhas = [
        ("resumo", n_resumo, LIMITE_RESUMO),
        ("introducao ate discussao", n_corpo, LIMITE_CORPO),
        ("referencias", n_refs, LIMITE_REFERENCIAS),
        ("ilustracoes", n_tabelas + n_figuras, LIMITE_ILUSTRACOES),
        ("palavras-chave", n_chave, DESCRITORES),
    ]

    print(caminho.name)
    excedeu = False
    for nome, valor, limite in linhas:
        if nome == "palavras-chave":
            ok = valor == limite
            marca = "OK" if ok else "SAO %d, DEVEM SER %d" % (valor, limite)
        else:
            ok = valor <= limite
            marca = "OK" if ok else "EXCEDE em %d" % (valor - limite)
        excedeu |= not ok
        print("  %-26s %5d  (limite %4d)  %s" % (nome, valor, limite, marca))

    if detalhado:
        print("\n  corpo por secao:")
        secao, acc = INICIO_CORPO, {}
        for t in _faixa(paras, INICIO_CORPO, FIM_CORPO):
            if t in TITULOS:
                secao = t
                continue
            acc[secao] = acc.get(secao, 0) + palavras(t)
        for k, v in acc.items():
            print("    %-28s %5d" % (k, v))

    return 1 if excedeu else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "-v"]
    alvo = Path(args[0]) if args else PADRAO
    raise SystemExit(main(alvo, "-v" in sys.argv))
