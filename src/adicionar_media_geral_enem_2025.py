"""Calcula a média simples por participante e a mediana por escola."""

from pathlib import Path

import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "data/enem/microdados/2025/RESULTADOS_2025.csv"
ALVO_GERAL = "NU_NOTA_GERAL_MEDIANA"
CONTAGEM_GERAL = "NU_NOTA_GERAL_N_VALIDAS"
NOTAS = [f"NU_NOTA_{area}" for area in ("MT", "CN", "CH", "LC", "REDACAO")]
PRESENCAS = [f"TP_PRESENCA_{area}" for area in ("MT", "CN", "CH", "LC")]


def agregar_mediana_por_escola(arquivo=RESULTADOS):
    partes = []
    for bloco in pd.read_csv(
        arquivo, sep=";", encoding="latin1", chunksize=150_000,
        usecols=["NU_ANO", "CO_ESCOLA", *NOTAS, *PRESENCAS],
        dtype={"CO_ESCOLA": "Int64"},
    ):
        validos = (bloco.NU_ANO.eq(2025) & bloco.CO_ESCOLA.notna()
                   & bloco[PRESENCAS].eq(1).all(axis=1) & bloco[NOTAS].notna().all(axis=1))
        dados = bloco.loc[validos, ["CO_ESCOLA", *NOTAS]]
        medias = dados[NOTAS].mean(axis=1, skipna=False)
        partes.append(pd.DataFrame({"CO_ESCOLA": dados.CO_ESCOLA, "media_individual": medias}))
    # Todas as médias individuais entram na mediana, inclusive entre blocos do CSV.
    escolas = pd.concat(partes, ignore_index=True).groupby("CO_ESCOLA").media_individual.agg(
        **{ALVO_GERAL: "median", CONTAGEM_GERAL: "size"}
    )
    return escolas.loc[escolas[CONTAGEM_GERAL].ge(50)].reset_index().assign(NU_ANO=2025)[
        ["NU_ANO", "CO_ESCOLA", ALVO_GERAL]
    ]
