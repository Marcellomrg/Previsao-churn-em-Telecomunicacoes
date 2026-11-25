"""Calcula as features derivadas e atualiza somente a tabela final."""

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
BANCO = RAIZ / "data/enem_censo_escolar_2025.db"
CHAVES = ["NU_ANO", "CO_ESCOLA"]
FONTES = [
    "QT_DOC_MED", "QT_DOC_BAS", "QT_MAT_ZR_RUR", "QT_MAT_ZR_URB",
    "QT_TUR_BAS", "QT_TUR_MED", "QT_SALAS_UTILIZADAS", "QT_DESKTOP_ALUNO",
    "QT_COMP_PORTATIL_ALUNO", "IN_INTERNET_APRENDIZAGEM", "IN_DESKTOP_ALUNO",
    "IN_BIBLIOTECA_SALA_LEITURA", "IN_LABORATORIO_INFORMATICA",
    "IN_MATERIAL_PED_CIENTIFICO", "IN_EJA",
    "QT_TUR_PROF_TEC",
    "TDI_MED_03_CAT_0", "TDI_MED_01_CAT_0", "TP_DEPENDENCIA",
    "TP_LOCALIZACAO",
]


def adicionar_features(caminho=BANCO):
    with closing(sqlite3.connect(caminho.resolve().as_uri() + "?mode=rw", uri=True)) as db, db:
        db.execute("PRAGMA temp_store = MEMORY")
        db.execute("BEGIN IMMEDIATE")
        campos = ", ".join(
            f'{"i" if coluna.startswith("TDI_") else "c"}."{coluna}"'
            for coluna in FONTES
        )
        base = pd.read_sql_query(f"""
            SELECT f.NU_ANO, f.CO_ESCOLA, {campos}
            FROM "final" AS f
            LEFT JOIN censo_escolar AS c
                ON c.NU_ANO_CENSO = f.NU_ANO AND c.CO_ENTIDADE = f.CO_ESCOLA
            LEFT JOIN indicadores AS i
                ON i.NU_ANO_CENSO = f.NU_ANO AND i.CO_ENTIDADE = f.CO_ESCOLA
        """, db)
        if base.empty or base[CHAVES].isna().any().any() or base.duplicated(CHAVES).any():
            raise ValueError("A tabela final precisa ter uma linha por escola e ano.")

        x = base[FONTES].apply(pd.to_numeric, errors="coerce")
        x = x.where(x.ge(0) & np.isfinite(x))
        # O dicionário do Censo marca valores extremos com 88888 nestes campos.
        for c in ["QT_DESKTOP_ALUNO", "QT_COMP_PORTATIL_ALUNO"]:
            x[c] = x[c].mask(x[c].eq(88888))
        for c in x.columns[x.columns.str.startswith("IN_")]:
            x[c] = x[c].where(x[c].isin([0, 1]))
        for c in ["TDI_MED_03_CAT_0", "TDI_MED_01_CAT_0"]:
            x[c] = x[c].where(x[c].le(100))

        def dividir(numerador, denominador):
            return numerador / denominador.where(denominador.gt(0))

        turmas = x.QT_TUR_BAS
        dados = base[CHAVES].copy()
        dados["FE_PROP_DOCENTES_MED"] = dividir(x.QT_DOC_MED, x.QT_DOC_BAS)
        dados["FE_PROP_MATRICULAS_RURAIS"] = dividir(x.QT_MAT_ZR_RUR, x.QT_MAT_ZR_RUR + x.QT_MAT_ZR_URB)
        dados["FE_TURMAS_BAS_POR_SALA"] = dividir(turmas, x.QT_SALAS_UTILIZADAS)
        dados["FE_TURMAS_MED_POR_SALA"] = dividir(x.QT_TUR_MED, x.QT_SALAS_UTILIZADAS)
        dados["FE_COMPUTADORES_POR_SALA"] = dividir(x.QT_DESKTOP_ALUNO + x.QT_COMP_PORTATIL_ALUNO, x.QT_SALAS_UTILIZADAS)
        dados["FE_PERFIL_DIGITAL"] = (2 * x.IN_INTERNET_APRENDIZAGEM + x.IN_DESKTOP_ALUNO).map({
            0: "Nenhum dos dois", 1: "Só computador de mesa", 2: "Só internet", 3: "Ambos",
        })
        dados["FE_QT_RECURSOS_PEDAGOGICOS"] = x.IN_BIBLIOTECA_SALA_LEITURA + x.IN_LABORATORIO_INFORMATICA + x.IN_MATERIAL_PED_CIENTIFICO
        dados["FE_LOG_TOTAL_TURMAS"] = np.log1p(turmas)
        tem_tecnico = x.QT_TUR_PROF_TEC.gt(0).where(x.QT_TUR_PROF_TEC.notna())
        dados["FE_PERFIL_OFERTA"] = (2 * x.IN_EJA + tem_tecnico).map({
            0: "Nenhuma das duas", 1: "Só técnica", 2: "Só EJA", 3: "EJA e técnica",
        })

        rede = x.TP_DEPENDENCIA.map({1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"})
        local = x.TP_LOCALIZACAO.map({1: "Urbana", 2: "Rural"})
        dados["FE_REDE_LOCALIZACAO"] = rede.str.cat(local, sep=" / ")

        # Mantém a tabela e seu índice. A criação e a atualização são uma transação.
        existentes = {r[1] for r in db.execute('PRAGMA table_info("final")')}
        for antiga in ("FE_DOCENTES_BAS_POR_TURMA", "FE_DOCENTES_MED_POR_TURMA"):
            if antiga in existentes:
                db.execute(f'ALTER TABLE "final" DROP COLUMN "{antiga}"')
        if "FE_DIF_DISTORCAO_MED_3_1_PP" in existentes:
            db.execute('ALTER TABLE "final" DROP COLUMN "FE_DIF_DISTORCAO_MED_3_1_PP"')
        if "FE_PROP_DOCENTES_BAS" in existentes:
            db.execute('ALTER TABLE "final" DROP COLUMN "FE_PROP_DOCENTES_BAS"')
        colunas = dados.columns[len(CHAVES):]
        db.execute('CREATE TEMP TABLE features (NU_ANO INTEGER, CO_ESCOLA INTEGER, PRIMARY KEY (NU_ANO, CO_ESCOLA))')
        for c in colunas:
            tipo = "REAL" if pd.api.types.is_numeric_dtype(dados[c]) else "TEXT"
            if c not in existentes:
                db.execute(f'ALTER TABLE "final" ADD COLUMN "{c}" {tipo}')
            db.execute(f'ALTER TABLE temp.features ADD COLUMN "{c}" {tipo}')
        linhas = dados.astype(object).where(dados.notna(), None).itertuples(index=False, name=None)
        db.executemany(f'INSERT INTO temp.features VALUES ({", ".join("?" for _ in dados.columns)})', linhas)
        atribuicoes = ", ".join(f'"{c}" = features."{c}"' for c in colunas)
        atualizacao = db.execute(f'''UPDATE "final" SET {atribuicoes} FROM temp.features
            WHERE "final".NU_ANO = features.NU_ANO AND "final".CO_ESCOLA = features.CO_ESCOLA''')
        if atualizacao.rowcount != len(dados):
            raise RuntimeError("O número de escolas atualizadas difere do esperado.")
        print(f"{len(colunas)} features calculadas para {len(dados):,} escolas.")


if __name__ == "__main__":
    adicionar_features()

