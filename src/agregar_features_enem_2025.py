"""Resume as demais informações do ENEM por escola, sem escolher uma moda."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from adicionar_media_geral_enem_2025 import ALVO_GERAL, NOTAS, RESULTADOS


RAIZ = Path(__file__).resolve().parents[1]
BANCO = RAIZ / "data/enem_censo_escolar_2025.db"
CHAVES = ["NU_ANO", "CO_ESCOLA"]
TEXTOS_PROVA = {f"TX_{tipo}_{area}" for tipo in ("RESPOSTAS", "GABARITO")
                for area in ("CN", "CH", "LC", "MT")}
LEITURA = {"sep": ";", "encoding": "latin1", "dtype": {"CO_ESCOLA": "Int64"}}


def resumir_categoria(c, proporcao):
    """Proporções usam registros preenchidos. Cadastro exige consenso por escola."""
    if c.startswith("CO_"):
        return {}
    categoria = proporcao.index.get_level_values(1)
    if c.endswith("_ESC"):
        valores = pd.Series(categoria.to_numpy(), index=proporcao.index)
        grupos = valores.groupby(level=0)
        return {c: grupos.first().where(grupos.size().eq(1))}
    if c in ("NO_MUNICIPIO_PROVA", "SG_UF_PROVA"):
        return {}
    matriz = proporcao.unstack(fill_value=0)
    return {f"{c}_PROP_{int(float(k))}": matriz[k] for k in matriz}

def agregar_features_enem(escolas, arquivo=RESULTADOS):
    originais = pd.read_csv(arquivo, nrows=0, **LEITURA).columns
    campos = [c for c in originais if not c.startswith(("CO_", "NU_NOTA_"))
              and c not in {*CHAVES, *TEXTOS_PROVA, "NO_MUNICIPIO_PROVA", "SG_UF_PROVA"}]
    distintos = [c for c in campos if c.startswith("TX_") or c == "NU_SEQUENCIAL"]
    categorias = [c for c in campos if c not in distintos]
    frequencias = {c: [] for c in categorias}
    pares = {c: [] for c in distintos}
    for bloco in pd.read_csv(arquivo, usecols=CHAVES + campos, chunksize=75_000, **LEITURA):
        bloco = bloco.loc[bloco.NU_ANO.eq(2025) & bloco.CO_ESCOLA.isin(escolas)]
        for c in categorias:
            frequencias[c].append(bloco.groupby(["CO_ESCOLA", c]).size())
        for c in distintos:
            preenchidos = bloco[c].dropna()
            pares[c].append(pd.DataFrame({
                "CO_ESCOLA": bloco.loc[preenchidos.index, "CO_ESCOLA"],
                "valor": pd.util.hash_pandas_object(preenchidos, index=False).to_numpy(),
            }).drop_duplicates())

    agregadas = {}
    for c, partes in frequencias.items():
        contagem = pd.concat(partes).groupby(level=[0, 1]).sum()
        por_escola = contagem.groupby(level=0)
        proporcao = contagem / por_escola.transform("sum")
        agregadas.update(resumir_categoria(c, proporcao))
    for c, partes in pares.items():
        nome = "QT_PARTICIPANTES" if c == "NU_SEQUENCIAL" else f"{c}_N_DISTINTOS"
        agregadas[nome] = pd.concat(partes, ignore_index=True).drop_duplicates().groupby("CO_ESCOLA").size()
    return pd.DataFrame(agregadas).reindex(escolas)


def atualizar_features_enem(caminho=BANCO, usar_resumos_existentes=False):
    uri = caminho.resolve().as_uri()
    with closing(sqlite3.connect(uri + "?mode=ro", uri=True)) as db:
        escolas = pd.read_sql_query(f"SELECT CO_ESCOLA, {ALVO_GERAL} FROM final ORDER BY CO_ESCOLA", db)
        if usar_resumos_existentes:
            campos = [r[1] for r in db.execute("PRAGMA table_info(final)") if r[1].endswith("_DISTRIBUICAO")]
            if not campos:
                raise ValueError("Não há distribuições anteriores para converter.")
            anterior = pd.read_sql_query("SELECT CO_ESCOLA, " + ", ".join(campos) + " FROM final", db)
            novas = {}
            for campo in campos:
                c = campo.removesuffix("_DISTRIBUICAO")
                linhas = [(escola, k, v) for escola, bruto in anterior[["CO_ESCOLA", campo]].dropna().itertuples(index=False, name=None)
                          for k, v in json.loads(bruto).items()]
                frequencia = pd.DataFrame(linhas, columns=["CO_ESCOLA", c, "proporcao"])
                if not c.startswith(("NO_", "SG_")):
                    frequencia[c] = pd.to_numeric(frequencia[c])
                novas.update(resumir_categoria(c, frequencia.set_index(["CO_ESCOLA", c]).proporcao))
            resumo = pd.DataFrame(novas).reindex(escolas.CO_ESCOLA)
        else:
            resumo = agregar_features_enem(escolas.CO_ESCOLA)
    resumo = resumo.reset_index(names="CO_ESCOLA")
    colunas = resumo.columns[1:]
    with closing(sqlite3.connect(uri + "?mode=rw", uri=True)) as db, db:
        db.execute("PRAGMA temp_store = MEMORY")
        db.execute("BEGIN IMMEDIATE")
        assert db.execute(f"SELECT CO_ESCOLA, {ALVO_GERAL} FROM final ORDER BY CO_ESCOLA").fetchall() == list(
            escolas.itertuples(index=False, name=None)
        )
        tipos = {c: "TEXT" if c.startswith(("NO_", "SG_")) else
                 "INTEGER" if c.endswith("_N_DISTINTOS") or c == "QT_PARTICIPANTES" else "REAL" for c in colunas}
        definicoes = ", ".join(f'"{c}" {tipo}' for c, tipo in tipos.items())
        db.execute(f"CREATE TEMP TABLE resumo_enem (CO_ESCOLA INTEGER PRIMARY KEY, {definicoes})")
        marcadores = ", ".join("?" for _ in resumo.columns)
        valores = resumo.astype(object).where(resumo.notna(), None).itertuples(index=False, name=None)
        db.executemany(f"INSERT INTO resumo_enem VALUES ({marcadores})", valores)
        atribuicoes = ", ".join(f'"{c}" = resumo_enem."{c}"' for c in colunas)
        existentes = {r[1] for r in db.execute(f'PRAGMA table_info("final")')}
        obsoletas = {f"{c}_N_DISTINTOS" for c in TEXTOS_PROVA}
        for antiga in sorted(c for c in existentes if c in obsoletas or c.endswith("_DISTRIBUICAO")):
            db.execute(f'ALTER TABLE "final" DROP COLUMN "{antiga}"')
        for c, tipo in tipos.items():
            if c not in existentes:
                db.execute(f'ALTER TABLE "final" ADD COLUMN "{c}" {tipo}')
        alteradas = db.execute(f'''
            UPDATE "final" SET {atribuicoes} FROM resumo_enem
            WHERE "final".CO_ESCOLA = resumo_enem.CO_ESCOLA AND "final".NU_ANO = 2025
        ''').rowcount
        if alteradas != len(escolas):
            raise RuntimeError(f"Número inesperado de escolas atualizadas em final: {alteradas}.")
    print(f"{len(colunas)} campos do ENEM agregados em {len(escolas):,} escolas.")


if __name__ == "__main__":
    atualizar_features_enem()


