"""Materializa o banco SQLite de 2025 a partir dos arquivos locais."""

import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from adicionar_features_escolares_2025 import FONTES, adicionar_features
from adicionar_indicadores_escolares_2025 import adicionar_indicadores
from adicionar_media_geral_enem_2025 import agregar_mediana_por_escola
from agregar_features_enem_2025 import agregar_features_enem
from importar_enem_original_2025 import importar_enem_original


def montar_banco():
    raiz = Path(__file__).resolve().parents[1]
    destino = raiz / "data/enem_censo_escolar_2025.db"
    pasta_censo = raiz / "data/censo_escolar/2025/dados"
    arquivos = {
        "Escola": "Tabela_Escola_2025_V2.csv",
        "Matricula": "Tabela_Matricula_2025_V2.csv",
        "Docente": "Tabela_Docente_2025_V2.csv",
        "Turma": "Tabela_Turma_2025_V2.csv",
    }
    features = (raiz / "output/selecao_censo_2025/features_100.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    mapa = pd.read_csv(raiz / "artifacts/censo_eda_comparacao/mapa_colunas_dicionarios.csv")
    assert len(features) == len(set(features)) == 100
    features = list(dict.fromkeys([*features, "QT_TUR_MED", "TP_DEPENDENCIA", "TP_LOCALIZACAO"]))
    origens = mapa.set_index("coluna").loc[features, "origem_2025"].to_dict()
    assert set(origens.values()) <= arquivos.keys()
    enem = agregar_mediana_por_escola()
    enem = enem.join(agregar_features_enem(enem.CO_ESCOLA), on="CO_ESCOLA")
    assert enem["NU_ANO"].eq(2025).all()
    assert enem["CO_ESCOLA"].notna().all() and enem["CO_ESCOLA"].is_unique


    with TemporaryDirectory(dir=destino.parent, prefix="montagem_") as pasta_temp:
        provisório = Path(pasta_temp) / destino.name
        with closing(sqlite3.connect(provisório.resolve().as_uri(), uri=True)) as banco, banco:
            banco.execute("PRAGMA temp_store = MEMORY")
            enem.to_sql("resumo_enem_escolar", banco, index=False)
            importar_enem_original(banco)

            # A base auxiliar fica em disco e é descartada após a montagem.
            banco.execute("ATTACH DATABASE ? AS origem", (str(Path(pasta_temp) / "origem.db"),))
            with closing(sqlite3.connect(Path(pasta_temp) / "origem.db")) as origem:
                origem.execute("PRAGMA temp_store = MEMORY")
                for tabela, arquivo in arquivos.items():
                    colunas = ["NU_ANO_CENSO", "CO_ENTIDADE"]
                    colunas.extend(c for c in features if origens[c] == tabela)
                    for bloco in pd.read_csv(
                        pasta_censo / arquivo, sep=";", encoding="latin1", chunksize=15_000,
                        usecols=colunas, low_memory=False,
                        dtype={"CO_ENTIDADE": "int64", "NU_ANO_CENSO": "int64"},
                    ):
                        assert bloco["NU_ANO_CENSO"].eq(2025).all()
                        bloco.to_sql(tabela, origem, if_exists="append", index=False)
                    origem.execute(
                        f"CREATE UNIQUE INDEX chave_{tabela} ON {tabela} (NU_ANO_CENSO, CO_ENTIDADE)"
                    )
                    print(f"Censo importado: {tabela}", flush=True)
                origem.commit()

            campos = ", ".join(f'{origens[c]}."{c}"' for c in features)
            sql_censo = (raiz / "sql/05_materializar_censo_selecionado_2025.sql").read_text(encoding="utf-8")
            banco.executescript(sql_censo.replace("/* CAMPOS_SELECIONADOS */", campos))
            sql_final = (raiz / "sql/06_materializar_final_2025.sql").read_text(encoding="utf-8")
            banco.executescript(sql_final)
            banco.execute("DROP TABLE resumo_enem_escolar")

            for nome in ("enem", "censo_escolar", "final"):
                linhas = banco.execute(f'SELECT COUNT(*) FROM "{nome}"').fetchone()[0]
                colunas = len(banco.execute(f'PRAGMA table_info("{nome}")').fetchall())
                print(f"{nome}: {linhas} linhas, {colunas} colunas", flush=True)

            total, encontrados = banco.execute(
                'SELECT COUNT(*), SUM(escola_encontrada_no_censo) FROM "final"'
            ).fetchone()
            assert total == len(enem)
            assert [r[1] for r in banco.execute("PRAGMA table_info(censo_escolar)")] == [
                "NU_ANO_CENSO", "CO_ENTIDADE", *features
            ]
            assert banco.execute("PRAGMA main.quick_check").fetchone() == ("ok",)
            print(f"Correspondências no Censo: {encontrados} de {total}", flush=True)
        adicionar_indicadores(provisório, raiz / "data/indicadores_educacionais/2025")
        adicionar_features(provisório)
        with closing(sqlite3.connect(provisório)) as banco, banco:
            banco.execute("PRAGMA temp_store = MEMORY")
            banco.execute("BEGIN IMMEDIATE")
            excluir = {*FONTES, "CO_ENTIDADE", "NU_ANO_CENSO", "escola_encontrada_no_censo"}
            manter = [r[1] for r in banco.execute('PRAGMA table_info("final")') if r[1] not in excluir]
            campos_finais = ", ".join(f'"{c}"' for c in manter)
            banco.execute(f'CREATE TABLE final_limpa AS SELECT {campos_finais} FROM "final"')
            banco.execute('DROP TABLE "final"')
            banco.execute('ALTER TABLE final_limpa RENAME TO "final"')
            banco.execute('CREATE UNIQUE INDEX final_chave ON "final" (NU_ANO, CO_ESCOLA)')
            assert banco.execute("PRAGMA quick_check").fetchone() == ("ok",)
            assert banco.execute('SELECT COUNT(*) FROM "final"').fetchone() == (len(enem),)
            assert banco.execute("SELECT COUNT(*) FROM indicadores").fetchone()[0] > 0
            colunas = {r[1] for r in banco.execute('PRAGMA table_info("final")')}
            assert not colunas.intersection(FONTES)
            assert not any(c.endswith("_MODA") for c in colunas)
            assert {c for c in colunas if c.startswith("NU_NOTA_GERAL_")} == {
                "NU_NOTA_GERAL_MEDIANA",
            }
            print(f"Final: {len(enem):,} escolas e {len(colunas)} colunas.")
        provisório.replace(destino)
    print(destino)


if __name__ == "__main__":
    montar_banco()

