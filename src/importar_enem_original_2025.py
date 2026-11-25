"""Importa as linhas originais de RESULTADOS_2025 sem filtros por escola."""
import csv
import sqlite3
from contextlib import closing
from pathlib import Path

from adicionar_media_geral_enem_2025 import RAIZ, RESULTADOS


def importar_enem_original(db, arquivo=RESULTADOS):
    with Path(arquivo).open(encoding="latin1", newline="") as entrada:
        leitor = csv.reader(entrada, delimiter=";")
        campos = next(leitor)
        tipos = []
        for c in campos:
            tipo = "TEXT" if c.startswith(("NO_", "SG_", "TX_")) else "REAL" if c.startswith("NU_NOTA_") else "INTEGER"
            tipos.append(f'"{c}" {tipo}')
        db.execute("CREATE TABLE enem_original_importacao (" + ", ".join(tipos) + ")")
        insercao = "INSERT INTO enem_original_importacao VALUES (" + ",".join("?" for _ in campos) + ")"
        linhas = (tuple(v if v != "" else None for v in linha) for linha in leitor)
        total = db.executemany(insercao, linhas).rowcount
    validas = db.execute("SELECT count(*) FROM enem_original_importacao WHERE NU_ANO=2025").fetchone()[0]
    if not total or validas != total:
        raise ValueError("A contagem de registros de 2025 diverge da importação.")
    db.execute("DROP TABLE IF EXISTS enem")
    db.execute("ALTER TABLE enem_original_importacao RENAME TO enem")
    print(f"ENEM original: {total} registros e {len(campos)} colunas.", flush=True)


if __name__ == "__main__":
    banco = RAIZ / "data/enem_censo_escolar_2025.db"
    with closing(sqlite3.connect(banco.as_uri() + "?mode=rw", uri=True)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        importar_enem_original(db)
