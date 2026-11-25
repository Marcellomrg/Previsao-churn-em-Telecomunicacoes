"""Reúne as planilhas escolares do Inep e acrescenta os indicadores ao SQLite."""

import sqlite3
from contextlib import closing
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook


def importar_planilha(banco, pasta, sigla):
    arquivo = pasta / f"{sigla}_2025_ESCOLAS.zip"
    with ZipFile(arquivo) as zipado:
        membro, = [nome for nome in zipado.namelist() if nome.lower().endswith(".xlsx")]
        planilha = load_workbook(BytesIO(zipado.read(membro)), read_only=True, data_only=True)
        try:
            linhas = iter(planilha.active.values)
            cabecalho = next(linha for linha in linhas if linha[0] == "NU_ANO_CENSO")
            indice_escola = cabecalho.index("CO_ENTIDADE")
            indice_ano = cabecalho.index("NU_ANO_CENSO")
            medidas = [(i, f"{sigla}_{nome}") for i, nome in enumerate(cabecalho)
                       if isinstance(nome, str) and "_CAT_" in nome]
            colunas = [nome for _, nome in medidas]
            definicoes = ", ".join(f'"{nome}" REAL' for nome in colunas)
            banco.execute(f"""
                CREATE TEMP TABLE {sigla} (
                    NU_ANO_CENSO INTEGER NOT NULL,
                    CO_ENTIDADE INTEGER NOT NULL,
                    {definicoes},
                    PRIMARY KEY (NU_ANO_CENSO, CO_ENTIDADE)
                )
            """)
            parametros = ", ".join("?" for _ in range(len(colunas) + 2))
            lote = []
            for linha in linhas:
                if linha[indice_ano] != 2025:
                    continue
                escola = int(linha[indice_escola])
                valores = []
                for indice, nome in medidas:
                    valor = linha[indice]
                    if valor in (None, "", "--", "-"):
                        valores.append(None)
                    elif isinstance(valor, (int, float)) and valor >= 0 and (sigla == "ATU" or valor <= 100):
                        valores.append(valor)
                    else:
                        raise ValueError(f"{arquivo.name}: valor inesperado em {nome} para {escola}: {valor!r}")
                lote.append((2025, escola, *valores))
                if len(lote) == 5000:
                    banco.executemany(f"INSERT INTO {sigla} VALUES ({parametros})", lote)
                    lote.clear()
            banco.executemany(f"INSERT INTO {sigla} VALUES ({parametros})", lote)
            total = banco.execute(f"SELECT COUNT(*) FROM {sigla}").fetchone()[0]
            if not total or not colunas:
                raise ValueError(f"{arquivo.name}: planilha sem dados escolares de 2025")
            print(f"{sigla}: {total} escolas e {len(colunas)} medidas", flush=True)
            return colunas
        finally:
            planilha.close()


def adicionar_indicadores(arquivo_banco, pasta_indicadores):
    with closing(sqlite3.connect(arquivo_banco.resolve().as_uri() + "?mode=rw", uri=True)) as banco:
        banco.execute("PRAGMA temp_store = MEMORY")
        banco.execute("BEGIN IMMEDIATE")
        with banco:
            siglas = ("AFD", "ATU", "TDI")
            medidas = {sigla: importar_planilha(banco, pasta_indicadores, sigla) for sigla in siglas}
            campos = ", ".join(f'{sigla}."{nome}"' for sigla, nomes in medidas.items() for nome in nomes)
            chaves = " UNION ".join(f"SELECT NU_ANO_CENSO, CO_ENTIDADE FROM {sigla}" for sigla in siglas)
            juncoes = "\n".join(f"LEFT JOIN {sigla} USING (NU_ANO_CENSO, CO_ENTIDADE)" for sigla in siglas)
            banco.execute("DROP TABLE IF EXISTS indicadores")
            banco.execute(f"""
                CREATE TABLE indicadores AS
                WITH escolas AS ({chaves})
                SELECT escolas.NU_ANO_CENSO, escolas.CO_ENTIDADE, {campos}
                FROM escolas
                {juncoes}
            """)
            banco.execute("CREATE UNIQUE INDEX indicadores_chave ON indicadores (NU_ANO_CENSO, CO_ENTIDADE)")

            existentes = {coluna[1] for coluna in banco.execute('PRAGMA table_info("final")')}
            total_antes = banco.execute('SELECT COUNT(*) FROM "final"').fetchone()[0]
            if not existentes or not total_antes:
                raise ValueError("A tabela final precisa conter a junção do ENEM com o Censo.")
            colunas = [nome for nomes in medidas.values() for nome in nomes]
            if "FE_COMPUTADORES_POR_SALA" in existentes:
                # A tabela pronta já teve as colunas usadas nas derivações removidas.
                colunas = [nome for nome in colunas if nome in existentes]
            for nome in colunas:
                if nome not in existentes:
                    banco.execute(f'ALTER TABLE "final" ADD COLUMN "{nome}" REAL')
            lista = ", ".join(f'"{nome}"' for nome in colunas)
            banco.execute(f"""
                UPDATE "final" AS base
                SET ({lista}) = (
                    SELECT {lista}
                    FROM indicadores
                    WHERE indicadores.CO_ENTIDADE = base.CO_ESCOLA
                      AND indicadores.NU_ANO_CENSO = base.NU_ANO
                )
            """)
            cobertura = banco.execute("""
                SELECT COUNT(*), COUNT(DISTINCT CO_ESCOLA),
                       COUNT(AFD_MED_CAT_1), COUNT(ATU_MED_CAT_0), COUNT(TDI_MED_CAT_0),
                       SUM(AFD_MED_CAT_1 IS NOT NULL AND ATU_MED_CAT_0 IS NOT NULL
                           AND TDI_MED_CAT_0 IS NOT NULL)
                FROM "final"
            """).fetchone()
            if cobertura[:2] != (total_antes, total_antes):
                raise ValueError("A atualização alterou a quantidade ou a unicidade de escolas.")
            print("Final (linhas, escolas, AFD EM, ATU EM, TDI EM, três preenchidos):", cobertura)
            print("Indicadores (linhas, colunas):",
                  banco.execute("SELECT COUNT(*) FROM indicadores").fetchone()[0], len(colunas) + 2)


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parents[1]
    adicionar_indicadores(
        raiz / "data/enem_censo_escolar_2025.db",
        raiz / "data/indicadores_educacionais/2025",
    )
