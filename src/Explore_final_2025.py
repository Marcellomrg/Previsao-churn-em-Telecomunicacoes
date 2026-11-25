# %% [markdown]
# # ENEM 2025 por escola — análise exploratória da tabela final
# Recorte: escolas com pelo menos 50 participantes com as cinco notas válidas.
# Execute na ordem com o Jupyter do VS Code. Os gráficos também são gravados em
# output/eda_mediana_geral_2025. O alvo é a mediana da média geral do ENEM (NU_NOTA_GERAL_MEDIANA).

# %% Carregar o banco e identificar a origem das colunas
import sqlite3
from contextlib import closing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

pd.set_option("display.max_columns", None)
raiz = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
caminho_banco = Path("data/enem_censo_escolar_2025.db")
if not (raiz / caminho_banco).is_file():
    raiz = raiz.parent
pasta_graficos = raiz / "output/eda_mediana_geral_2025"
pasta_graficos.mkdir(parents=True, exist_ok=True)
with closing(sqlite3.connect((raiz / caminho_banco).as_uri() + "?mode=ro", uri=True)) as db:
    df = pd.read_sql_query('SELECT * FROM "final" ORDER BY NU_ANO, CO_ESCOLA', db)
    colunas_censo = [r[1] for r in db.execute("PRAGMA table_info(censo_escolar)")]
    colunas_indicadores = [r[1] for r in db.execute("PRAGMA table_info(indicadores)")]

colunas_enem = [c for c in df if c not in colunas_censo and c not in colunas_indicadores and not c.startswith("FE_")]
alvo = "NU_NOTA_GERAL_MEDIANA"
chaves = ["NU_ANO", "CO_ESCOLA"]
derivadas = df.columns[df.columns.str.startswith("FE_")].tolist()
print(f"{len(df):,} escolas | {df.shape[1]} colunas | anos: {df.NU_ANO.unique().tolist()}")
display(df.head())


def mostrar_grafico(fig, arquivo):
    fig.tight_layout()
    fig.savefig(pasta_graficos / f"{arquivo}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# %% Duplicados e chaves da escola
duplicadas = df.duplicated()
chaves_duplicadas = df.duplicated(chaves, keep=False)
chaves_nulas = df[chaves].isna().any(axis=1)
qualidade = pd.Series({
    "Linhas idênticas excedentes": duplicadas.sum(),
    "Linhas com chave repetida": chaves_duplicadas.sum(),
    "Linhas sem chave completa": chaves_nulas.sum(),
})
display(qualidade.to_frame("linhas"))
display(df.loc[chaves_duplicadas | chaves_nulas, chaves + [alvo]].head(20))
fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(qualidade.index, qualidade, color="#327a8a")
ax.bar_label(ax.containers[0], padding=5)
ax.set(xlim=(0, max(int(qualidade.max()), 1) * 1.25), xlabel="Linhas", title="Qualidade da chave escola e ano")
mostrar_grafico(fig, "01_duplicados_e_chaves")

# %% Nulos e códigos especiais do Censo
# A cópia de trabalho marca como nulo o código 88888 só nas colunas em que
# o dicionário do Censo documenta esse código. O banco é aberto apenas para leitura.
mapa = pd.read_csv(raiz / "artifacts/censo_eda_comparacao/mapa_colunas_dicionarios.csv")
marcadas = mapa.loc[mapa.categorias_2025.str.contains(r"\b88888\b", na=False), "coluna"]
com_codigo = sorted(set(marcadas) & set(colunas_censo) & set(df))
eda = df.copy()
eda[com_codigo] = eda[com_codigo].mask(eda[com_codigo].eq(88888))
nulos = pd.DataFrame({"nulos_sql": df.isna().sum(), "nulos_eda": eda.isna().sum()})
nulos["codigos_88888"] = nulos.nulos_eda - nulos.nulos_sql
nulos["pct_nulos_eda"] = 100 * nulos.nulos_eda / len(eda)
nulos = nulos.sort_values("pct_nulos_eda", ascending=False)
display(nulos)
print(f"Colunas com nulos na análise: {nulos.nulos_eda.gt(0).sum()}")
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
nulos.pct_nulos_eda.head(18).sort_values().plot.barh(ax=axes[0], color="#327a8a")
nulos.loc[derivadas, "pct_nulos_eda"].sort_values().plot.barh(ax=axes[1], color="#c16942")
for ax, titulo in zip(axes, ["Colunas com mais nulos", "Nulos nas features novas"]):
    ax.set(xlabel="Escolas (%)", xlim=(0, 100), title=titulo)
mostrar_grafico(fig, "02_nulos")

# %% Tipos semânticos, cardinalidade e categorias
# Códigos numéricos e flags entram como categorias, inclusive os 0/1 do Censo.
ids = [c for c in eda if c in [*chaves, "NU_ANO_CENSO", "CO_ENTIDADE"]]
categoricas = [c for c in eda if c not in ids and (
    c.startswith(("CO_", "TP_", "IN_", "NO_", "SG_")) or c == "escola_encontrada_no_censo"
    or not pd.api.types.is_numeric_dtype(eda[c])
)]
numericas = [c for c in eda if c not in ids + categoricas]
tipos = pd.Series({"Identificadores e ano": len(ids), "Categóricas": len(categoricas), "Numéricas": len(numericas)})
cardinalidade = eda[categoricas].nunique().sort_values(ascending=False).rename("categorias_sem_nulos")
display(cardinalidade.to_frame().join(nulos[["pct_nulos_eda"]]))
frequencias = pd.concat({c: eda[c].value_counts(dropna=False) for c in categoricas}, names=["variavel", "categoria"])
display(frequencias.rename("escolas").to_frame())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
tipos.plot.barh(ax=axes[0], color="#327a8a", title="Tipo de variável")
eixos_cardinalidade = ["0–1", "2", "3–10", "11–50", "51+"]
pd.cut(cardinalidade, [-1, 1, 2, 10, 50, np.inf], labels=eixos_cardinalidade).value_counts(sort=False).plot.bar(
    ax=axes[1], rot=0, color="#c16942", title="Cardinalidade das categóricas")
for ax in axes:
    ax.bar_label(ax.containers[0], padding=3)
mostrar_grafico(fig, "03_tipos_e_cardinalidade")

# %% Categorias das features novas e relação com o alvo
categorias_novas = [c for c in derivadas if c in categoricas]
fig, axes = plt.subplots(len(categorias_novas), 2, figsize=(14, 10), squeeze=False)
for c, (ax_contagem, ax_media) in zip(categorias_novas, axes):
    grupos = eda[c].fillna("Sem informação")
    resumo = eda[alvo].groupby(grupos).agg(escolas="size", media="mean").sort_values("escolas")
    display(resumo)
    resumo.escolas.plot.barh(ax=ax_contagem, color="#327a8a", title=c)
    ax_contagem.set(xlabel="Escolas", ylabel="")
    ax_media.scatter(resumo.media, range(len(resumo)), color="#c16942")
    ax_media.axvline(eda[alvo].mean(), color="gray", linestyle="--", label="Média do alvo em todas as escolas")
    ax_media.set(yticks=range(len(resumo)), yticklabels=resumo.index, xlabel="Média do alvo no grupo (pontos)")
    ax_media.legend()
mostrar_grafico(fig, "04_categorias_features_e_nota")

# %% Resumo das numéricas e distribuição do alvo
display(eda[numericas].describe(percentiles=[.01, .25, .50, .75, .99]).T)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(eda[alvo].dropna(), bins=38, color="#327a8a")
axes[0].axvline(eda[alvo].median(), color="#ab482a", linestyle="--", label="Mediana")
axes[0].set(xlabel="Mediana da média geral do ENEM (pontos)", ylabel="Escolas", title="Distribuição do alvo")
axes[0].legend()
axes[1].boxplot(eda[alvo].dropna(), vert=False)
axes[1].set(xlabel="Mediana da média geral do ENEM (pontos)", title="Dispersão das medianas por escola", yticks=[])
mostrar_grafico(fig, "05_distribuicao_do_alvo")

# %% Outliers por IQR e distribuições das features numéricas novas
# As marcações ajudam a revisar os dados. Limites IQR não são aplicados a
# campos constantes ou com IQR zero, e nenhuma escola é removida.
v = eda[numericas]
q1, q3 = v.quantile(.25), v.quantile(.75)
iqr = q3 - q1
inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
validas_iqr = iqr.gt(0)
outliers = (v.lt(inferior) | v.gt(superior)).sum().where(validas_iqr)
relatorio_iqr = pd.DataFrame({
    "observadas": v.count(), "IQR": iqr, "limite_inferior": inferior.where(validas_iqr),
    "limite_superior": superior.where(validas_iqr), "fora_dos_limites": outliers,
    "pct_fora_dos_limites": 100 * outliers / v.count().replace(0, np.nan),
}).sort_values("pct_fora_dos_limites", ascending=False)
display(relatorio_iqr)
display(relatorio_iqr.loc[[alvo, *[c for c in derivadas if c in numericas]]])
print(f"Variáveis sem IQR positivo: {(~validas_iqr).sum()}")
fig, ax = plt.subplots(figsize=(10, 6))
relatorio_iqr.pct_fora_dos_limites.dropna().head(15).sort_values().plot.barh(ax=ax, color="#c16942")
ax.set(xlabel="% entre os valores observados", title="Colunas com mais valores fora dos limites IQR")
mostrar_grafico(fig, "06_outliers_iqr")

novas_numericas = [c for c in derivadas if c in numericas]
fig, axes = plt.subplots(4, 3, figsize=(15, 10))
for c, ax in zip(novas_numericas, axes.flat):
    ax.hist(eda[c].dropna(), bins=24, color="#327a8a")
    ax.set(title=c.removeprefix("FE_"), ylabel="Escolas")
for ax in list(axes.flat)[len(novas_numericas):]:
    ax.set_visible(False)
mostrar_grafico(fig, "07_distribuicoes_features")

# %% Correlações numéricas entre variáveis e com a mediana da média geral do ENEM
# Usamos Censo, indicadores e features derivadas, além do alvo. As notas,
# presenças e contagens do ENEM 2025 ficam fora dos candidatos a preditor.
candidatas = [c for c in numericas if c in colunas_censo + colunas_indicadores + derivadas]
matriz = eda[[alvo, *candidatas]]
spearman = matriz.corr(method="spearman", min_periods=100)
pearson = matriz.corr(method="pearson", min_periods=100)
associacoes = pd.DataFrame({"spearman": spearman[alvo], "pearson": pearson[alvo],
    "escolas_com_os_dois_valores": matriz.loc[matriz[alvo].notna()].count(),
}).drop(index=alvo).sort_values("spearman", key=lambda s: s.abs(), ascending=False)
display(associacoes)
display(associacoes.reindex(novas_numericas))
spearman.to_csv(pasta_graficos / "correlacao_spearman_completa.csv")
associacoes.to_csv(pasta_graficos / "correlacao_com_alvo.csv")

top = associacoes.spearman.dropna().head(14).index.tolist()
selecionadas = [alvo, *top]
fig, ax = plt.subplots(figsize=(12, 10))
img = ax.imshow(spearman.loc[selecionadas, selecionadas], vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(selecionadas)), selecionadas, rotation=55, ha="right", fontsize=8)
ax.set_yticks(range(len(selecionadas)), selecionadas, fontsize=8)
ax.set_title("Correlação de Spearman entre o alvo e as variáveis mais associadas")
fig.colorbar(img, ax=ax, shrink=.75, label="Spearman (ρ)")
mostrar_grafico(fig, "08_matriz_correlacao")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, cols, titulo in [(axes[0], top, "Maiores associações numéricas"),
                         (axes[1], novas_numericas, "Features criadas")]:
    associacoes.loc[cols, ["spearman", "pearson"]].sort_values("spearman").plot.barh(ax=ax, color=["#327a8a", "#c16942"])
    ax.set(xlabel="Coeficiente com a mediana da média geral do ENEM", title=titulo, xlim=(-1, 1))
    ax.axvline(0, color="gray", linewidth=.7)
mostrar_grafico(fig, "09_correlacao_com_alvo")

# %% [markdown]
# A matriz de confusão exige classes observadas e previstas, portanto se aplica
# a um modelo de classificação. Para a média contínua, a avaliação posterior
# pode usar MAE, RMSE, gráfico de valores reais e previstos e gráfico de resíduos.
# Esta exploração descreve a base. A escolha de features guiada pela relação com
# o alvo deve ser refeita no treino quando houver uma divisão para avaliação.

