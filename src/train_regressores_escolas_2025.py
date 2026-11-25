# %% SEMMA — Sample: escolas de uma única edição
from pathlib import Path
from contextlib import closing
import sqlite3

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder, FunctionTransformer
from xgboost import XGBRegressor

raiz = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if not (raiz / "data").is_dir():
    raiz = raiz.parent
saida = raiz / "output/regressao_escolas_final_2025"
saida.mkdir(parents=True, exist_ok=True)
alvo = "NU_NOTA_GERAL_MEDIANA"
banco = raiz / "data/enem_censo_escolar_2025.db"
with closing(sqlite3.connect(banco.as_uri() + "?mode=ro", uri=True)) as conexao:
    df = pd.read_sql_query('SELECT * FROM final ORDER BY NU_ANO, CO_ESCOLA', conexao)
assert not df.duplicated().any() and df.CO_ESCOLA.notna().all()
assert df.NU_ANO.eq(2025).all() and df.CO_ESCOLA.is_unique and df[alvo].notna().all()
X = df.set_index("CO_ESCOLA").drop(columns=["NU_ANO", alvo])
y = df.set_index("CO_ESCOLA")[alvo]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=42)

# %% Explore: ler os códigos especiais antes de contar valores ausentes
mapa = pd.read_csv(raiz / "artifacts/censo_eda_comparacao/mapa_colunas_dicionarios.csv")
codigos_especiais = {}
for codigo, padrao in [(88888, r"\b88888\b"), (9, r"(?m)^\s*9\s*[-–]\s*Não informado")]:
    marcadas = mapa.loc[mapa.categorias_2025.str.contains(padrao, case=False, na=False), "coluna"]
    codigos_especiais[codigo] = X.columns.intersection(marcadas).tolist()

categoricas = [c for c in X if c.startswith("IN_") or
              (c.startswith("TP_") and "_PROP_" not in c) or
              not pd.api.types.is_numeric_dtype(X[c])]

def normalizar(base, codigos, categorias):
    base = base.copy()
    for codigo, colunas in codigos.items():
        base[colunas] = base[colunas].mask(base[colunas].eq(codigo))
    for c in categorias:
        base[c] = base[c].astype("string").astype(object).where(base[c].notna(), np.nan)
    return base

normalizacao = FunctionTransformer(normalizar, kw_args={"codigos": codigos_especiais, "categorias": categoricas})
treino_eda = normalizacao.transform(X_train)

def proporcao_composicao(c):
    return c.startswith(("AFD_", "TP_STATUS_REDACAO"))

def colunas_numericas(base):
    return [c for c in base.select_dtypes("number") if base[c].isna().mean() <= .5]

def colunas_composicao(base):
    return [c for c in colunas_numericas(base) if proporcao_composicao(c)]

def demais_numericas(base):
    return [c for c in colunas_numericas(base) if not proporcao_composicao(c)]

def colunas_categoricas(base):
    return [c for c in base.select_dtypes(exclude="number") if base[c].isna().mean() <= .5]

numericas = colunas_numericas(treino_eda)
nominais = colunas_categoricas(treino_eda)
perfil = pd.DataFrame({"nulos": treino_eda.isna().sum(), "fracao_nulos": treino_eda.isna().mean(),
                       "distintos": treino_eda.nunique()})
perfil["nulos_originais"] = X_train.isna().sum()
perfil["tipo"] = treino_eda.dtypes.astype(str)
perfil["tratamento"] = "excluída (>50% nulos)"
perfil.loc[numericas, "tratamento"] = "mediana do treino"
perfil.loc[colunas_composicao(treino_eda), "tratamento"] = "média do grupo de treino por categoria"
perfil.loc[nominais, "tratamento"] = "categoria Sem informação"
perfil.to_csv(saida / "perfil_e_imputacao.csv")
print(y_train.describe(), "\nTreino e teste:", len(y_train), len(y_test))
print(perfil.loc[perfil.nulos.gt(0)].to_string())

# %% Explore: distribuição, extremos e relações com o alvo contínuo
valores = treino_eda[numericas].join(y_train)
q1, q3 = valores.quantile(.25), valores.quantile(.75)
iqr = q3 - q1
fora = valores.lt(q1 - 1.5 * iqr) | valores.gt(q3 + 1.5 * iqr)
resumo = valores.describe().T
resumo["extremos_IQR"] = fora.sum().where(iqr > 0)
resumo["spearman_com_alvo"] = valores.corrwith(y_train, method="spearman")
resumo.to_csv(saida / "distribuicoes_treino.csv")
grupos = []
for c in nominais:
    grupos.append(treino_eda[[c]].join(y_train).groupby(c, dropna=False)[alvo]
                  .agg(["size", "median"]).rename_axis("categoria").reset_index().assign(feature=c))
pd.concat(grupos).to_csv(saida / "categorias_treino.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(y_train, bins=30, color="#296b76")
axes[0].set(xlabel="Mediana escolar (pontos)", ylabel="Escolas", title="Distribuição do alvo no treino")
gaps = perfil.loc[perfil.fracao_nulos.between(0, .5, inclusive="right"), "fracao_nulos"].nlargest(8)
(100 * gaps).sort_values().plot.barh(ax=axes[1], color="#9a5940")
axes[1].set(xlabel="Escolas sem informação (%)", title="Maiores lacunas entre as entradas mantidas")
fig.tight_layout()
fig.savefig(saida / "exploracao_treino.png", dpi=150)
plt.show()

# %% Modify: preparo dentro da validação. Ausência de registro difere de zero observado.
# Nenhuma coluna faltante mantida tem garantia documental de ausência física.
# TargetEncoder usa médias suavizadas, com codificação fora da dobra no fit_transform.
preparo = ColumnTransformer([
    ("numericas", Pipeline([("imputacao", SimpleImputer(strategy="median")),
                            ("escala", StandardScaler())]), demais_numericas),
    ("composicao", Pipeline([("imputacao", SimpleImputer(strategy="mean")),
                             ("escala", StandardScaler())]), colunas_composicao),
    ("categoricas", Pipeline([("imputacao", SimpleImputer(strategy="constant", fill_value="Sem informação")),
                              ("codificacao", TargetEncoder(target_type="continuous", cv=5, random_state=42))]),
     colunas_categoricas)
])

# %% Model: escolher configurações por MAE em três dobras do treino
modelos = {
    "RegressaoLinear": (LinearRegression(), {}),
    "RandomForest": (RandomForestRegressor(n_estimators=100, n_jobs=4, random_state=42),
                     {"modelo__min_samples_leaf": [3, 10]}),
    "XGBoost": (XGBRegressor(n_estimators=180, learning_rate=.05, objective="reg:squarederror",
                            tree_method="hist", n_jobs=4, random_state=42),
                {"modelo__max_depth": [2, 4]})
}
cv = KFold(n_splits=3, shuffle=True, random_state=42)
mlflow.set_tracking_uri("sqlite:///" + (raiz / "data/mlflow_regressao_escolar.db").as_posix())
experimento = "ENEM_mediana_escolar_2025"
if mlflow.get_experiment_by_name(experimento) is None:
    mlflow.create_experiment(experimento, artifact_location=(saida / "mlartifacts").as_uri())
mlflow.set_experiment(experimento)

# %% Assess: referência da mediana, métricas por escola e registro da pipeline completa
def avaliar(observado, previsto):
    return {"MAE": mean_absolute_error(observado, previsto),
            "RMSE": root_mean_squared_error(observado, previsto), "R2": r2_score(observado, previsto),
            "R_Pearson": np.corrcoef(observado, previsto)[0, 1] if np.ptp(previsto) else np.nan}

resultados = []
previsoes = pd.DataFrame({"observado": y_test})
for conjunto, real in [("treino", y_train), ("teste", y_test)]:
    resultados.append({"modelo": "Mediana do treino", "conjunto": conjunto,
                       **avaliar(real, np.full(len(real), y_train.median()))})
for nome, (modelo, parametros) in modelos.items():
    with mlflow.start_run(run_name=nome):
        busca = GridSearchCV(Pipeline([("normalizacao", normalizacao), ("preparo", preparo), ("modelo", modelo)]), parametros,
                             scoring="neg_mean_absolute_error", cv=cv, error_score="raise")
        busca.fit(X_train, y_train)
        mlflow.log_params(busca.best_params_)
        for arquivo in ("perfil_e_imputacao.csv", "distribuicoes_treino.csv",
                        "categorias_treino.csv", "exploracao_treino.png"):
            mlflow.log_artifact(str(saida / arquivo), artifact_path="exploracao")
        mlflow.log_metric("MAE_validacao", -busca.best_score_)
        for conjunto, dados, real in [("treino", X_train, y_train), ("teste", X_test, y_test)]:
            previsto = busca.predict(dados)
            metricas = avaliar(real, previsto)
            resultados.append({"modelo": nome, "conjunto": conjunto,
                               "MAE_validacao": -busca.best_score_, **metricas})
            mlflow.log_metrics({f"{k}_{conjunto}": v for k, v in metricas.items()})
            if conjunto == "teste":
                previsoes[nome] = previsto
        mlflow.sklearn.log_model(busca.best_estimator_, name="pipeline", serialization_format="cloudpickle",
                                registered_model_name="ENEM_Escola_" + nome,
                                pip_requirements=[f"scikit-learn=={sklearn.__version__}",
                                                  f"xgboost=={xgboost.__version__}", f"pandas=={pd.__version__}",
                                                  f"numpy=={np.__version__}", "cloudpickle"])
        print(nome, resultados[-1], flush=True)
metricas = pd.DataFrame(resultados)
metricas.to_csv(saida / "metricas.csv", index=False)
previsoes.to_csv(saida / "previsoes_teste.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
metricas[metricas.conjunto.eq("teste")].set_index("modelo")[["MAE","RMSE"]].plot.bar(ax=axes[0], rot=15)
axes[0].set(ylabel="Pontos ENEM", xlabel="", title="Erro por escola no teste")
escolhido = metricas.dropna(subset=["MAE_validacao"]).sort_values("MAE_validacao").iloc[0].modelo
axes[1].scatter(previsoes[escolhido], y_test-previsoes[escolhido], alpha=.25, s=9)
axes[1].axhline(0, color="gray")
axes[1].set(xlabel="Mediana prevista", ylabel="Observada menos prevista", title=f"Resíduos no teste: {escolhido}")
fig.tight_layout()
fig.savefig(saida / "avaliacao_teste.png", dpi=150)
plt.show()




# %% Assess: gráfico observado versus previsto e requisição de uma escola
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, previsoes[escolhido], alpha=.25, s=10)
limites = [min(y_test.min(), previsoes[escolhido].min()), max(y_test.max(), previsoes[escolhido].max())]
ax.plot(limites, limites, color="gray", linestyle="--")
ax.set(xlabel="Mediana observada (pontos ENEM)", ylabel="Mediana prevista (pontos ENEM)", title=escolhido)
fig.tight_layout()
fig.savefig(saida / "observado_previsto.png", dpi=160)
escola = X_test.iloc[[0]]
(saida / "requisicao.json").write_text('{"dataframe_split":' + escola.to_json(orient="split", index=False) + "}", encoding="utf-8")
versoes = mlflow.MlflowClient().search_model_versions(f"name = 'ENEM_Escola_{escolhido}'")
versao = max(int(v.version) for v in versoes)
print(f"models:/ENEM_Escola_{escolhido}/{versao} | escola {escola.index[0]} | previsão {previsoes.loc[escola.index[0], escolhido]}")
