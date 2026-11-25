"""Experimento pareado por escola com os regressores do projeto, sem usar o teste."""
# %% Carregar somente as escolas do treino usado no experimento original
from pathlib import Path
import sqlite3
from contextlib import closing

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split

raiz = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if not (raiz / "data").is_dir():
    raiz = raiz.parent
with closing(sqlite3.connect((raiz / "data/enem_censo_escolar_2025.db").as_uri() + "?mode=ro", uri=True)) as db:
    base = pd.read_sql_query("SELECT * FROM final ORDER BY NU_ANO, CO_ESCOLA", db)
treino, _ = train_test_split(base, test_size=.30, random_state=42)
X = treino.drop(columns=["CO_ESCOLA", "NU_ANO", "NU_NOTA_GERAL_MEDIANA"])
y = treino.NU_NOTA_GERAL_MEDIANA
campos = ["ATU_MED_CAT_0", "ATU_MED_01_CAT_0", "ATU_MED_02_CAT_0", "ATU_MED_03_CAT_0",
          "FE_TURMAS_BAS_POR_SALA", "FE_TURMAS_MED_POR_SALA", "FE_COMPUTADORES_POR_SALA"]
mlflow.set_tracking_uri("sqlite:///" + (raiz / "data/mlflow_regressao_escolar.db").as_posix())
modelos = {nome: mlflow.sklearn.load_model(f"models:/ENEM_Escola_{nome}/1")
           for nome in ("RegressaoLinear", "RandomForest", "XGBoost")}

# %% Limitar apenas a cauda superior. Recalcular cada teto dentro da dobra.
linhas = []
for dobra, (ajuste, validacao) in enumerate(KFold(3, shuffle=True, random_state=42).split(X), 1):
    a, v = X.iloc[ajuste], X.iloc[validacao]
    q1, q3 = a[campos].quantile(.25), a[campos].quantile(.75)
    teto = (q3 + 1.5 * (q3 - q1)).where(q3 > q1)
    for metodo in ("original", "limite_IQR"):
        xa, xv = a.copy(), v.copy()
        if metodo == "limite_IQR":
            xa[campos] = xa[campos].clip(upper=teto, axis="columns")
            xv[campos] = xv[campos].clip(upper=teto, axis="columns")
        for nome, referencia in modelos.items():
            modelo = clone(referencia).fit(xa, y.iloc[ajuste])
            previsto = modelo.predict(xv)
            linhas.append({"dobra": dobra, "metodo": metodo, "modelo": nome,
                           "MAE": mean_absolute_error(y.iloc[validacao], previsto),
                           "RMSE": root_mean_squared_error(y.iloc[validacao], previsto)})
    print(f"Dobra {dobra} concluída", flush=True)
resultado = pd.DataFrame(linhas)
resultado.to_csv(raiz / "output/regressao_escolas_2025/sensibilidade_outliers.csv", index=False)
print(resultado.groupby(["modelo", "metodo"])[["MAE", "RMSE"]].mean().round(3))
