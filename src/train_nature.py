# %%
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn import model_selection,preprocessing,experimental,impute,linear_model,ensemble,metrics
from sklearn.experimental import enable_iterative_imputer
from boruta import BorutaPy
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn import pipeline
# Caminhos do projeto, para executar o arquivo ou suas celulas.
if "__file__" in globals():
    pasta_projeto = Path(__file__).resolve().parents[1]
else:
    pasta_projeto = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()
# %%
# MLflow local: usa um banco separado e nao altera o mlflow.db existente.
mlflow.set_tracking_uri("sqlite:///" + (pasta_projeto / "mlflow_nature.db").as_posix())
mlflow.set_experiment("XCL-Churn")
# %%
# IMPORTANDO DADOS....
df = pd.read_csv(pasta_projeto / "data" / "Train.csv")
df.head()
# %%
# Limpeza de dados
# Dropando as colunas ZONE1,ZONE2,MRG,TOP_PACK,FREQ_TOP_PACK,REGION
df = df.drop(columns=['ZONE1',
                       'ZONE2',
                         'MRG', 
                         'TOP_PACK',
                         'FREQ_TOP_PACK'
                         ,'REGION'
                         ,'user_id'])
df.head()

# %%
# Codificação de dados
#Verificando tipos
df.dtypes
# %%
# Selecionando features categoricas
cat = df.columns[df.dtypes == 'object'].to_list()
cat
# %%
#Tratando Features Categoricas
encoder = preprocessing.OrdinalEncoder()

df[cat] = encoder.fit_transform(df[cat])

df.head()

# %%
# Preenchendo valores ausentes
# Separando a target para imputer e scalers receberem apenas features
target = "CHURN"
y = df[target].copy()
X = df.drop(columns=target)

nulos = X.columns[X.isna().sum() > 0].to_list()
nulos

# %%
imputer = impute.IterativeImputer(estimator=linear_model.BayesianRidge(),
                                  initial_strategy='median',
                                  max_iter=20,
                                  tol=0.00001,
                                  random_state=42)
imputer
# %%
df_imputed = pd.DataFrame(imputer.fit_transform(X),
                          columns=X.columns,
                          index=X.index)

df_imputed.head()
# %%
# Dimensionamento dos dados
scaler_robust = preprocessing.RobustScaler()

df_robust = pd.DataFrame(scaler_robust.fit_transform(df_imputed),
                         columns=df_imputed.columns,
                         index=df_imputed.index)

df_robust.head()
# %%
scaler_standard = preprocessing.StandardScaler()

df_stantard = pd.DataFrame(scaler_standard.fit_transform(df_robust),
                           columns=df_robust.columns,
                           index=df_robust.index)

df_stantard.head()
# %%
scaler_minmax = preprocessing.MinMaxScaler()

df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df_stantard),
                         columns=df_stantard.columns,
                         index=df_stantard.index)

df_minmax.head()
# %%
# Selecionando Recursos
features = df_minmax.columns.tolist()

target
# %%
# Todas as colunas sao features; y conserva os rotulos originais
X = df_minmax

y
# %%
rfc = ensemble.RandomForestClassifier(n_estimators=100,
                                      random_state=42,
                                      n_jobs=-1)
rfc
# %%
boruta = BorutaPy(estimator=rfc,
                  n_estimators=100,
                  perc=90,
                  max_iter=100,
                  alpha= 0.05,
                  random_state=42,
                  verbose=2)
boruta
# %%
boruta.fit(X.to_numpy(), y.to_numpy())
# %%
selected_features = X.columns[boruta.support_].tolist()
if not selected_features:
    raise ValueError("Boruta nao confirmou nenhuma feature; confira ranking e dados antes de treinar.")

selected_features
# %%
ranking = pd.DataFrame({
    'Feature': X.columns,
    'Ranking': boruta.ranking_,
    'Selecionada': boruta.support_
})

ranking.sort_values('Ranking')
# %%
# Melhores features selecionadas
X_selected = X[selected_features]
X_selected.head()
# %%
# Divisao entre treino e teste
X_train,X_test,y_train,y_test = model_selection.train_test_split(X[selected_features]
                                                                 ,y
                                                                 ,test_size=0.4
                                                                 ,random_state=42
                                                                 ,stratify=y,

                                                                 )

# %%
#Balanceamento da classe minoritaria da target
smote = SMOTE(sampling_strategy=0.5,
              k_neighbors=5,
              random_state=42)

smote

# %%
X_train,y_train = smote.fit_resample(X_train,y_train)
# %%
# MODELAGEM - XGBoost

modelo_xgb = XGBClassifier(random_state=42)
modelo_xgb
# %%
# MODELAGEM - Catboost
modelo_cat = CatBoostClassifier(random_seed=42,
                                verbose=False,
                                allow_writing_files=False)

modelo_cat
# %%
# MODELAGEM — LightGBM
modelo_lgbm = LGBMClassifier(random_state=42,
                             verbosity = -1,)
modelo_lgbm
# %%
# Parametros para testar no XGBoost
params_xgb = {
    "n_estimators":[300],
    "max_depth":[8],
    "learning_rate":[0.05],
    "subsample":[0.8]
}
# Parametros para testar no CatBoost
params_cat = {
    "iterations":[400],
    "depth":[7],
    "learning_rate": [0.03],
    "l2_leaf_reg":  [3],
}
# Parametros para testar no LightGBM
params_lgbm = {
    "n_estimators": [100], # Escolha local: quantidade nao informada no artigo.
    "num_leaves": [64],
    "max_depth": [7],
    "learning_rate": [0.04],
    "feature_fraction": [0.85],
}
# Configuracoes finais publicadas: um candidato por modelo.
# Mantemos a validacao de cinco folds, mas isto NAO reconstroi a busca original.
# %%
cv = model_selection.StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
def criarbusca(modelo,params):

    return model_selection.RandomizedSearchCV(
        estimator=modelo,
        param_distributions=params,
        n_iter=1,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=2,
        refit=True,
        error_score="raise",
    )

busca_xgb = criarbusca(modelo_xgb,params_xgb)
busca_cat = criarbusca(modelo_cat,params_cat)
busca_lgbm = criarbusca(modelo_lgbm,params_lgbm)

# Valida e ajusta cada modelo separadamente antes da combinacao.
busca_xgb.fit(X_train, y_train)
busca_cat.fit(X_train, y_train)
busca_lgbm.fit(X_train, y_train)

# %%
# Aplica os parametros encontrados aos modelos que formarao o ensemble.
modelo_xgb.set_params(**busca_xgb.best_params_)
modelo_cat.set_params(**busca_cat.best_params_)
modelo_lgbm.set_params(**busca_lgbm.best_params_)

voting = ensemble.VotingClassifier(
    estimators=[
        ("xgb",modelo_xgb),
        ("cat",modelo_cat),
        ("lgbm",modelo_lgbm),

    ],
    voting="soft",
    # Pesos escolhidos na ablacao do artigo, na ordem XGB, CatBoost e LightGBM.
    weights=[3,2,3],
    n_jobs=1,
)
voting
#  %%

with mlflow.start_run(run_name="XCL-Churn") as run:
    model_pipeline = pipeline.Pipeline(steps=[
        ("Algoritmo",voting)
    ])

    model_pipeline
    # Autolog precisa estar ativo durante o fit, como no projeto Loyalty.
    mlflow.sklearn.autolog(log_post_training_metrics=False, log_datasets=False)
    model_pipeline.fit(X_train, y_train)
    # Evita que avaliacoes em outras celulas criem registros automaticos adicionais.
    mlflow.sklearn.autolog(disable=True)

    # Previsoes no X_train balanceado pelo SMOTE
    y_pred_train = model_pipeline.predict(X_train)
    y_proba_train = model_pipeline.predict_proba(X_train)[:, 1]
    # Previsoes no teste
    y_pred_test = model_pipeline.predict(X_test)
    y_proba_test = model_pipeline.predict_proba(X_test)[:, 1]


    modelos = {
        "XGBoost": busca_xgb,
        "CatBoost": busca_cat,
        "LightGBM": busca_lgbm,
        "XCL-Churn": model_pipeline,
    }

    def avaliar_modelo(y_real, y_pred):
        return {
            "Acuracia": metrics.accuracy_score(y_real, y_pred),
            "Precisao": metrics.precision_score(y_real, y_pred, pos_label=1, zero_division=0),
            "Recall": metrics.recall_score(y_real, y_pred, pos_label=1, zero_division=0),
            "F1": metrics.f1_score(y_real, y_pred, pos_label=1, zero_division=0),
        }

    # Comparadores mencionados na discussao do artigo.
    # Seus hiperparametros nao foram publicados: usamos defaults e seed 42.
    comparadores = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Ridge": linear_model.RidgeClassifier(),
        "SGD": linear_model.SGDClassifier(random_state=42),
        "Random Forest": ensemble.RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    for nome, modelo in comparadores.items():
        modelo.fit(X_train, y_train)
        modelos[nome] = modelo

    # Stacking: o meta-modelo aprende com previsoes fora do fold de treinamento.
    # Regressao logistica como meta-modelo e uma escolha local nao detalhada no artigo.
    estimadores_base = [
        ("xgb", modelo_xgb), ("cat", modelo_cat),
        ("lgbm", modelo_lgbm),
    ]
    stacking = ensemble.StackingClassifier(
        estimators=estimadores_base,
        final_estimator=linear_model.LogisticRegression(max_iter=1000, random_state=42),
        stack_method="predict_proba", cv=cv, n_jobs=1)
    stacking.fit(X_train, y_train)
    modelos["Stacking"] = stacking

    # Blending: reserva 20% do treino para ensinar o meta-modelo.
    # Percentual e meta-modelo sao escolhas locais; o teste final nao participa.
    # cv="prefit" reutiliza as bases ajustadas SOMENTE na primeira parcela.
    X_base, X_meta, y_base, y_meta = model_selection.train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    # Novos modelos, declarados explicitamente, sem alterar os modelos principais.
    xgb_blending = XGBClassifier(random_state=42, **busca_xgb.best_params_)
    cat_blending = CatBoostClassifier(
        random_seed=42, verbose=False, allow_writing_files=False, **busca_cat.best_params_)
    lgbm_blending = LGBMClassifier(random_state=42, verbosity=-1, **busca_lgbm.best_params_)

    xgb_blending.fit(X_base, y_base)
    cat_blending.fit(X_base, y_base)
    lgbm_blending.fit(X_base, y_base)
    bases_blending = [
        ("xgb", xgb_blending), ("cat", cat_blending), ("lgbm", lgbm_blending),
    ]
    blending = ensemble.StackingClassifier(
        estimators=bases_blending,
        final_estimator=linear_model.LogisticRegression(max_iter=1000, random_state=42),
        stack_method="predict_proba", cv="prefit")
    blending.fit(X_meta, y_meta)
    modelos["Blending"] = blending

    # Mesmos clientes e mesmas metricas para todos os modelos.
    linhas = []
    for nome, modelo in modelos.items():
        metricas_train = avaliar_modelo(y_train, modelo.predict(X_train))
        metricas_test = avaliar_modelo(y_test, modelo.predict(X_test))
        linhas.append({"Modelo": nome, "Conjunto": "Treino", **metricas_train})
        linhas.append({"Modelo": nome, "Conjunto": "Teste", **metricas_test})

    resultados = pd.DataFrame(linhas)
    resultados
    # ASSESS - mesmo padrao do projeto Loyalty: log_metrics, savefig e log_artifact.
    pasta_arquivos = pasta_projeto / "artifacts" / "nature" / run.info.run_id
    pasta_arquivos.mkdir(parents=True, exist_ok=True)

    for nome, modelo in modelos.items():
        treino = resultados[(resultados["Modelo"] == nome) &
                            (resultados["Conjunto"] == "Treino")].iloc[0]
        teste = resultados[(resultados["Modelo"] == nome) &
                           (resultados["Conjunto"] == "Teste")].iloc[0]

        mlflow.log_metrics({
            f"{nome}_acc_train": treino["Acuracia"],
            f"{nome}_precision_train": treino["Precisao"],
            f"{nome}_recall_train": treino["Recall"],
            f"{nome}_f1_train": treino["F1"],
            f"{nome}_acc_test": teste["Acuracia"],
            f"{nome}_precision_test": teste["Precisao"],
            f"{nome}_recall_test": teste["Recall"],
            f"{nome}_f1_test": teste["F1"],
        })

        # Registra cada modelo diretamente no MLflow, sem criar arquivos .pkl.
        # A Pipeline XCL-Churn ja foi registrada pelo autolog durante o fit.
        if nome != "XCL-Churn":
            mlflow.sklearn.log_model(
                modelo, name=nome.replace(" ", "_"), serialization_format="cloudpickle")

    # Tabela com as quatro metricas, em escala de 0 a 1.
    arquivo_metricas = str(pasta_arquivos / "metricas_por_modelo.csv")
    resultados.to_csv(arquivo_metricas, index=False)
    mlflow.log_artifact(arquivo_metricas)

    # Um grafico para cada metrica, comparando todos os modelos.
    for metrica in ["Acuracia", "Precisao", "Recall", "F1"]:
        tabela = resultados.pivot(index="Modelo", columns="Conjunto", values=metrica)
        tabela = tabela.loc[list(modelos), ["Treino", "Teste"]]
        (tabela * 100).plot.bar(figsize=(14, 6), rot=30)
        plt.title(f"Comparacao de {metrica}")
        plt.ylabel(f"{metrica} (%)")
        plt.ylim(0, 105)
        plt.tight_layout()
        arquivo_grafico = str(pasta_arquivos / f"comparacao_{metrica.lower()}.png")
        plt.savefig(arquivo_grafico)
        mlflow.log_artifact(arquivo_grafico)
        plt.show()
        plt.close()

    # Matriz de confusao do XCL-Churn, no treino e no teste.
    for conjunto, y_real, y_pred in [
        ("Treino", y_train, y_pred_train),
        ("Teste", y_test, y_pred_test),
    ]:
        metrics.ConfusionMatrixDisplay.from_predictions(
            y_real, y_pred, labels=[0, 1], cmap="Blues", values_format="d")
        plt.title(f"XCL-Churn - {conjunto}")
        arquivo_grafico = str(pasta_arquivos / f"matriz_confusao_{conjunto.lower()}.png")
        plt.savefig(arquivo_grafico)
        mlflow.log_artifact(arquivo_grafico)
        plt.show()
        plt.close()

    # Curva ROC: mesma logica do seu projeto, sem OOT (nao usado neste protocolo).
    auc_train = metrics.roc_auc_score(y_train, y_proba_train)
    auc_test = metrics.roc_auc_score(y_test, y_proba_test)
    mlflow.log_metrics({"auc_train": auc_train, "auc_test": auc_test})

    roc_train = metrics.roc_curve(y_train, y_proba_train)
    roc_test = metrics.roc_curve(y_test, y_proba_test)
    plt.figure(figsize=(10, 4))
    plt.plot(roc_train[0], roc_train[1])
    plt.plot(roc_test[0], roc_test[1])
    plt.xlabel("1 - Especificidade")
    plt.ylabel("Recall")
    plt.title("XCL-Churn - Curva ROC")
    plt.grid(True)
    plt.legend([f"AUC Treino: {auc_train:.4f}", f"AUC Teste: {auc_test:.4f}"])
    plt.tight_layout()
    arquivo_grafico = str(pasta_arquivos / "curva_roc.png")
    plt.savefig(arquivo_grafico)
    mlflow.log_artifact(arquivo_grafico)
    plt.show()
    plt.close()

    # Precision-Recall tambem aparece na Figura 12 do artigo.
    pr_train = metrics.precision_recall_curve(y_train, y_proba_train)
    pr_test = metrics.precision_recall_curve(y_test, y_proba_test)
    plt.figure(figsize=(10, 4))
    plt.plot(pr_train[1], pr_train[0], label="Treino")
    plt.plot(pr_test[1], pr_test[0], label="Teste")
    plt.xlabel("Recall")
    plt.ylabel("Precisao")
    plt.title("XCL-Churn - Curva Precision-Recall")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    arquivo_grafico = str(pasta_arquivos / "curva_precision_recall.png")
    plt.savefig(arquivo_grafico)
    mlflow.log_artifact(arquivo_grafico)
    plt.show()
    plt.close()

    # Experimentos complementares: sempre salvando no mesmo run.
    # Registra as features realmente selecionadas; nao forca uma lista ambigua.
    ranking.to_csv(pasta_arquivos / "boruta_ranking.csv", index=False)
    mlflow.log_artifact(str(pasta_arquivos / "boruta_ranking.csv"))
    print("Features selecionadas pelo Boruta:", selected_features)
    mlflow.log_params({"cv_folds": cv.n_splits, "candidatos_por_modelo": 1,
                       "protocolo": "Configuracao final publicada; busca original nao reconstruida",
                       "lgbm_n_estimators_nao_informado": 100})
    # Comparacao somente de metricas; medicao de tempo omitida por simplicidade.
    comparacao_ensembles = resultados[resultados["Modelo"].isin(["XCL-Churn", "Stacking", "Blending"])]
    comparacao_ensembles.to_csv(pasta_arquivos / "comparacao_ensembles.csv", index=False)
    mlflow.log_artifact(str(pasta_arquivos / "comparacao_ensembles.csv"))

    # Combinacoes da Tabela 5, inclusive as repetidas no texto publicado.
    # Repeticoes aqui terao os mesmos resultados. Nao usamos o teste para
    # escolher pesos: o modelo principal permanece com 3,2,3 a priori.
    pesos_artigo = [
        (1,1,1), (1,1,2), (1,1,3), (1,1,2), (1,2,1), (1,2,2), (1,2,3),
        (2,2,1), (2,2,2), (2,2,3), (2,3,1), (2,3,2), (2,3,3),
        (2,3,1), (2,3,2), (2,3,3), (3,1,1), (3,1,2), (3,1,3),
        (3,1,1), (3,1,2), (3,1,3), (3,2,1), (3,2,2), (3,2,3),
    ]
    probabilidades = np.array([modelos[nome].predict_proba(X_test)
                              for nome in ["XGBoost", "CatBoost", "LightGBM"]])
    linhas_pesos = []
    for pesos in pesos_artigo:
        media = np.average(probabilidades, axis=0, weights=pesos)
        pred = model_pipeline.classes_[media.argmax(axis=1)]
        linhas_pesos.append({"Peso_XGB": pesos[0], "Peso_Cat": pesos[1], "Peso_LGBM": pesos[2],
                            **avaliar_modelo(y_test, pred)})
    resultados_pesos = pd.DataFrame(linhas_pesos)
    resultados_pesos.to_csv(pasta_arquivos / "comparacao_pesos.csv", index=False)
    mlflow.log_artifact(str(pasta_arquivos / "comparacao_pesos.csv"))

    # SHAP explica a probabilidade do ensemble, nao apenas um dos tres modelos.
    # Amostras pequenas controlam o custo. Tamanhos e KernelExplainer sao escolhas
    # locais, pois o artigo nao informa como configurou os explicadores.
    def prever_probabilidade(dados):
        return model_pipeline.predict_proba(pd.DataFrame(dados, columns=selected_features))

    def prever_churn(dados):
        return prever_probabilidade(dados)[:, 1]

    fundo_shap = X_train.sample(n=min(50, len(X_train)), random_state=42)
    amostra_shap = X_test.sample(n=min(100, len(X_test)), random_state=42)
    np.random.seed(42)
    explicador_shap = shap.KernelExplainer(prever_churn, fundo_shap)
    valores_shap = explicador_shap.shap_values(amostra_shap, nsamples=200, l1_reg=0)
    importancia_shap = pd.Series(np.abs(valores_shap).mean(axis=0), index=selected_features)

    mlflow.log_params({"shap_fundo": len(fundo_shap), "shap_amostra": len(amostra_shap),
                       "shap_nsamples": 200, "shap_explicador": "KernelExplainer"})
    importancia_shap.sort_values(ascending=False).rename("SHAP_medio_absoluto").to_csv(
        pasta_arquivos / "importancia_shap.csv", index_label="Feature")
    mlflow.log_artifact(str(pasta_arquivos / "importancia_shap.csv"))
    importancia_shap.sort_values().plot.barh(figsize=(9, 6))
    plt.title("XCL-Churn - Importancia SHAP na amostra")
    plt.xlabel("Media do valor absoluto SHAP (probabilidade de churn)")
    plt.tight_layout()
    plt.savefig(pasta_arquivos / "importancia_shap.png")
    mlflow.log_artifact(str(pasta_arquivos / "importancia_shap.png"))
    plt.show()
    plt.close()
    shap.summary_plot(valores_shap, amostra_shap, show=False)
    plt.tight_layout()
    plt.savefig(pasta_arquivos / "shap_distribuicao.png")
    mlflow.log_artifact(str(pasta_arquivos / "shap_distribuicao.png"))
    plt.close()

    # LIME explica um cliente, escolhido a priori: a primeira linha do teste.
    # Se uma categorica foi selecionada, recupera codigos inteiros para o LIME
    # e os converte de volta aos valores escalados antes de chamar o modelo.
    dados_lime = X_train.sample(n=min(5000, len(X_train)), random_state=42).copy()
    cliente_lime = X_test.iloc[0].copy()
    categorias_lime = {}
    for coluna in cat:
        if coluna in selected_features:
            valores = np.sort(X_train[coluna].unique())
            categorias_lime[coluna] = valores
            dados_lime[coluna] = np.searchsorted(valores, dados_lime[coluna])
            cliente_lime[coluna] = np.searchsorted(valores, cliente_lime[coluna])

    def prever_lime(dados):
        dados = pd.DataFrame(dados, columns=selected_features)
        for coluna, valores in categorias_lime.items():
            dados[coluna] = valores[dados[coluna].astype(int)]
        return prever_probabilidade(dados)

    explicador_lime = LimeTabularExplainer(
        dados_lime.to_numpy(), feature_names=selected_features,
        categorical_features=[selected_features.index(c) for c in categorias_lime],
        class_names=["Nao churn", "Churn"], mode="classification", random_state=42)
    explicacao_lime = explicador_lime.explain_instance(
        cliente_lime.to_numpy(), prever_lime, labels=(1,),
        num_features=len(selected_features), num_samples=5000)
    mlflow.log_params({"lime_amostra": len(dados_lime), "lime_perturbacoes": 5000,
                       "lime_indice_cliente": str(X_test.index[0])})
    explicacao_lime.save_to_file(str(pasta_arquivos / "lime_cliente.html"))
    mlflow.log_artifact(str(pasta_arquivos / "lime_cliente.html"))
    explicacao_lime.as_pyplot_figure(label=1)
    plt.tight_layout()
    plt.savefig(pasta_arquivos / "lime_cliente.png")
    mlflow.log_artifact(str(pasta_arquivos / "lime_cliente.png"))
    plt.show()
    plt.close()

# %%
print("Execucao no MLflow:", run.info.run_id)
print("Modelos e graficos salvos em:", pasta_arquivos)
# Os modelos registrados recebem as features ja tratadas, na ordem de selected_features.
