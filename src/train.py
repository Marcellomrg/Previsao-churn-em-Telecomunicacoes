# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection,ensemble,pipeline,metrics,preprocessing
from feature_engine import imputation,encoding,selection
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_name="churn")
pd.set_option('display.max_rows', 200)
# %%
# SEMMA - SAMPLE
# IMPORTANDO OS DADOS....
df_train = pd.read_csv("../data/Train.csv")
df_train.head()
# %%
# Separando as Features
X = df_train.columns[1:-1].to_list()
X
# %%
# Separando target
y = df_train.columns[-1]
y
# %%
# SEPARANDO MINHAS FEATURES E VARIAVEL TARGET
X_train,X_test,y_train,y_test = model_selection.train_test_split(df_train[X],df_train[y],random_state=42,
                                                                 test_size=0.2,
                                                                 stratify=df_train[y])
# %%
###########################
# SEMMA - EXPLORE
df_explore = df_train[X].copy()
df_explore.head()
# %%
# FEATURES CATEGORICAS
cat = df_explore.columns[df_explore.dtypes == 'object']
cat
# %%
# FEATURES NUMERICAS
num = df_explore.columns[df_explore.dtypes != 'object']
num
# %%
# FEATURES COM VARIAVEIS NULAS 
nulos = df_explore.columns[df_explore.isna().sum() > 0].to_list()
nulos
# %%
df_explore.isna().sum()
# %%
df_explore.isna().mean()
# %%
bivariada_num = df_train.groupby(by='CHURN')[num].mean().T
bivariada_num['ratio'] = bivariada_num[0] / bivariada_num[1]
bivariada_num
# %%
df_explore['REGION'].unique()
# %%
df_explore['TENURE'].unique()
# %%
df_explore['MRG'].unique()
# %%
df_explore['TOP_PACK'].unique()
# %%
df_train.groupby('TOP_PACK')['CHURN'].agg(
    ['count', 'mean']
).sort_values('count', ascending=False)
# %%
# Separando os pacotes em grupos 
def classify_top_pack_column(X):
    X = X.copy()

    def classify(pack):
        if pd.isna(pack):
            return 'NO_PACK'

        pack = str(pack).lower().strip()

        if 'mixt' in pack or 'mix:' in pack:
            return 'MIXT'
        elif 'all-net' in pack:
            return 'ALL_NET'
        elif 'on-net' in pack or 'on net' in pack:
            return 'ON_NET'
        elif 'data' in pack or 'gprs' in pack:
            return 'DATA'
        elif 'wifi' in pack or 'mifi' in pack:
            return 'WIFI'
        elif 'jokko' in pack:
            return 'JOKKO'
        elif 'fnf' in pack:
            return 'FNF'
        elif 'ivr' in pack or 'vas' in pack:
            return 'VAS'
        elif 'evc' in pack:
            return 'EVC'
        elif 'cvm' in pack:
            return 'CVM'
        elif 'internat' in pack:
            return 'INTERNATIONAL'
        elif 'fifa' in pack:
            return 'FIFA'
        elif 'apanews' in pack:
            return 'NEWS'
        elif 'postpaid' in pack:
            return 'POSTPAID'
        elif 'clir' in pack:
            return 'CLIR'
        elif 'ymgx' in pack:
            return 'YMGX'
        elif 'pilot' in pack:
            return 'PILOT'
        elif 'incoming' in pack:
            return 'INCOMING'
        else:
            return 'OTHER'

    X['TOP_PACK'] = X['TOP_PACK'].apply(classify)

    return X


# %%
"""Separando as features para retirar: ZONE1 ,ZONE2 E MRG
,pois ZONE1 E ZONE2 tem mais de 90% de dado nulos 
e MRG só tem dados NO atrapalha na variabilidade do dado"""
to_remove = ['ZONE1','ZONE2','MRG']
to_remove
# %%
# SEMMA - MODIFY
# REMOVENDO FEATURES QUE NAO AJUDAM NO MEU MODELO
drop_features = selection.DropFeatures(features_to_drop=to_remove)
drop_features

# %%
# TRATANDO DADOS NULOS CATEGORICOS
cat_nulos = imputation.CategoricalImputer(fill_value='Unknown',
                                          variables='REGION')
cat_nulos
# %%
# Tratando DADOS NULOS NUMERICOS
fill_0 = list(set(nulos) - set(to_remove) - set(['REGION']) - set(['TOP_PACK']))
fill_0
# %%
imputation_0 = imputation.ArbitraryNumberImputer(arbitrary_number=0,
                                                 variables=fill_0)
imputation_0
# %%
# Tratando Variaveis Categoricas
cat_features = list(set(cat) - set(to_remove))
cat_features
# %%
onehot = encoding.OneHotEncoder(variables=cat_features)
onehot
# %%
# Modificando os dados da feature TOP_PACK
top_pack_transformer = preprocessing.FunctionTransformer(classify_top_pack_column,
                                                         validate=False)

top_pack_transformer
# %%
# SEMMA -MODEL
model = ensemble.RandomForestClassifier(random_state=42,
                                        n_jobs=-1)
model
# %%
params = {
    "n_estimators":[20,50,100,150,200],
    "min_samples_leaf":[50,100,200,300],
    "max_depth":[2,3,5,10]
}

grid = model_selection.GridSearchCV(model
                                    ,param_grid=params
                                    ,verbose=3
                                    ,scoring="roc_auc"
                                    ,cv=3)
grid
# %%
with mlflow.start_run():

    mlflow.sklearn.autolog()

    model_pipeline = pipeline.Pipeline(steps=[
        ("Remoção das features:",drop_features),
        ("Classificação TOP_PACK",top_pack_transformer),
        ("Imputação features categoricas",cat_nulos),
        ("Imputaçâo features numericas",imputation_0),
        ("Onehot Encoding",onehot),
        ("Algoritmo",grid)
    ])
    model_pipeline.fit(X_train,y_train)

    # SEMMA - ASSES

    y_pred_train = model_pipeline.predict(X_train)
    y_proba_train = model_pipeline.predict_proba(X_train)[:,1]

    y_pred_test = model_pipeline.predict(X_test)
    y_proba_test = model_pipeline.predict_proba(X_test)[:,1]

    acc_train = metrics.accuracy_score(y_train,y_pred_train)
    auc_train = metrics.roc_auc_score(y_train,y_proba_train)

    acc_test = metrics.accuracy_score(y_test,y_pred_test)
    auc_test = metrics.roc_auc_score(y_test,y_proba_test)

    mlflow.log_metrics({
        "acc_train":acc_train,
        "auc_train":auc_train,
        "acc_test":acc_test,
        "auc_test":auc_test,
    })
    roc_train = metrics.roc_curve(y_train,y_proba_train)
    roc_test = metrics.roc_curve(y_test,y_proba_test)

    plt.figure(figsize=(10,4))
    plt.plot(roc_train[0],roc_train[1])
    plt.plot(roc_test[0],roc_test[1])
    plt.xlabel("1 - Especificidade")
    plt.ylabel("Recall")
    plt.title("Curva ROC")
    plt.grid(True)
    plt.legend(f"AUC Treino: {auc_train:.4f}",
               f"AUC Teste: {acc_test:.4f}",
                )
    plt.savefig("curva_roc.png")
    mlflow.log_artifact("curva_roc.png")


