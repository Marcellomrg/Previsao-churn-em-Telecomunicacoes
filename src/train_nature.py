# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection,preprocessing,experimental,impute,linear_model,ensemble
from sklearn.experimental import enable_iterative_imputer
from boruta import BorutaPy
# %%
# IMPORTANDO DADOS....
df = pd.read_csv("../data/Train.csv")
df
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
nulos = df.columns[df.isna().sum() > 1].to_list()
nulos

# %%
imputer = impute.IterativeImputer(estimator=linear_model.BayesianRidge(),
                                  initial_strategy='median',
                                  max_iter=20,
                                  tol=0.00001,
                                  random_state=42)
imputer
# %%
df_imputed = pd.DataFrame(imputer.fit_transform(df),
                          columns=df.columns,
                          index=df.index)

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
features = df_minmax.columns[0:-1].tolist()

target = df_minmax.columns[-1]

target
# %%
X = df_minmax.drop(columns=[target])

y = df_minmax[target]

y
# %%
rfc = ensemble.RandomForestClassifier(n_estimators=100,
                                      random_state=42,
                                      n_jobs=-1)
rfc
# %%
boruta = BorutaPy(estimator=rfc,
                  n_estimators=100,
                  max_iter=100,
                  alpha= 0.005,
                  random_state=42,
                  verbose=2)
boruta
# %%
boruta.fit(X,y)
# %%
selected_features = X.columns[boruta.support_].tolist()

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
