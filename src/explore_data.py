# %%
import pandas as pd
import matplotlib.pyplot as plt
# %%
#CARREGANDO DADOS....
df = pd.read_csv("../data/Train.csv")
df.head()
# %%
# Separando features da target
features = df.columns[1:-1].to_list()
features
"""['REGION',
 'TENURE',
 'MONTANT',
 'FREQUENCE_RECH',
 'REVENUE',
 'ARPU_SEGMENT',
 'FREQUENCE',
 'DATA_VOLUME',
 'ON_NET',
 'ORANGE',
 'TIGO',
 'ZONE1',
 'ZONE2',
 'MRG',
 'REGULARITY',
 'TOP_PACK',
 'FREQ_TOP_PACK']"""

# %%
target = df.columns[-1]
target
""" CHURN """
# %%
# Observando as features numericas e categoricas
df.dtypes
"""user_id         object
REGION             object
TENURE             object
MONTANT           float64
FREQUENCE_RECH    float64
REVENUE           float64
ARPU_SEGMENT      float64
FREQUENCE         float64
DATA_VOLUME       float64
ON_NET            float64
ORANGE            float64
TIGO              float64
ZONE1             float64
ZONE2             float64
MRG                object
REGULARITY          int64
TOP_PACK           object
FREQ_TOP_PACK     float64
CHURN               int64
"""
# %%
# Conferindo valores duplicados
df.duplicated().sum()

# %%
cat = df.columns[df.dtypes == 'object'].to_list()
cat
"""['REGION', 'TENURE', 'MRG', 'TOP_PACK']"""
# %%
num = df.columns[df.dtypes != 'object'].to_list()
num
"""['MONTANT',
 'FREQUENCE_RECH',
 'REVENUE',
 'ARPU_SEGMENT',
 'FREQUENCE',
 'DATA_VOLUME',
 'ON_NET',
 'ORANGE',
 'TIGO',
 'ZONE1',
 'ZONE2',
 'REGULARITY',
 'FREQ_TOP_PACK',
 'CHURN']"""
# %%
# Separando features dos nulos
null = df.columns[df.isna().mean() > 0].to_list()

null_pct = df[null].isna().mean().sort_values(ascending=False) * 100

plt.figure(figsize=(10,6))
plt.bar(null_pct.index,null_pct.values)
plt.xlabel("Features")
plt.ylabel("Valores nulos(%)")
plt.xticks(rotation=90)

# %%
# Observando a distribuicao das features númericas 
features_numericas = df[num]

fig, axes = plt.subplots(5, 3, figsize=(15, 18))

for col, ax in zip(num, axes.flat):
    ax.hist(df[col].dropna(),bins="auto")
    ax.set_title(f"Distribuicao de {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Freq")
    ax.set_xscale("log")

plt.tight_layout()
plt.show()
# %%
# Observando a distribuicao das features categoricas
col = 'REGION'
plt.figure(figsize=(8, 5))
df[col].value_counts().plot(kind="bar")
plt.title(f"Distribuicao de {col}")
plt.xlabel(col)
plt.ylabel("Freq")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# %%
col = 'TENURE'
plt.figure(figsize=(8, 5))
df[col].value_counts().plot(kind="bar")
plt.title(f"Distribuicao de {col}")
plt.xlabel(col)
plt.ylabel("Freq")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# %%
col = 'MRG'
plt.figure(figsize=(8, 5))
df[col].value_counts().plot(kind="bar")
plt.title(f"Distribuicao de {col}")
plt.xlabel(col)
plt.ylabel("Freq")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# %%
col = 'TOP_PACK'
plt.figure(figsize=(8, 5))
df[col].value_counts().plot(kind="bar")
plt.title(f"Distribuicao de {col}")
plt.xlabel(col)
plt.ylabel("Freq")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# %%
#Verificando quantas categorias existem em TOP_PACK
df['TOP_PACK'].nunique()
"""Existe cerca de 140 categorias"""
# %%
# Fazendo uma analise bivariadas das features numericas
features_num = list(set(num) - set(['CHURN']))
bivariada = df.groupby(by=target)[features_num].mean().T
bivariada['ratio'] = bivariada[0] / bivariada[1]
bivariada = bivariada.sort_values(by='ratio',ascending=False)
bivariada
# %%
# Boxplot das features numéricas
fig, axes = plt.subplots(7, 2, figsize=(15, 20))

for col, ax in zip(features_num, axes.flat):
    ax.boxplot(df[col].dropna(), vert=False)
    ax.set_title(col)
    ax.set_xlabel("Valores")

plt.tight_layout()
plt.show()

# %%
# Matriz de correlacao entre as features numericas
correlacao = df[features_num].corr()

plt.figure(figsize=(12, 8))
plt.imshow(correlacao, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(correlacao.columns)), correlacao.columns, rotation=90)
plt.yticks(range(len(correlacao.columns)), correlacao.columns)
plt.title("Correlação entre as Features")
plt.tight_layout()

plt.show()

# %%
df["TOP_PACK"].unique()
# %%
df['FREQUENCE_RECH'].describe().apply(lambda x: f"{x:.2f}")
# %%
df["FREQUENCE_RECH"].quantile([0.95])
# %%
df["FREQUENCE_RECH"]