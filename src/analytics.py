# %%
# Importando Bibliotecas
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
# %%
# Carregando dados 
data = pd.read_csv("../data/Train.csv")
data.head()
# %%
# Modificando nomes das colunas
data = data.rename(columns={"REGION":"Regiao"
                            ,"TENURE":"Tempo_Cliente"
                            ,"MONTANT":"Valor_Gasto"
                            ,"FREQUENCE_RECH":"Freq_Recarga"
                            ,"REVENUE":"Receita"
                            ,"ARPU_SEGMENT":"Segmento_Arpu"
                            ,"FREQUENCE":"Freq"
                            ,"DATA_VOLUME":"Volume_Dados"
                            ,"ON_NET":"Chamadas_Mesma_Operadora"
                            ,"ORANGE":"Chamadas_Orange"
                            ,"TIGO":"Chamadas_Tigo"
                            ,"ZONE1":"Zona1"
                            ,"ZONE2":"Zona2"
                            ,"MRG":"Margem"
                            ,"REGULARITY":"Regularidade"
                            ,"TOP_PACK":"Top_Pack"
                            ,"FREQ_TOP_PACK":"Freq_top_pack"
                            ,"CHURN":"Churn"})
data.head()
# %%
# Verificando o shape 
data.shape
print(f"AMOSTRAS : {data.shape[0]}")
print(f"Colunas : {data.shape[1]}")
# %%
# Explorando informacoes gerais dos meus dados
data.info()
# %%
# Verificando dados nulos
col_nulos = data.columns[data.isna().mean()>0].to_list()
col_nulos
# %%
# Verificando os tipos de variaveis
data.dtypes
# %%
#Verificando Variaveis do tipo object
cat = data.columns[data.dtypes=='object'].to_list()
cat = cat[1:]
cat
# %%
# Verificando Variaveis do tipo num 
num = data.columns[data.dtypes != 'object'].to_list()
num = num[:-1]
num
# %%
# Verificando a distrubuicao das variaveis numericas
data[num].hist(figsize=(12,8))
plt.show()
# %%
# Verificando a distribuicao das variaveis categoricas
plt.figure(figsize=(10,5))

data["Regiao"].value_counts().plot(kind="bar")

plt.title("Distribuição da Região")
plt.ylabel("Quantidade")
plt.show()

# %%
plt.figure(figsize=(10,5))

data["Tempo_Cliente"].value_counts().plot(kind="bar")

plt.title("Distribuição da Tempo_Cliente")
plt.ylabel("Quantidade")
plt.show()
# %%
plt.figure(figsize=(10,5))

data["Margem"].value_counts().plot(kind="bar")

plt.title("Distribuição da Margem")
plt.ylabel("Quantidade")
plt.show()
# %%
plt.figure(figsize=(10,5))

data["Top_Pack"].value_counts().plot(kind="bar")

plt.title("Distribuição da Top_Pack")
plt.ylabel("Quantidade")
plt.show()
# %%
bivariada = data.groupby(by='Churn')[num].mean().T
bivariada
# %%
bivariada['rate'] = bivariada[0] / bivariada[1]
bivariada
# %%
bivariada_cat = data.groupby(by='Regiao')[['Churn']].mean().T
bivariada_cat.plot(kind='bar', figsize=(8,4))

plt.title("Taxa de Churn por Região")
plt.ylabel("Churn médio")
plt.show()

# %%
bivariada_cat = data.groupby(by='Tempo_Cliente')[['Churn']].mean().T


bivariada_cat.plot(kind='bar', figsize=(8,4))

plt.title("Taxa de Churn por Tempo de Cliente")
plt.ylabel("Churn médio")
plt.show()
# %%
bivariada_cat = data.groupby(by='Margem')[['Churn']].mean().T
bivariada_cat.plot(kind='bar', figsize=(8,4))

plt.title("Taxa de Churn por Margem")
plt.ylabel("Churn médio")
plt.show()

# %%
bivariada_cat = data.groupby(by='Top_Pack')[['Churn']].mean().sort_values('Churn', ascending=False).head(10)

bivariada_cat.plot(kind='bar', figsize=(10,5))

plt.title("Top 10 Pacotes com maior taxa de Churn")
plt.ylabel("Churn médio")
plt.show()
# %%
data.isna().mean()
# %%
for col in num:
    
    plt.figure(figsize=(6,4))
    
    plt.boxplot(data[col].dropna())
    
    plt.title(f"Boxplot - {col}")
    plt.ylabel(col)
    
    plt.show()
# %%
