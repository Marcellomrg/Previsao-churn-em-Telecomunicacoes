# Previsão do churn em telecomunicações
Desenvolver um modelo de machine learning que seja capaz de prever se uma cliente pode se tornar churn 

## Experimento XCL-Churn

O script `src/train_nature.py` gera métricas e gráficos comparativos por modelo.
O código permanece em células `# %%`, com a função de busca e a Pipeline original.
Os modelos comparados são XGBoost, CatBoost, LightGBM, XCL-Churn, Decision Tree,
Ridge, SGD, Random Forest, Stacking e Blending.

No ambiente `tcc_churn`, execute as células ou, na raiz do projeto:

```powershell
python src/train_nature.py
```

O padrão de registro segue o projeto DataScience-Loyalty-predict: `start_run`,
`autolog`, `log_metrics`, `plt.savefig` e `log_artifact`. Uma execução `XCL-Churn`
contém a Pipeline registrada pelo autolog, as métricas identificadas por modelo,
os gráficos e os demais modelos registrados com `mlflow.sklearn.log_model`.
Não são gerados arquivos `.pkl` avulsos.
O autolog pode incluir diagnósticos automáticos além das quatro métricas da
tabela comparativa. Não é feito novo treinamento para salvar os modelos.
O banco `mlflow_nature.db` é separado do `mlflow.db` já existente.
Para abrir a interface, execute na raiz do projeto:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow_nature.db
```

Abra http://127.0.0.1:5000 e procure o experimento `XCL-Churn`.
Os arquivos dos modelos ficam no armazenamento local de artefatos do MLflow;
é necessário preservar os artefatos e o banco, não somente o arquivo `.db`.

### Reutilização dos modelos

Os gráficos e tabelas ficam em `artifacts/nature/<id da execução>/` e no MLflow.
Para carregar um modelo, copie seu URI na interface do MLflow:

```python
modelo = mlflow.sklearn.load_model("models:/<id do modelo no MLflow>")
previsoes = modelo.predict(X_test)
```

O modelo recebe as colunas já tratadas, na mesma ordem de `X_test`, não o CSV
bruto. Os modelos registrados não incluem o pré-processamento: para novos dados,
é necessário preservar encoder, imputer, escaladores e a ordem das features.
Use somente `transform`, sem aplicar SMOTE ou refazer `fit` na previsão.
Artefatos de execuções anteriores foram preservados.

### Limites da replicação

Referência: https://www.nature.com/articles/s41598-025-34624-w

- Mantida a ordem de pré-processamento antes do split, escolhida para reproduzir
  o texto. Isso não constitui uma avaliação livre de vazamento.
- A busca usa um candidato por modelo com as configurações finais publicadas e
  cinco folds. Isso não reconstrói a busca original, cujas grades não foram
  publicadas. `scoring='roc_auc'` e 100 árvores no LightGBM são escolhas locais.
- Os pesos `[3, 2, 3]` seguem a ablação; outra seção menciona pesos iguais.
- As métricas de treino usam `X_train` após SMOTE, incluindo amostras sintéticas;
  o teste não é balanceado.
- Configurações dos comparadores, meta-modelo, holdout do blending e amostras
  de SHAP/LIME são escolhas locais onde faltam detalhes publicados.
- A comparação de pesos não escolhe o modelo pelo teste. Tempos de execução
  não são medidos. Não há garantia de resultados idênticos aos do artigo.
