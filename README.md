# Estimação da nota geral do ENEM por escola

Projeto de TCC que estima, para cada escola, a **mediana das médias gerais individuais do ENEM 2025**. A média individual atribui o mesmo peso às quatro provas objetivas e à redação. A base reúne 8.592 escolas com pelo menos 50 participantes com as cinco notas válidas.

## Dados e integração

O projeto usa os resultados do ENEM, os microdados do Censo Escolar e os indicadores educacionais do Inep de 2025. A integração usa código da escola e ano. As junções preservam as escolas elegíveis do ENEM. A tabela `final` tem uma linha por escola, com cadastro, proporções da prova, características escolares e indicadores.

Para reconstruir o SQLite, coloque as fontes oficiais nestes caminhos:

| Fonte | Caminho local |
|---|---|
| Resultados ENEM 2025 | `data/enem/microdados/2025/RESULTADOS_2025.csv` |
| Censo Escolar | `data/censo_escolar/2025/dados/`, com `Tabela_Escola_2025_V2.csv`, `Tabela_Matricula_2025_V2.csv`, `Tabela_Docente_2025_V2.csv` e `Tabela_Turma_2025_V2.csv` |
| Indicadores por escola | `data/indicadores_educacionais/2025/`, com `AFD_2025_ESCOLAS.zip`, `ATU_2025_ESCOLAS.zip` e `TDI_2025_ESCOLAS.zip` |

## Execução

Use Python 3.12 em um ambiente virtual. Execute a partir da raiz ou use as células `# %%` dos scripts de análise e treino no VS Code.

```powershell
python -m pip install -r requirements-enem.txt
python src/montar_banco_escolar_2025.py
python src/Explore_final_2025.py
python src/train_regressores_escolas_2025.py
```

A montagem grava `data/enem_censo_escolar_2025.db`. O treinamento usa 70% das escolas para ajuste e 30% para teste, com semente 42. A busca usa três dobras no treino e escolhe pelo MAE médio. O preparo aplica o corte de mais de 50% de ausência na parcela de ajuste, imputa por tipo e codifica cada variável categórica em uma coluna com `TargetEncoder`. O experimento de sensibilidade a outliers está em `src/avaliar_outliers_escolas_2025.py`.

| Modelo | MAE no teste (pontos) | RMSE no teste (pontos) | R² |
|---|---:|---:|---:|
| Regressão Linear | 18,30 | 23,98 | 0,859 |
| Random Forest | 15,83 | 20,68 | 0,895 |
| XGBoost | 15,62 | 20,34 | 0,898 |

A avaliação é retrospectiva entre escolas de 2025 e inclui situações de redação da mesma edição. Uma avaliação de previsão para outra edição requer validação temporal.

## Resultados e modelo

- [Apresentação da defesa](output/apresentacoes/TCC_ENEM_2025_Defesa_Metodologia.pptx)
- [Métricas completas](output/regressao_escolas_final_2025/metricas.csv)
- [Inventário de variáveis](output/regressao_escolas_final_2025/inventario_features_defesa.csv)
- [Deploy com MLflow](output/regressao_escolas_final_2025/deploy.md)

O treino grava o registro MLflow em `data/mlflow_regressao_escolar.db`. Os dados brutos, os bancos locais, os ambientes virtuais e os binários dos modelos são gerados ou obtidos localmente.
