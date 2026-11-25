# Reproduzir a inferência local no Windows

A versão 2 de ENEM_Escola_XGBoost foi escolhida pelo MAE de 15,57 pontos
na validação cruzada. Regressão Linear e Random Forest também foram registradas
na versão 2 de seus respectivos modelos.

No PowerShell, a partir da raiz do projeto:

```powershell
$env:PATH = "$(Resolve-Path tmp/venv-regressao/Scripts);$env:PATH"
$env:MLFLOW_TRACKING_URI = 'sqlite:///J:/ws-python/TCC-Churn/data/mlflow_regressao_escolar.db'
mlflow models serve -m models:/ENEM_Escola_XGBoost/2 --env-manager local --host 127.0.0.1 --port 5055
```

Em outro PowerShell, também na raiz:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5055/invocations -Method Post -ContentType 'application/json' -InFile output/regressao_escolas_final_2025/requisicao.json
```

A escola de teste 22027297 retornou 565,5380859375 pontos, valor igual ao
da pipeline antes do arredondamento do CSV. A mediana observada foi 571,65.
