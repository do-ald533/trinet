# Model cards

Um card por modelo, gerado a partir do artefato de maior versão em `artifacts/`.

| Modelo | Tarefa | Versão | Status | Métricas no teste | Quality gate | Card |
|---|---|---|---|---|---|---|
| `candlestick-cnn` | classificação de imagens | `1.0.0` | `candidate` | accuracy 0.5249; macro_f1 0.4453 | accuracy >= 0.4900 (ok) | [card](candlestick-cnn.md) |
| `credit-default-mlp` | classificação tabular | `1.0.0` | `candidate` | accuracy 0.8201; precision 0.6609; recall 0.3827; f1 0.4847; roc_auc 0.7714; pr_auc 0.5390; ks 0.4151; brier 0.1370 | accuracy >= 0.7800 (ok) | [card](credit-default-mlp.md) |
| `volatility-mlp` | regressão tabular | `1.0.0` | `candidate` | rmse 0.1332; mae 0.0741; mae_over_mean 0.3876; rmse_over_mean 0.6971; qlike 0.8707; r2_log 0.2331; bias -0.0231 | rmse <= 0.1351 (ok) | [card](volatility-mlp.md) |

---

Gerado por `tools/gen_model_card.py`. Rode `uv run --package trinet-contract python tools/gen_model_card.py` para atualizar.
