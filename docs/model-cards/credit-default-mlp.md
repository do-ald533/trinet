# Model card: credit-default-mlp

> **Uso educacional.** Trabalho de curso (CIC407, T2). Este modelo **não é** um modelo de decisão de crédito real e nada aqui é recomendação financeira: os dados são de Taiwan, 2005, e o limiar de decisão usa custos ilustrativos.

> **Candidato.** Exportado pelo notebook, mas ainda não promovido a `release`. A release exige versão >= 1.0.0 e quality gate aprovado (`make check-artifacts-release`).

Classificador de inadimplencia de cartao de credito (UCI #350, Taiwan 2005): MLP Normalization -> Dense(relu) -> Dropout -> Dense(2, softmax), treinado com class_weight. SEX e MARRIAGE nao sao features. Projeto educacional: nao e um modelo de decisao de credito real.

| Campo | Valor |
|---|---|
| Versão | `1.0.0` |
| Status | `candidate` |
| Tarefa | classificação tabular (`classification`) |
| Criado em | `2026-11-14T13:04:29+00:00` |
| Artefato | `artifacts/credit-default-mlp/1.0.0/` |
| Schema do contrato | `1.0` |

## 1. Uso pretendido

- Estimar a probabilidade de inadimplência no mês seguinte de um cliente de cartão de crédito a partir do limite, da escolaridade, da idade, do histórico de atraso (`pay_1` a `pay_6`) e dos valores de fatura e de pagamento dos últimos 6 meses.
- Demonstração didática da disciplina CIC407 (T2) e exemplo de serviço de inferência.

**Fora do escopo**

- Concessão, limite, preço ou cobrança de crédito reais.
- Qualquer população fora do conjunto de treino: clientes de cartão em Taiwan, 2005.
- Uso de sexo ou estado civil como insumo: eles não são entradas do modelo nem da API.

## 2. Dados

- Default of Credit Card Clients (UCI Machine Learning Repository #350): 30.000 clientes, 23 atributos e o alvo `Y` (inadimplência no mês seguinte). Sem NaN; inadimplência de 22,12% (6.636 clientes).
- Licença CC BY 4.0, DOI 10.24432/C55S3H. Citação: Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository.
- Problemas verificados em 14/11/2026: 35 linhas duplicadas (sem o ID), removidas antes do split; `EDUCATION` com os códigos 0, 5 e 6 não documentados (14, 280 e 51 linhas); `MARRIAGE = 0` (54 linhas); `PAY_*` com os valores -2 e 0 sem documentação; 590 `BILL_AMT1` negativos.
- Divisão 70% / 15% / 15% estratificada e sem duplicatas. O conjunto é baixado da UCI por quem roda o notebook e não é versionado.

Hash dos dados de treino (SHA-256): `45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0`

## 3. Pré-processamento

A API converte campos crus nomeados em um vetor float32 usando só as três primitivas fechadas do contrato (imputação por constante, one-hot por vocabulário e razão A/B). Campos desconhecidos, `null` sem imputação, valores não finitos e fora da faixa são rejeitados com HTTP 422.

**Campos de entrada**

| Campo | Tipo | Valores válidos | Se ausente ou null |
|---|---|---|---|
| `limit_bal` | numérico | >= 1 e <= 1e+07 | obrigatório (HTTP 422) |
| `age` | numérico | >= 18 e <= 120 | obrigatório (HTTP 422) |
| `pay_1` | numérico | >= -2 e <= 8 | obrigatório (HTTP 422) |
| `pay_2` | numérico | >= -2 e <= 8 | obrigatório (HTTP 422) |
| `pay_3` | numérico | >= -2 e <= 8 | obrigatório (HTTP 422) |
| `pay_4` | numérico | >= -2 e <= 8 | obrigatório (HTTP 422) |
| `pay_5` | numérico | >= -2 e <= 8 | obrigatório (HTTP 422) |
| `pay_6` | numérico | >= -2 e <= 8 | obrigatório (HTTP 422) |
| `bill_amt1` | numérico | >= -1e+07 e <= 1e+07 | obrigatório (HTTP 422) |
| `bill_amt2` | numérico | >= -1e+07 e <= 1e+07 | obrigatório (HTTP 422) |
| `bill_amt3` | numérico | >= -1e+07 e <= 1e+07 | obrigatório (HTTP 422) |
| `bill_amt4` | numérico | >= -1e+07 e <= 1e+07 | obrigatório (HTTP 422) |
| `bill_amt5` | numérico | >= -1e+07 e <= 1e+07 | obrigatório (HTTP 422) |
| `bill_amt6` | numérico | >= -1e+07 e <= 1e+07 | obrigatório (HTTP 422) |
| `pay_amt1` | numérico | >= 0 e <= 1e+07 | obrigatório (HTTP 422) |
| `pay_amt2` | numérico | >= 0 e <= 1e+07 | obrigatório (HTTP 422) |
| `pay_amt3` | numérico | >= 0 e <= 1e+07 | obrigatório (HTTP 422) |
| `pay_amt4` | numérico | >= 0 e <= 1e+07 | obrigatório (HTTP 422) |
| `pay_amt5` | numérico | >= 0 e <= 1e+07 | obrigatório (HTTP 422) |
| `pay_amt6` | numérico | >= 0 e <= 1e+07 | obrigatório (HTTP 422) |
| `education` | categórico | uma de: `graduate`, `university`, `high_school`, `other` | obrigatório (HTTP 422) |

**Vetor de entrada do modelo** (25 colunas, nesta ordem)

| Colunas | Primitiva | Definição |
|---|---|---|
| 0 | passthrough | valor numérico de `limit_bal` (já imputado) |
| 1 | passthrough | valor numérico de `age` (já imputado) |
| 2 | passthrough | valor numérico de `pay_1` (já imputado) |
| 3 | passthrough | valor numérico de `pay_2` (já imputado) |
| 4 | passthrough | valor numérico de `pay_3` (já imputado) |
| 5 | passthrough | valor numérico de `pay_4` (já imputado) |
| 6 | passthrough | valor numérico de `pay_5` (já imputado) |
| 7 | passthrough | valor numérico de `pay_6` (já imputado) |
| 8 | passthrough | valor numérico de `bill_amt1` (já imputado) |
| 9 | passthrough | valor numérico de `bill_amt2` (já imputado) |
| 10 | passthrough | valor numérico de `bill_amt3` (já imputado) |
| 11 | passthrough | valor numérico de `bill_amt4` (já imputado) |
| 12 | passthrough | valor numérico de `bill_amt5` (já imputado) |
| 13 | passthrough | valor numérico de `bill_amt6` (já imputado) |
| 14 | passthrough | valor numérico de `pay_amt1` (já imputado) |
| 15 | passthrough | valor numérico de `pay_amt2` (já imputado) |
| 16 | passthrough | valor numérico de `pay_amt3` (já imputado) |
| 17 | passthrough | valor numérico de `pay_amt4` (já imputado) |
| 18 | passthrough | valor numérico de `pay_amt5` (já imputado) |
| 19 | passthrough | valor numérico de `pay_amt6` (já imputado) |
| 20-23 | one_hot | one-hot de `education`, na ordem: `graduate`, `university`, `high_school`, `other` |
| 24 | ratio | `bill_amt1 / limit_bal` (HTTP 422 se o divisor for 0) |

A padronização das entradas fica dentro do grafo do modelo, não na API.

**Responsabilidade do chamador**

- `education` é categórica na API: mapeie os códigos da UCI (1 = `graduate`, 2 = `university`, 3 = `high_school`; 0, 4, 5 e 6 = `other`) antes de chamar.
- `limit_bal` precisa ser >= 1: a utilização `bill_amt1 / limit_bal` não aceita divisor zero, e a API responde 422 (`below_minimum`) a valores menores.

## 4. Arquitetura e runtimes

- Entrada: tensor `features` float32, forma (batch, 25).
- Saída: tensor `Identity:0` float32, forma (batch, 2): 2 classes: `no_default`, `default`.

Os runtimes seguem a ordem de preferência do manifesto (escada de fallback): a API usa o primeiro que passa na verificação de SHA-256 e no warm-up. Veja o [ADR 0001](../adr/0001-onnx-na-api-e-escada-de-fallback.md).

| Ordem | Runtime | Arquivo | SHA-256 | Detalhe |
|---|---|---|---|---|
| 1 | `onnx` | `model.onnx` | `efc156a4b067` | opset 17 |
| 2 | `numpy_mlp` | `weights.npz` | `54b8faf6b42a` | normalize(norm) > dense(dense_0, relu) > dense(dense_1, relu) > dense(out, softmax) |

A estrutura das camadas treináveis (larguras, profundidade) está no notebook, em `notebook/`, e, quando exportada, em `results.json`.

## 5. Métricas por split

| Split | accuracy | precision | recall | f1 | roc_auc | pr_auc | ks | brier |
|---|---|---|---|---|---|---|---|---|
| `validation` | 0.8177 | 0.6526 | 0.3746 | 0.4760 | 0.7812 | 0.5607 | 0.4313 | 0.1348 |
| `test` | 0.8201 | 0.6609 | 0.3827 | 0.4847 | 0.7714 | 0.5390 | 0.4151 | 0.1370 |

## 6. Quality gate de exportação

O notebook se recusa a escrever o artefato se o gate falhar; o limiar é a métrica de validação com uma margem definida depois do treino.

| Métrica | Split | Regra | Valor | Resultado |
|---|---|---|---|---|
| `accuracy` | `validation` | >= 0.7800 | 0.8177 | aprovado |

## 7. Training/serving skew

| Medida | Valor |
|---|---|
| Amostras comparadas | 8984 |
| Maior diferença absoluta (Keras x runtime) | 1.788e-07 |
| Concordância do top-1 | 1.0000 |
| Variação de acurácia (pontos percentuais) | não medido |

## 8. Limitações

- Classes desbalanceadas (22,12% de inadimplentes): a acurácia sozinha engana. O notebook reporta ROC-AUC, PR-AUC, KS, Brier e calibração, contra baselines (classe majoritária e regressão logística).
- O limiar de decisão foi escolhido na validação por custo (um falso negativo custa 5 falsos positivos). A premissa é ilustrativa, não calibrada com dados de um credor, e o limiar de 0,5 é reportado para comparação.
- Dados de um único mercado e de 2005: o comportamento de crédito, os produtos e o contexto regulatório mudaram.
- Códigos sem documentação (`EDUCATION` 0, 5, 6; `PAY_*` -2 e 0) foram mantidos ou agrupados como descrito no notebook, sem confirmação do significado original.
- As métricas valem para a distribuição dos dados de treino e teste; fora dela o desempenho não é garantido.
- A API expõe contadores por classe prevista, mas não histogramas por feature; o que dá e o que não dá para detectar de drift está no [ADR 0007](../adr/0007-monitoramento-de-drift-psi.md).

## 9. Considerações éticas e de uso

- **Atributos protegidos.** `SEX` e `MARRIAGE` ficam fora das features (princípio de não discriminação em crédito: a Regulation B / ECOA, nos EUA, trata sexo e estado civil como insumos vedados). Excluí-los não elimina o risco: idade, escolaridade e o histórico de pagamento podem funcionar como proxy de grupos protegidos.
- **Diagnóstico de equidade.** O notebook mede, no conjunto de teste e por `SEX`, a taxa de aprovação e a taxa de verdadeiros positivos de cada grupo. É um diagnóstico de limitações, não uma garantia de equidade, e os dados são de Taiwan, 2005.
- **Decisão automatizada.** A LGPD (Lei 13.709/2018), art. 20, prevê o direito do titular de pedir a revisão de decisões tomadas unicamente com base em tratamento automatizado, inclusive as de perfil de crédito. É um resumo do projeto: confirme o texto vigente em planalto.gov.br antes de citá-lo. Um uso real exigiria explicação, revisão humana e canal de contestação, que este serviço não oferece.
- **Privacidade.** O serviço não guarda as entradas: os logs registram apenas o SHA-256 do corpo da requisição.
- **Não é uma decisão de crédito.** O resultado é uma probabilidade estimada por um modelo didático e nunca deve ser usado para aprovar ou negar crédito a uma pessoa.

Contexto da escolha dos dados e dos alvos: [ADR 0010](../adr/0010-pivo-para-o-mercado-financeiro.md).

## 10. Proveniência

| Campo | Valor |
|---|---|
| Git SHA | não registrado |
| Conjunto de dados | Default of Credit Card Clients (UCI #350, CC BY 4.0, DOI 10.24432/C55S3H) |
| SHA-256 dos dados | `45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0` |
| Seed | 42 |
| Python | 3.13.13 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| Hardware | 11th Gen Intel(R) Core(TM) i7-11370H @ 3.30GHz \| 8 threads \| 33 GB RAM \| GPU: nao (CPU) |
| Golden samples | `golden.json`: 220 casos (registros crus), 220 rotulados |
| SHA-256 de `model.onnx` | `efc156a4b067bda13b971a7c08184e3ad6f58edbb81fa25c1975ecf53a81c719` |
| SHA-256 de `weights.npz` | `54b8faf6b42afa00eb3496007b6ab93e3747dc5e69edd123e29c1bc68eabbabd` |

## 11. Resultados adicionais (`results.json`)

### `ablation_mean_over_3_seeds`

| class_weight | depth | width | val_pr_auc | val_roc_auc | val_f1_05 | val_f1_best | params | epochs |
|---|---|---|---|---|---|---|---|---|
| False | 2 | 128 | 0.5563 | 0.7785 | 0.4672 | 0.5408 | 11714.00 | 33.0000 |
| True | 1 | 16 | 0.5363 | 0.7705 | 0.5363 | 0.5415 | 450.00 | 40.0000 |
| True | 1 | 32 | 0.5366 | 0.7733 | 0.5383 | 0.5476 | 898.00 | 40.0000 |
| True | 1 | 64 | 0.5415 | 0.7776 | 0.5360 | 0.5477 | 1794.00 | 39.0000 |
| True | 1 | 128 | 0.5431 | 0.7773 | 0.5369 | 0.5472 | 3586.00 | 27.3333 |
| True | 2 | 16 | 0.5385 | 0.7729 | 0.5355 | 0.5405 | 570.00 | 40.0000 |
| True | 2 | 32 | 0.5406 | 0.7760 | 0.5386 | 0.5422 | 1394.00 | 38.0000 |
| True | 2 | 64 | 0.5456 | 0.7786 | 0.5364 | 0.5445 | 3810.00 | 31.6667 |
| True | 2 | 128 | 0.5484 | 0.7783 | 0.5337 | 0.5442 | 11714.00 | 28.0000 |
| True | 3 | 16 | 0.5364 | 0.7681 | 0.5273 | 0.5359 | 642.00 | 36.3333 |
| True | 3 | 32 | 0.5423 | 0.7759 | 0.5353 | 0.5403 | 1514.00 | 34.3333 |
| True | 3 | 64 | 0.5450 | 0.7770 | 0.5377 | 0.5433 | 4306.00 | 33.3333 |
| True | 3 | 128 | 0.5489 | 0.7785 | 0.5358 | 0.5455 | 13730.00 | 26.6667 |

### `architecture`

```json
{
  "class_weight": false,
  "depth": 2,
  "dropout": 0.2,
  "features": [
    "limit_bal",
    "age",
    "pay_1",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6",
    "education=graduate",
    "education=university",
    "education=high_school",
    "education=other",
    "bill_utilisation"
  ],
  "width": 128
}
```

### `data`

| Chave | Valor |
|---|---|
| `csv_sha256` | 45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0 |
| `train` | 20961 |
| `val` | 4492 |
| `test` | 4492 |

### `decision`

| Chave | Valor |
|---|---|
| `argmax_threshold` | 0.5000 |
| `cost_threshold_recommended` | 0.1750 |
| `cost_fn_over_fp` | 5.0000 |
| `note` | limiar por custo escolhido na validacao (premissa ilustrativa) |

### `disclaimer`

Projeto educacional (T2 CIC407). Nao e um modelo de decisao de credito real; custos do limiar sao ilustrativos.

### `extra_gate`

| Chave | Valor |
|---|---|
| `metric` | pr_auc |
| `split` | validation |
| `threshold` | 0.5500 |
| `value` | 0.5607 |
| `higher_is_better` | True |

### `fairness_by_sex_test`

| grupo | clientes | inadimplência real | taxa de aprovação | TPR (revocação) | FPR | PR-AUC |
|---|---|---|---|---|---|---|
| masculino | 1757 | 0.2362 | 0.4952 | 0.7880 | 0.4173 | 0.5398 |
| feminino | 2735 | 0.2113 | 0.5371 | 0.7457 | 0.3871 | 0.5397 |

### `golden_cases`

220

### `manifest_vs_notebook_features_max_abs_diff`

0.0000

### `notebook_test_table`

```json
{
  "MLP (0,5)": {
    "Brier": 0.13699203729629517,
    "F1 (1)": 0.4846938775510204,
    "KS": 0.4150597480448305,
    "PR-AUC": 0.538972384427955,
    "ROC-AUC": 0.7714157720793194,
    "acurácia": 0.8201246660730187,
    "custo/cliente": 0.7257346393588602,
    "limiar": 0.5,
    "precisão (1)": 0.6608695652173913,
    "revocação (1)": 0.3826787512588117
  },
  "MLP (limiar por custo)": {
    "Brier": 0.13699203729629517,
    "F1 (1)": 0.4818817546090273,
    "KS": 0.4150597480448305,
    "PR-AUC": 0.538972384427955,
    "ROC-AUC": 0.7714157720793194,
    "acurácia": 0.6371326803205699,
    "custo/cliente": 0.5721282279608192,
    "limiar": 0.175,
    "precisão (1)": 0.35206688341848585,
    "revocação (1)": 0.7633434038267876
  },
  "maioria (sempre adimplente)": {
    "Brier": 0.22105966162065896,
    "F1 (1)": 0.0,
    "KS": 0.0,
    "PR-AUC": 0.22105966162065896,
    "ROC-AUC": 0.5,
    "acurácia": 0.7789403383793411,
    "custo/cliente": 1.1052983081032948,
    "limiar": 0.5,
    "precisão (1)": 0.0,
    "revocação (1)": 0.0
  },
  "regressão logística (0,5)": {
    "Brier": 0.1462913304567337,
    "F1 (1)": 0.3548872180451128,
    "KS": 0.3685993437342334,
    "PR-AUC": 0.48972808908637677,
    "ROC-AUC": 0.7185540279527427,
    "acurácia": 0.808993766696349,
    "custo/cliente": 0.865093499554764,
    "limiar": 0.5,
    "precisão (1)": 0.7002967359050445,
    "revocação (1)": 0.23766364551863042
  },
  "regressão logística (custo)": {
    "Brier": 0.1462913304567337,
    "F1 (1)": 0.4827024827024827,
    "KS": 0.3685993437342334,
    "PR-AUC": 0.48972808908637677,
    "ROC-AUC": 0.7185540279527427,
    "acurácia": 0.7170525378450578,
    "custo/cliente": 0.6391362422083704,
    "limiar": 0.23,
    "precisão (1)": 0.40505464480874315,
    "revocação (1)": 0.5971802618328298
  }
}
```

### `onnx`

```json
{
  "bytes": 49903,
  "input": {
    "name": "features",
    "shape": [
      null,
      25
    ],
    "type": "tensor(float)"
  },
  "ir_version": 8,
  "keras_export_error": null,
  "method": "keras.export(format='onnx') [tf2onnx]",
  "ops": [
    "Add",
    "MatMul",
    "Mul",
    "Relu",
    "Softmax"
  ],
  "opset": 17,
  "output": {
    "name": "Identity:0",
    "shape": [
      null,
      2
    ],
    "type": "tensor(float)"
  }
}
```

### `parity_max_abs_diff`

| Chave | Valor |
|---|---|
| `clone` | 1.79e-07 |
| `numpy` | 2.00e-07 |
| `onnx` | 1.79e-07 |

### `timings_seconds`

| Chave | Valor |
|---|---|
| `train` | 10.7512 |
| `export` | 1.0512 |
| `notebook_part_10` | 210.67 |

### `versions`

| Chave | Valor |
|---|---|
| `python` | 3.13.13 |
| `tensorflow` | 2.21.0 |
| `keras` | 3.15.1 |
| `onnx` | 1.23.2 |
| `onnxruntime` | 1.31.1 |
| `numpy` | 2.5.4 |

---

Gerado por `tools/gen_model_card.py` a partir de `artifacts/credit-default-mlp/1.0.0/manifest.json`. Não edite à mão: rode `uv run --package trinet-contract python tools/gen_model_card.py`.
