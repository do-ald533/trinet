# Model card: volatility-mlp

> **Uso educacional.** Trabalho de curso (CIC407, T2). Nada neste modelo ou na API é recomendação de investimento, sinal de compra ou venda, nem base para decisões financeiras reais.

> **Candidato.** Exportado pelo notebook, mas ainda não promovido a `release`. A release exige versão >= 1.0.0 e quality gate aprovado (`make check-artifacts-release`).

Regressor da volatilidade realizada dos proximos 5 pregoes de acoes do S&P 500 (anualizada, em fracao): MLP sobre ln(y) padronizado, com clone de inferencia Normalization(invert) -> exp. Entrada: features ja calculadas no fechamento de t. Projeto educacional: nao e recomendacao de investimento.

| Campo | Valor |
|---|---|
| Versão | `1.0.0` |
| Status | `candidate` |
| Tarefa | regressão tabular (`regression`) |
| Criado em | `2026-11-14T13:04:30+00:00` |
| Artefato | `artifacts/volatility-mlp/1.0.0/` |
| Schema do contrato | `1.0` |

## 1. Uso pretendido

- Estimar a volatilidade realizada dos próximos 5 pregões de uma ação do S&P 500, anualizada e em fração (0,20 = 20% ao ano), a partir de 20 features calculadas pelo chamador no fechamento do dia `t`.
- Demonstração didática da disciplina CIC407 (T2). Compara-se com três baselines: a média do treino, a persistência (`rv_5`) e um HAR linear.

**Fora do escopo**

- Recomendação de investimento, sinal de compra ou venda, dimensionamento de posição ou precificação de opções.
- Prever a direção do preço: o modelo estima a magnitude das variações, não o sinal.
- Ativos fora do S&P 500, ou regimes de mercado muito diferentes de 2013-2018.

## 2. Dados

- S&P 500, 5 anos (Kaggle `camnugent/sandp500`): 619.040 linhas e 7 colunas (`date`, `open`, `high`, `low`, `close`, `volume`, `Name`), 505 tickers, de 2013-02-08 a 2018-02-07 (1.259 pregões). Preços ajustados a splits.
- A licença e a origem do arquivo **não foram verificadas**: o CSV nunca é versionado nem redistribuído (o download via `kagglehub` é anônimo, feito por quem roda o notebook).
- Limpeza (PLANO v1.2, V3): remove linhas com NaN em OHLC, volume 0, `high < low` e `close` fora de [`low`, `high`]. Uma amostra em `t` exige a janela [t-59, t+5] contígua e nenhum |retorno log| > 0,4 nela (descarta artefatos de ações corporativas e 3 eventos reais).
- Divisão cronológica por data (todos os tickers juntos): 70% / 15% / 15% dos pregões, com embargo (purge) de 5 pregões em cada fronteira, porque o alvo olha 5 dias à frente. O teste é tocado uma vez.

Hash dos dados de treino (SHA-256): `6aea253cd19de60b568143991aaf1fa482456565c389205658d236e595e716cf`

## 3. Pré-processamento

A API converte campos crus nomeados em um vetor float32 usando só as três primitivas fechadas do contrato (imputação por constante, one-hot por vocabulário e razão A/B). Campos desconhecidos, `null` sem imputação, valores não finitos e fora da faixa são rejeitados com HTTP 422.

**Campos de entrada**

| Campo | Tipo | Valores válidos | Se ausente ou null |
|---|---|---|---|
| `ret_1d` | numérico | >= -0.4 e <= 0.4 | obrigatório (HTTP 422) |
| `ret_2d` | numérico | >= -0.4 e <= 0.4 | obrigatório (HTTP 422) |
| `ret_3d` | numérico | >= -0.4 e <= 0.4 | obrigatório (HTTP 422) |
| `ret_4d` | numérico | >= -0.4 e <= 0.4 | obrigatório (HTTP 422) |
| `ret_5d` | numérico | >= -0.4 e <= 0.4 | obrigatório (HTTP 422) |
| `rv_5` | numérico | >= 0 e <= 10 | obrigatório (HTTP 422) |
| `rv_10` | numérico | >= 0 e <= 10 | obrigatório (HTTP 422) |
| `rv_20` | numérico | >= 1e-06 e <= 10 | obrigatório (HTTP 422) |
| `rv_60` | numérico | >= 0 e <= 10 | obrigatório (HTTP 422) |
| `pk_5` | numérico | >= 0 e <= 10 | obrigatório (HTTP 422) |
| `pk_20` | numérico | >= 0 e <= 10 | obrigatório (HTTP 422) |
| `range_5` | numérico | >= 0 e <= 1 | obrigatório (HTTP 422) |
| `volume_ratio` | numérico | >= 1e-06 e <= 1000 | obrigatório (HTTP 422) |
| `log_dollar_volume` | numérico | >= 0 e <= 40 | obrigatório (HTTP 422) |
| `gap` | numérico | >= -1 e <= 10 | obrigatório (HTTP 422) |
| `mom_20` | numérico | >= -1 e <= 10 | obrigatório (HTTP 422) |
| `mom_60` | numérico | >= -1 e <= 10 | obrigatório (HTTP 422) |
| `drawdown_60` | numérico | >= -1 e <= 0 | obrigatório (HTTP 422) |
| `mkt_rv_20` | numérico | >= 0 e <= 5 | obrigatório (HTTP 422) |
| `dow` | categórico | uma de: `mon`, `tue`, `wed`, `thu`, `fri` | obrigatório (HTTP 422) |

**Vetor de entrada do modelo** (25 colunas, nesta ordem)

| Colunas | Primitiva | Definição |
|---|---|---|
| 0 | passthrough | valor numérico de `ret_1d` (já imputado) |
| 1 | passthrough | valor numérico de `ret_2d` (já imputado) |
| 2 | passthrough | valor numérico de `ret_3d` (já imputado) |
| 3 | passthrough | valor numérico de `ret_4d` (já imputado) |
| 4 | passthrough | valor numérico de `ret_5d` (já imputado) |
| 5 | passthrough | valor numérico de `rv_5` (já imputado) |
| 6 | passthrough | valor numérico de `rv_10` (já imputado) |
| 7 | passthrough | valor numérico de `rv_20` (já imputado) |
| 8 | passthrough | valor numérico de `rv_60` (já imputado) |
| 9 | passthrough | valor numérico de `pk_5` (já imputado) |
| 10 | passthrough | valor numérico de `pk_20` (já imputado) |
| 11 | passthrough | valor numérico de `range_5` (já imputado) |
| 12 | passthrough | valor numérico de `volume_ratio` (já imputado) |
| 13 | passthrough | valor numérico de `log_dollar_volume` (já imputado) |
| 14 | passthrough | valor numérico de `gap` (já imputado) |
| 15 | passthrough | valor numérico de `mom_20` (já imputado) |
| 16 | passthrough | valor numérico de `mom_60` (já imputado) |
| 17 | passthrough | valor numérico de `drawdown_60` (já imputado) |
| 18 | passthrough | valor numérico de `mkt_rv_20` (já imputado) |
| 19-23 | one_hot | one-hot de `dow`, na ordem: `mon`, `tue`, `wed`, `thu`, `fri` |
| 24 | ratio | `rv_5 / rv_20` (HTTP 422 se o divisor for 0) |

A padronização das entradas fica dentro do grafo do modelo, não na API.

A saída já sai desnormalizada, na unidade do alvo (o alvo de treino é ln(y) padronizado e o modelo termina em `exp`).

**Responsabilidade do chamador**

- A API recebe **features já calculadas**: a engenharia de séries temporais é do chamador (escolha registrada no [ADR 0010](../adr/0010-pivo-para-o-mercado-financeiro.md)).
- Toda feature em `t` usa só dados até o fechamento de `t`. Definições: `ret_1d` a `ret_5d` são log-retornos defasados; `rv_5`, `rv_10`, `rv_20` e `rv_60` são volatilidades realizadas anualizadas; `pk_5` e `pk_20`, estimadores de Parkinson (high/low); `range_5`, a média de (high - low) / close; `volume_ratio`, o volume sobre a média de 20 dias; `gap`, open / close anterior - 1; `mkt_rv_20`, a volatilidade média do mercado (proxy de VIX sobre os 505 tickers). A fonte de verdade das fórmulas é a rede 2 do notebook (`notebook/parts/20_volatility.py`).
- A razão `rv_5 / rv_20` exige `rv_20 >= 1e-6`: abaixo disso (inclusive zero) a API responde 422 (`below_minimum`).

## 4. Arquitetura e runtimes

- Entrada: tensor `features` float32, forma (batch, 25).
- Saída: tensor `Identity:0` float32, forma (batch, 1): valor de `realized_volatility_5d` em volatilidade anualizada (fração).

Os runtimes seguem a ordem de preferência do manifesto (escada de fallback): a API usa o primeiro que passa na verificação de SHA-256 e no warm-up. Veja o [ADR 0001](../adr/0001-onnx-na-api-e-escada-de-fallback.md).

| Ordem | Runtime | Arquivo | SHA-256 | Detalhe |
|---|---|---|---|---|
| 1 | `onnx` | `model.onnx` | `4bdc862c682e` | opset 17 |
| 2 | `numpy_mlp` | `weights.npz` | `23b0beac774a` | normalize(norm) > dense(dense_0, relu) > dense(out, linear) > denormalize(target) > exp |

A estrutura das camadas treináveis (larguras, profundidade) está no notebook, em `notebook/`, e, quando exportada, em `results.json`.

## 5. Métricas por split

| Split | rmse | mae | mae_over_mean | rmse_over_mean | qlike | r2_log | bias |
|---|---|---|---|---|---|---|---|
| `validation` | 0.1286 | 0.0763 | 0.3955 | 0.6664 | 0.7996 | 0.2094 | -0.0150 |
| `test` | 0.1332 | 0.0741 | 0.3876 | 0.6971 | 0.8707 | 0.2331 | -0.0231 |

## 6. Quality gate de exportação

O notebook se recusa a escrever o artefato se o gate falhar; o limiar é a métrica de validação com uma margem definida depois do treino.

| Métrica | Split | Regra | Valor | Resultado |
|---|---|---|---|---|
| `rmse` | `validation` | <= 0.1351 | 0.1286 | aprovado |

## 7. Training/serving skew

| Medida | Valor |
|---|---|
| Amostras comparadas | 181404 |
| Maior diferença absoluta (Keras x runtime) | 3.576e-07 |
| Concordância do top-1 | não se aplica (regressão) |
| Variação de acurácia (pontos percentuais) | não se aplica (regressão) |

## 8. Limitações

- Viés de sobrevivência: só entram as ações que eram constituintes do índice em fev/2018. Empresas que saíram, faliram ou foram compradas antes estão ausentes.
- Período curto: cinco anos de um único regime de mercado, com a alta de volatilidade de jan-fev/2018 **dentro do teste**. Ela é um evento fora do que o treino viu, então o erro nesse trecho é esperado e deve ser lido separado (o notebook mostra o erro por sub-período).
- O preço de fechamento ajustado vem de uma fonte não auditada; 14 retornos com |log| > 0,4 indicam ajustes de desdobramento e cisão incompletos (alguns eventos são reais).
- Se o chamador calcular uma feature de forma diferente da do notebook, surge *training-serving skew* que o golden da API **não detecta**: o golden cobre só o caminho da API (validação, razão, one-hot e grafo).
- O alvo de treino é ln(y) padronizado e a saída passa por `exp`: isso estima a mediana condicional da volatilidade, que tende a ficar abaixo da média quando o erro em log tem variância.
- Não há classe de rejeição nem intervalo de confiança: a saída é um número único.
- As métricas valem para a distribuição dos dados de treino e teste; fora dela o desempenho não é garantido.
- A API expõe contadores por classe prevista, mas não histogramas por feature; o que dá e o que não dá para detectar de drift está no [ADR 0007](../adr/0007-monitoramento-de-drift-psi.md).

## 9. Considerações éticas e de uso

- **Não é recomendação.** A volatilidade prevista não é o risco total de uma posição e não indica retorno futuro. Não use o resultado para operar.
- **Vieses declarados.** Sobrevivência e regime único (veja as limitações) tendem a tornar as métricas otimistas para qualquer uso fora da amostra.
- **Dados.** Não há dados pessoais. Os preços não são redistribuídos por terem origem e licença não verificadas.

Contexto da escolha dos dados e dos alvos: [ADR 0010](../adr/0010-pivo-para-o-mercado-financeiro.md).

## 10. Proveniência

| Campo | Valor |
|---|---|
| Git SHA | não registrado |
| Conjunto de dados | S&P 500 5 anos (Kaggle camnugent/sandp500, all_stocks_5yr.csv; licenca nao verificada, dados nao redistribuidos) |
| SHA-256 dos dados | `6aea253cd19de60b568143991aaf1fa482456565c389205658d236e595e716cf` |
| Seed | 42 |
| Python | 3.13.13 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| Hardware | 11th Gen Intel(R) Core(TM) i7-11370H @ 3.30GHz \| 8 threads \| 33 GB RAM \| GPU: nao (CPU) |
| Golden samples | `golden.json`: 250 casos (registros crus), 200 rotulados |
| SHA-256 de `model.onnx` | `4bdc862c682eb49f894c1dafab131c65ea58713a6d3f6beec6f682433b1f5c61` |
| SHA-256 de `weights.npz` | `23b0beac774a0a4132011a1dc0ab4b4d84f0720d923ead4beeab4c269fa750d6` |

## 11. Resultados adicionais (`results.json`)

### `ablation_mean_over_3_seeds`

| loss | depth | width | val_QLIKE | val_RMSE | val_MAE | val_R2log | params | epochs |
|---|---|---|---|---|---|---|---|---|
| huber | 1 | 32 | 0.8064 | 0.1289 | 0.0770 | 0.1979 | 865.00 | 25.3333 |
| huber | 1 | 64 | 0.8101 | 0.1289 | 0.0769 | 0.1990 | 1729.00 | 21.0000 |
| huber | 1 | 128 | 0.8071 | 0.1288 | 0.0768 | 0.2018 | 3457.00 | 19.0000 |
| huber | 2 | 32 | 0.8127 | 0.1291 | 0.0771 | 0.1970 | 1377.00 | 24.3333 |
| huber | 2 | 64 | 0.8215 | 0.1294 | 0.0771 | 0.1933 | 3777.00 | 10.6667 |
| huber | 2 | 128 | 0.8224 | 0.1293 | 0.0769 | 0.1990 | 11649.00 | 8.3333 |
| huber | 3 | 32 | 0.8157 | 0.1295 | 0.0773 | 0.1926 | 1505.00 | 16.0000 |
| huber | 3 | 64 | 0.8152 | 0.1294 | 0.0771 | 0.1945 | 4289.00 | 9.0000 |
| huber | 3 | 128 | 0.8305 | 0.1295 | 0.0770 | 0.1974 | 13697.00 | 8.3333 |

### `architecture`

```json
{
  "depth": 1,
  "features": [
    "ret_1d",
    "ret_2d",
    "ret_3d",
    "ret_4d",
    "ret_5d",
    "rv_5",
    "rv_10",
    "rv_20",
    "rv_60",
    "pk_5",
    "pk_20",
    "range_5",
    "volume_ratio",
    "log_dollar_volume",
    "gap",
    "mom_20",
    "mom_60",
    "drawdown_60",
    "mkt_rv_20",
    "dow=mon",
    "dow=tue",
    "dow=wed",
    "dow=thu",
    "dow=fri",
    "rv_ratio"
  ],
  "huber_delta_z": 1.098,
  "loss": "mse",
  "target_mu_ln": -1.6999622161210115,
  "target_sigma_ln": 0.5631523086074486,
  "width": 32
}
```

### `baselines_test`

```json
{
  "HAR linear": {
    "MAE": 0.07661465965344658,
    "MAE/média": 0.4009408811686054,
    "QLIKE": 0.8835699862563847,
    "RMSE": 0.13543883116171887,
    "RMSE/média": 0.7087803372886571,
    "R² (log)": 0.1856847611128547,
    "viés": -0.020059650705928855
  },
  "média do treino": {
    "MAE": 0.09852226480339325,
    "MAE/média": 0.5155880590435038,
    "QLIKE": 0.8377956529608512,
    "RMSE": 0.14586209252086574,
    "RMSE/média": 0.7633274907040807,
    "R² (log)": -0.26271197047415273,
    "viés": 0.0239848617534952
  },
  "persistência (rv_5)": {
    "MAE": 0.10217691695781375,
    "MAE/média": 0.5347136345115164,
    "QLIKE": 1.9230043140353066,
    "RMSE": 0.17482723718102422,
    "RMSE/média": 0.9149082805392234,
    "R² (log)": -0.34197199870931283,
    "viés": -0.0036774156809198707
  }
}
```

### `data`

```json
{
  "cleaning": {
    "close_fora_de_low_high": 9,
    "high_menor_que_low": 1,
    "linhas_brutas": 619040,
    "linhas_limpas": 619020,
    "linhas_removidas": 20,
    "nan_ohlc": 11,
    "pregoes": 1259,
    "tickers_brutos": 505,
    "tickers_limpos": 505,
    "volume_zero": 4
  },
  "samples": {
    "test": 89792,
    "train": 398738,
    "val": 91612
  },
  "sp500_rows": 619020
}
```

### `disclaimer`

Projeto educacional (T2 CIC407). Nao e recomendacao de investimento.

### `error_by_month_test`

```json
{
  "2017-05": {
    "MAE MLP": 0.058432441418206735,
    "RMSE HAR linear": 0.1031197428766384,
    "RMSE MLP": 0.09911687117743104,
    "RMSE média do treino": 0.12228350827446707,
    "RMSE persistência (rv_5)": 0.15326977607570627,
    "amostras": 5000,
    "viés MLP": -0.0014629252716705367,
    "y médio": 0.16383575595317337
  },
  "2017-06": {
    "MAE MLP": 0.060176110237103786,
    "RMSE HAR linear": 0.1021517988266171,
    "RMSE MLP": 0.09849344652501876,
    "RMSE média do treino": 0.11883791417366263,
    "RMSE persistência (rv_5)": 0.12971504707411144,
    "amostras": 11003,
    "viés MLP": -0.018925891495716603,
    "y médio": 0.1851922213706845
  },
  "2017-07": {
    "MAE MLP": 0.07866824665851084,
    "RMSE HAR linear": 0.14527077518732961,
    "RMSE MLP": 0.1432939912077365,
    "RMSE média do treino": 0.15499704569432166,
    "RMSE persistência (rv_5)": 0.18545760535725178,
    "amostras": 10020,
    "viés MLP": -0.022821243922752844,
    "y médio": 0.18625797578023656
  },
  "2017-08": {
    "MAE MLP": 0.06676682554023149,
    "RMSE HAR linear": 0.12362992813660481,
    "RMSE MLP": 0.12136171672498017,
    "RMSE média do treino": 0.1370449031228869,
    "RMSE persistência (rv_5)": 0.17339657259253488,
    "amostras": 11523,
    "viés MLP": -0.013828406533642888,
    "y médio": 0.18045075301460453
  },
  "2017-09": {
    "MAE MLP": 0.07331400594681109,
    "RMSE HAR linear": 0.1693305671938946,
    "RMSE MLP": 0.16850149288172295,
    "RMSE média do treino": 0.17890848168443227,
    "RMSE persistência (rv_5)": 0.22026159636575735,
    "amostras": 10022,
    "viés MLP": -0.022004828536142336,
    "y médio": 0.18582783889482002
  },
  "2017-10": {
    "MAE MLP": 0.07919481977352602,
    "RMSE HAR linear": 0.14948083127439815,
    "RMSE MLP": 0.14600739974027918,
    "RMSE média do treino": 0.158491240828385,
    "RMSE persistência (rv_5)": 0.17837269447563367,
    "amostras": 11054,
    "viés MLP": -0.02479194625727618,
    "y médio": 0.1857587676064696
  },
  "2017-11": {
    "MAE MLP": 0.08213731279518849,
    "RMSE HAR linear": 0.14584633984475798,
    "RMSE MLP": 0.14244439884347343,
    "RMSE média do treino": 0.15483618644627486,
    "RMSE persistência (rv_5)": 0.19937189064360591,
    "amostras": 10545,
    "viés MLP": -0.02832107928472186,
    "y médio": 0.20440946584893932
  },
  "2017-12": {
    "MAE MLP": 0.06518692134237425,
    "RMSE HAR linear": 0.09576136286696268,
    "RMSE MLP": 0.09396644114351597,
    "RMSE média do treino": 0.11656286376780982,
    "RMSE persistência (rv_5)": 0.13315072796726032,
    "amostras": 10060,
    "viés MLP": 0.0012134362504250128,
    "y médio": 0.1741196936374533
```

### `error_by_regime_test`

```json
{
  "alta": {
    "MAE HAR linear": 0.16425702111175922,
    "MAE MLP": 0.16209289424027296,
    "RMSE HAR linear": 0.2489598718579588,
    "RMSE MLP": 0.24733977401147578,
    "amostras": 22534,
    "viés HAR linear": -0.1570858359513852,
    "viés MLP": -0.15521746451402652,
    "y médio": 0.3626679679536317
  },
  "baixa": {
    "MAE HAR linear": 0.052939052582906924,
    "MAE MLP": 0.04796976323504408,
    "RMSE HAR linear": 0.06706714318761449,
    "RMSE MLP": 0.06045435391305069,
    "amostras": 39456,
    "viés HAR linear": 0.04995396280560834,
    "viés MLP": 0.04438679191557484,
    "y médio": 0.10017887639234369
  },
  "media": {
    "MAE HAR linear": 0.03917892767278316,
    "MAE MLP": 0.03977875817701914,
    "RMSE HAR linear": 0.05122835931304669,
    "RMSE MLP": 0.05042485917607475,
    "amostras": 27802,
    "viés HAR linear": -0.00835938009194779,
    "viés MLP": -0.0117220760542092,
    "y médio": 0.18103308145124844
  }
}
```

### `extra_gate`

| Chave | Valor |
|---|---|
| `metric` | qlike |
| `split` | validation |
| `threshold` | 0.8200 |
| `value` | 0.7996 |
| `higher_is_better` | False |

### `golden_cases`

250

### `golden_tolerance`

| Chave | Valor |
|---|---|
| `output_atol` | 1.00e-04 |
| `note` | unidades naturais (fracao a.a.); a API soma rtol=1e-5 |

### `manifest_vs_notebook_features_max_abs_diff`

0.0000

### `notebook_test_table`

```json
{
  "HAR linear": {
    "MAE": 0.07661465965344658,
    "MAE/média": 0.4009408811686054,
    "QLIKE": 0.8835699862563847,
    "RMSE": 0.13543883116171887,
    "RMSE/média": 0.7087803372886571,
    "R² (log)": 0.1856847611128547,
    "viés": -0.020059650705928855
  },
  "MLP": {
    "MAE": 0.07407369578414219,
    "MAE/média": 0.3876434744140136,
    "QLIKE": 0.8706641912224933,
    "RMSE": 0.13321432513195275,
    "RMSE/média": 0.6971390220133085,
    "R² (log)": 0.2330663264398346,
    "viés": -0.02307825020043296
  },
  "média do treino": {
    "MAE": 0.09852226480339325,
    "MAE/média": 0.5155880590435038,
    "QLIKE": 0.8377956529608512,
    "RMSE": 0.14586209252086574,
    "RMSE/média": 0.7633274907040807,
    "R² (log)": -0.26271197047415273,
    "viés": 0.0239848617534952
  },
  "persistência (rv_5)": {
    "MAE": 0.10217691695781375,
    "MAE/média": 0.5347136345115164,
    "QLIKE": 1.9230043140353066,
    "RMSE": 0.17482723718102422,
    "RMSE/média": 0.9149082805392234,
    "R² (log)": -0.34197199870931283,
    "viés": -0.0036774156809198707
  }
}
```

### `onnx`

```json
{
  "bytes": 6323,
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
    "Exp",
    "MatMul",
    "Mul",
    "Relu"
  ],
  "opset": 17,
  "output": {
    "name": "Identity:0",
    "shape": [
      null,
      1
    ],
    "type": "tensor(float)"
  }
}
```

### `parity_max_abs_diff`

| Chave | Valor |
|---|---|
| `clone` | 4.17e-07 |
| `numpy` | 3.12e-07 |
| `onnx` | 3.58e-07 |

### `sizes`

| Chave | Valor |
|---|---|
| `train` | 398738 |
| `val` | 91612 |
| `test` | 89792 |

### `split`

| Chave | Valor |
|---|---|
| `train_start` | 2013-02-08 |
| `train_end` | 2016-08-08 |
| `val_start` | 2016-08-16 |
| `val_end` | 2017-05-09 |
| `test_start` | 2017-05-17 |
| `test_end` | 2018-01-31 |
| `new_start` | 2018-02-01 |
| `new_end` | 2018-02-07 |

### `tercile_accuracy_test`

| Chave | Valor |
|---|---|
| `mlp` | 0.5053 |
| `har` | 0.4804 |
| `persistence` | 0.4602 |
| `majority` | 0.4394 |

### `tercile_edges`

```json
[
  0.14407703857207588,
  0.22880985657794298
]
```

### `timings_seconds`

| Chave | Valor |
|---|---|
| `train` | 9.8844 |
| `export` | 0.8252 |
| `notebook_part_20` | 179.22 |

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

Gerado por `tools/gen_model_card.py` a partir de `artifacts/volatility-mlp/1.0.0/manifest.json`. Não edite à mão: rode `uv run --package trinet-contract python tools/gen_model_card.py`.
