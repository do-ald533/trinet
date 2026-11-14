# Model card: candlestick-cnn

> **Uso educacional.** Trabalho de curso (CIC407, T2). Nada neste modelo ou na API é recomendação de investimento, sinal de compra ou venda, nem base para decisões financeiras reais.

> **Candidato.** Exportado pelo notebook, mas ainda não promovido a `release`. A release exige versão >= 1.0.0 e quality gate aprovado (`make check-artifacts-release`).

Regime de volatilidade dos proximos 5 pregoes (baixa/media/alta, tercis do treino) a partir de um grafico OHLC + media movel + volume de 20 pregoes de acoes do S&P 500, renderizado em 64x64x3. Uso educacional; nao e recomendacao de investimento.

| Campo | Valor |
|---|---|
| Versão | `1.0.0` |
| Status | `candidate` |
| Tarefa | classificação de imagens (`image_classification`) |
| Criado em | `2026-11-14T13:05:02+00:00` |
| Artefato | `artifacts/candlestick-cnn/1.0.0/` |
| Schema do contrato | `1.0` |

## 1. Uso pretendido

- Classificar o regime de volatilidade dos próximos 5 pregões (`baixa`, `media` ou `alta`) a partir de um gráfico renderizado de 20 pregões: barras OHLC, média móvel de 20 dias e volume, em 64x64x3 pixels.
- Responde à pergunta de conclusão do projeto: o gráfico acrescenta algo além das features numéricas do `volatility-mlp`? A ideia vem de Jiang, Kelly e Xiu (2023), *(Re-)Imag(in)ing Price Trends*, Journal of Finance.
- Demonstração didática da disciplina CIC407 (T2) e exemplo de serviço de inferência com imagens.

**Fora do escopo**

- Recomendação de investimento ou previsão da direção do preço: o alvo é regime de volatilidade, e a direção foi descartada por ser quase aleatória.
- Gráficos que não sejam o produto de `tools/render_chart.py` (capturas de tela, outras plataformas, outra escala ou outro tamanho): a API aceita qualquer PNG válido, mas fora da rasterização do treino o resultado não tem significado.

## 2. Dados

- S&P 500, 5 anos (Kaggle `camnugent/sandp500`): 619.040 linhas e 7 colunas (`date`, `open`, `high`, `low`, `close`, `volume`, `Name`), 505 tickers, de 2013-02-08 a 2018-02-07 (1.259 pregões). Preços ajustados a splits.
- A licença e a origem do arquivo **não foram verificadas**: o CSV nunca é versionado nem redistribuído (o download via `kagglehub` é anônimo, feito por quem roda o notebook).
- Limpeza (PLANO v1.2, V3): remove linhas com NaN em OHLC, volume 0, `high < low` e `close` fora de [`low`, `high`]. Uma amostra em `t` exige a janela [t-59, t+5] contígua e nenhum |retorno log| > 0,4 nela (descarta artefatos de ações corporativas e 3 eventos reais).
- Divisão cronológica por data (todos os tickers juntos): 70% / 15% / 15% dos pregões, com embargo (purge) de 5 pregões em cada fronteira, porque o alvo olha 5 dias à frente. O teste é tocado uma vez.
- Rótulos: tercis de `y` (volatilidade realizada futura de 5 pregões) calculados só no treino. Chance = 33,3%. O treino usa em torno de 60 mil imagens, por amostragem com passo (PLANO v1.2, V4).
- Os golden samples do repositório são gráficos **sintéticos** (passeio aleatório geométrico), nunca imagens derivadas do CSV de preços.

Hash dos dados de treino (SHA-256): `6aea253cd19de60b568143991aaf1fa482456565c389205658d236e595e716cf`

## 3. Pré-processamento

Pipeline da API (o mesmo que o notebook mede em `serving_skew`):

- formatos aceitos: PNG, validados pelos magic bytes (o tipo declarado pelo cliente não é confiado); até 262144 bytes (0.25 MiB);
- checagem do tamanho pelo cabeçalho antes de decodificar (limite de pixels da API, `TRINET_MAX_IMAGE_PIXELS`);
- decodificação, conversão para RGB e, se o tamanho for diferente, redimensionamento bilinear (Pillow) para 64x64;
- float32 na faixa [0, 255] (o reescalonamento fica dentro do grafo);
- a orientação EXIF é ignorada de propósito.

**Responsabilidade do chamador**

- O chamador renderiza a janela de 20 pregões com a mesma função do notebook: `python tools/render_chart.py acao.csv -o grafico.png` (NumPy puro, sem antialiasing, pixels 0 ou 255). Um teste do apêndice do notebook garante que as duas cópias da função produzem arrays idênticos.
- Só PNG: JPEG destruiria o traço de 1 pixel, e a API responde 415 a ele.

## 4. Arquitetura e runtimes

- Entrada: tensor `chart` float32, forma (batch, 64, 64, 3).
- Saída: tensor `Identity:0` float32, forma (batch, 3): 3 classes: `baixa`, `media`, `alta`.

Os runtimes seguem a ordem de preferência do manifesto (escada de fallback): a API usa o primeiro que passa na verificação de SHA-256 e no warm-up. Veja o [ADR 0001](../adr/0001-onnx-na-api-e-escada-de-fallback.md).

| Ordem | Runtime | Arquivo | SHA-256 | Detalhe |
|---|---|---|---|---|
| 1 | `onnx` | `model.onnx` | `76db5861d692` | opset 17 |

A estrutura das camadas treináveis (larguras, profundidade) está no notebook, em `notebook/`, e, quando exportada, em `results.json`.

## 5. Métricas por split

| Split | accuracy | loss | macro_f1 |
|---|---|---|---|
| `validation` | 0.5205 | 1.0032 | - |
| `test` | 0.5249 | - | 0.4453 |

## 6. Quality gate de exportação

O notebook se recusa a escrever o artefato se o gate falhar; o limiar é a métrica de validação com uma margem definida depois do treino.

| Métrica | Split | Regra | Valor | Resultado |
|---|---|---|---|---|
| `accuracy` | `validation` | >= 0.4900 | 0.5205 | aprovado |

## 7. Training/serving skew

| Medida | Valor |
|---|---|
| Amostras comparadas | 12830 |
| Maior diferença absoluta (Keras x runtime) | 2.384e-07 |
| Concordância do top-1 | 1.0000 |
| Variação de acurácia (pontos percentuais) | +0.00 |

Critério do projeto para imagens (PLANO 4.5): top-1 igual em pelo menos 99,5% e |variação de acurácia| <= 0,3 pp. Situação: atendido.

## 8. Limitações

- Viés de sobrevivência: só entram as ações que eram constituintes do índice em fev/2018. Empresas que saíram, faliram ou foram compradas antes estão ausentes.
- Período curto: cinco anos de um único regime de mercado, com a alta de volatilidade de jan-fev/2018 **dentro do teste**. Ela é um evento fora do que o treino viu, então o erro nesse trecho é esperado e deve ser lido separado (o notebook mostra o erro por sub-período).
- O preço de fechamento ajustado vem de uma fonte não auditada; 14 retornos com |log| > 0,4 indicam ajustes de desdobramento e cisão incompletos (alguns eventos são reais).
- O rótulo é função de `y`, e a volatilidade recente já prediz a futura (persistência): a amplitude das barras no gráfico carrega essa informação. A CNN pode ter só reaprendido a persistência, por isso o notebook a compara com o baseline tabular para o mesmo alvo de 3 classes.
- Os tercis vêm do treino. No teste, que inclui a alta de jan-fev/2018, a distribuição de classes não é uniforme.
- Resolução fixa e uma única escolha de rasterização (canais, escala vertical e janela); outras escolhas mudariam o resultado.
- As métricas valem para a distribuição dos dados de treino e teste; fora dela o desempenho não é garantido.
- A API expõe contadores por classe prevista, mas não histogramas por feature; o que dá e o que não dá para detectar de drift está no [ADR 0007](../adr/0007-monitoramento-de-drift-psi.md).

## 9. Considerações éticas e de uso

- **Não é recomendação.** O artigo que inspira a ideia reporta ganhos em portfólio, e uma réplica menor concluiu que a CNN não passa de chute em acertos direcionais; este modelo não prevê direção e não deve embasar operações.
- **Vieses declarados.** Sobrevivência e regime único (veja as limitações).
- **Dados.** Não há dados pessoais. As imagens de mercado reais não são redistribuídas; o serviço não as guarda (os logs registram apenas o SHA-256 do corpo da requisição).

Contexto da escolha dos dados e dos alvos: [ADR 0010](../adr/0010-pivo-para-o-mercado-financeiro.md).

## 10. Proveniência

| Campo | Valor |
|---|---|
| Git SHA | `273c6f769a9220b1d14c04e21738968263387af3` |
| Conjunto de dados | S&P 500 5 anos (Kaggle camnugent/sandp500), janelas OHLCV de 20 pregoes renderizadas em 64x64x3 |
| SHA-256 dos dados | `6aea253cd19de60b568143991aaf1fa482456565c389205658d236e595e716cf` |
| Seed | 42 |
| Python | 3.13.13 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| Hardware | 11th Gen Intel(R) Core(TM) i7-11370H @ 3.30GHz \| 8 threads \| 33 GB RAM \| GPU: nao (CPU) |
| Golden samples | `golden.json`: 20 casos (imagens PNG), 20 rotulados |
| SHA-256 de `model.onnx` | `76db5861d692fafc3f300ccae17aacfd95a45512a5641ad718488a8e9b80509e` |

## 11. Resultados adicionais (`results.json`)

### `ablation`

| configuração | parâmetros | val acc (média) | val acc (desvio) | val loss (média) | s/época | cabe no orçamento |
|---|---|---|---|---|---|---|
| base | 137587 | 0.5095 | 0.0044 | 1.0022 | 8.8291 | True |
| dropout 0,25 | 137587 | 0.4851 | 0.0401 | 1.0271 | 8.9582 | True |
| augmentation (jitter ±3 linhas) | 137587 | 0.5119 | 0.0036 | 1.0047 | 9.1499 | True |
| mais rasa (2 blocos) | 263891 | 0.5072 | 0.0048 | 1.0149 | 8.0640 | True |
| mais larga (16/32/64, extra) | 286435 | 0.5162 | 0.0045 | 1.0269 | 18.1729 | False |

### `architecture`

```json
{
  "config": {
    "dense": 64,
    "dropout": 0.5,
    "filters": [
      8,
      16,
      32
    ],
    "jitter": 0
  },
  "parametros": 137587
}
```

### `baselines_acuracia_teste`

| Chave | Valor |
|---|---|
| `chance` | 0.3333 |
| `classe_majoritaria_treino_no_teste` | 0.4384 |
| `har_logistico_mesmas_imagens` | 0.5140 |
| `rede2_har` | 0.4804 |
| `rede2_mlp` | 0.5053 |

### `confusion_matrix_test`

```json
[
  [4899, 396, 330],
  [2799, 543, 669],
  [1410, 491, 1293]
]
```

### `dataset`

```json
{
  "classes": [
    "baixa",
    "media",
    "alta"
  ],
  "imagens_teste": 12830,
  "imagens_treino": 56956,
  "imagens_validacao": 13088,
  "stride": 7,
  "tercis_y_treino": [
    0.14407703857207588,
    0.22880985657794298
  ]
}
```

### `direction_sanity`

| Chave | Valor |
|---|---|
| `val` | 0.5270 |
| `test` | 0.5761 |
| `test_always_prior_class` | 0.5761 |
| `test_share_up` | 0.5761 |

### `export`

```json
{
  "clone_vs_treinado_max_abs_diff": 0.0,
  "golden_acuracia_sintetica_vs_regime_do_gerador": 0.65,
  "golden_casos": 20,
  "golden_png_maior_bytes": 594,
  "onnx": {
    "bytes": 557561,
    "ir_version": 8,
    "ops": [
      "Add",
      "Cast",
      "Concat",
      "Conv",
      "Gather",
      "MatMul",
      "MaxPool",
      "Mul",
      "Relu",
      "Reshape",
      "Shape",
      "Slice",
      "Softmax",
      "Transpose"
    ],
    "opset": 17
  },
  "rasterizacao_igual_a_tools_render_chart": {
    "checks": {
      "lote == unitario": true,
      "png decodificado == array": true,
      "png_bytes": true,
      "render_chart": true,
      "render_windows[fixed,0.18]": true,
      "render_windows[fixed,0.3]": true,
      "render_windows[window,0.3]": true
    },
    "cli_matches": true,
    "identical": true
  },
  "rota": "tf2onnx via Model.export(format='onnx', opset_version=17)"
}
```

### `export_seconds`

31.1838

### `hardware`

11th Gen Intel(R) Core(TM) i7-11370H @ 3.30GHz | 8 threads | 33 GB RAM | GPU: nao (CPU)

### `lr_sweep`

| taxa de aprendizado | val acc (média) | val acc (desvio) | val loss (média) | melhor época (média) |
|---|---|---|---|---|
| 3.00e-04 | 0.5077 | 0.0025 | 0.9940 | 7.3333 |
| 0.0010 | 0.5095 | 0.0044 | 1.0022 | 8.0000 |
| 0.0030 | 0.4782 | 0.0352 | 1.0402 | 5.6667 |

### `pilot_escala_vertical`

| escala vertical | val acc (melhor época) | desvio | janelas recortadas | px por 1% de preço |
|---|---|---|---|---|
| por janela (min/max) | 0.4153 | 0.0147 | 0.0000 | None |
| fixa, 0,15 | 0.4773 | 0.0343 | 0.1520 | 4.2000 |
| fixa, 0,20 | 0.4543 | 0.0183 | 0.0612 | 3.1500 |
| fixa, 0,30 | 0.4363 | 0.0037 | 0.0152 | 2.1000 |
| fixa, 0,45 | 0.4345 | 0.0035 | 0.0015 | 1.4000 |

### `timings_s`

| Chave | Valor |
|---|---|
| `epoca_media` | 34.9314 |
| `primeira_epoca` | 35.4935 |
| `renderizacao_imagens` | 11.6498 |
| `melhor_epoca` | 4 |

### `versions`

| Chave | Valor |
|---|---|
| `python` | 3.13.13 |
| `tensorflow` | 2.21.0 |
| `keras` | 3.15.1 |
| `numpy` | 2.5.4 |
| `onnx` | 1.23.2 |
| `onnxruntime` | 1.31.1 |
| `tf2onnx` | 1.17.0 |

---

Gerado por `tools/gen_model_card.py` a partir de `artifacts/candlestick-cnn/1.0.0/manifest.json`. Não edite à mão: rode `uv run --package trinet-contract python tools/gen_model_card.py`.
