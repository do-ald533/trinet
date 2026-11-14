# ADR 0010: Pivô para o mercado financeiro

- Status: aceito (substitui os datasets, alvos e nomes de modelo da v1.0 do PLANO)
- Data: 2026-11-14
- Origem: [PLANO](../../PLANO.md), seção "v1.2 — Pivô para mercado financeiro" (V1 a V5)

## Contexto

A v1.0 usava três datasets didáticos sem relação entre si: Dry Bean (classificação tabular), California Housing
(regressão tabular) e Intel Image Classification (imagens). Em 14/11/2026 o usuário pediu um viés de **mercado
financeiro**. A arquitetura, o contrato, a API, o CI/CD e a segurança já estavam prontos e não dependem do domínio;
mudam os dados, os alvos e os nomes dos modelos. O prazo da entrega (19/11) comprimiu o cronograma.

O projeto é um trabalho de curso. Nada nele é recomendação de investimento, e o modelo de crédito não é uma decisão de
crédito real.

## Decisão

**1. Três modelos, no lugar dos antigos.**

| Rede | Antes (v1.0) | Agora (v1.2) | Modelo |
|---|---|---|---|
| Classificação tabular | Dry Bean | *Default of Credit Card Clients* (UCI #350): risco de crédito | `credit-default-mlp` |
| Regressão tabular | California Housing | volatilidade realizada dos próximos 5 pregões de ações do S&P 500: risco de mercado | `volatility-mlp` |
| CNN de imagens | Intel Scenes | gráficos OHLC + volume renderizados, 3 classes de regime de volatilidade | `candlestick-cnn` |

As redes 2 e 3 modelam a **mesma grandeza** por duas vias, números e imagem, o que permite a pergunta de conclusão: o
gráfico acrescenta algo além das features? A **direção** do preço (sobe ou desce) foi descartada como alvo: é quase
aleatória, e uma célula de sanidade curta no notebook mostra isso. A ideia da CNN sobre gráficos vem de Jiang, Kelly e
Xiu (2023), *(Re-)Imag(in)ing Price Trends*, The Journal of Finance.

**2. O alvo da CNN é o mesmo da regressão.** As classes `baixa`, `media` e `alta` são os tercis da volatilidade
realizada futura `y`, calculados **só no treino**. A entrada é uma janela de 20 pregões rasterizada de forma
determinística em NumPy (sem antialiasing), 64x64x3, em três canais independentes (barras OHLC, média móvel de 20 dias
e volume). A API só aceita **PNG**: JPEG destruiria o traço de 1 pixel. A função de rasterização existe em duas
cópias, no notebook e em `tools/render_chart.py`, e o apêndice do notebook testa que elas dão arrays idênticos.

**3. O contrato ganhou a camada `exp`.** O alvo da regressão é `ln(y)` padronizado, então o clone de inferência termina
em `Normalization(invert=True)` seguido de `Activation("exponential")`, e o modelo devolve a volatilidade anualizada em
fração. O manifesto passou a aceitar `{"op": "exp"}` (`ExpLayer`, sem chave nem pesos) nas camadas do runtime NumPy; no
ONNX é um nó `Exp`. A resposta da regressão traz `value`, `unit` e `target`.

**4. As features de série temporal são calculadas pelo chamador.** A API de `volatility-mlp` recebe 20 campos já
calculados no fechamento do dia `t` (retornos defasados, volatilidades de 5 a 60 dias, estimadores de Parkinson,
`volume_ratio`, `gap`, momentum, `drawdown_60`, `mkt_rv_20` e `dow`). Só a razão `rv_5 / rv_20` e o one-hot de `dow`
são feitos pela API, com as primitivas fechadas do [ADR 0003](0003-contrato-de-artefato.md). Calcular as features na
API exigiria receber a série OHLCV, guardar estado (`mkt_rv_20` usa 505 tickers), criar uma primitiva de janela e subir
o `schema_version`: mais superfície de ataque e de código por um ganho que o trabalho não pede. As fórmulas ficam no
notebook e no model card.

**5. Divisão cronológica com embargo.** Nas séries, a divisão é por data, com todos os tickers juntos: 70% / 15% / 15%
dos pregões, com **purge de 5 pregões** em cada fronteira (o alvo olha 5 dias à frente). Toda feature em `t` usa só
dados até o fechamento de `t`. O teste é tocado uma vez. Uma amostra exige a janela `[t-59, t+5]` contígua e nenhum
retorno com |log| > 0,4 (artefatos de ações corporativas). Os 5 últimos pregões do dataset, sem alvo ainda, viram as
"novas instâncias". No crédito, que não tem datas, a divisão é 70/15/15 estratificada, sem duplicatas, com 20 linhas
reais (10 por classe) reservadas antes de tudo.

**6. Atributos protegidos fora das features.** `SEX` e `MARRIAGE` não entram no modelo de crédito nem na API; ficam só
para o **diagnóstico de equidade** no teste (taxa de aprovação e verdadeiros positivos por grupo). A LGPD, art. 20, dá
ao titular o direito de pedir revisão de decisões automatizadas, inclusive de crédito; o texto precisa ser confirmado em
planalto.gov.br antes de ser citado.

**7. Aviso e dados.** O aviso de uso educacional vai no README, nos model cards (que ganharam a seção "Considerações
éticas e de uso") e na descrição do OpenAPI. O CSV do S&P 500 tem licença e origem não verificadas e **nunca é
versionado**; só entram no repositório imagens sintéticas. Os downloads do notebook (UCI e `kagglehub`) são anônimos, e
o CI nunca baixa dataset.

## Consequências

- Os nomes `drybean-mlp`, `housing-mlp` e `intel-cnn`, e os textos sobre Dry Bean, California Housing e Intel Scenes
  em PLANO §2, §4.2 a §4.3 e nos exemplos de §5 e §6, estão obsoletos; vale a seção v1.2. O PLANO não foi editado, e
  a documentação, os dummies, os testes, o Makefile e o smoke usam os nomes novos.
- **Skew fora da API.** O golden prova que a API reproduz o notebook a partir das entradas dadas; não prova que o
  chamador calculou as features (ou renderizou o gráfico) da mesma forma. A mitigação é documental e parcial: fórmulas
  no card, `tools/render_chart.py` testado bit a bit contra o notebook.
- **Vieses declarados, e que se vão ler nas métricas.** Sobrevivência (só constituintes do índice em fev/2018), um
  único regime de mercado em cinco anos e a alta de volatilidade de jan-fev/2018 **dentro do teste**. O erro no teste
  pode ser pior que na validação por construção, e isso é um achado, não um defeito.
- **O limiar de decisão do crédito não está no contrato.** O notebook escolhe o limiar na validação com um custo
  ilustrativo (falso negativo custa 5 falsos positivos), mas o manifesto 1.0 não tem onde guardá-lo: a API devolve
  probabilidades e o `label` pelo corte em 0,5. Um campo de limiar exigiria subir o `schema_version`.
- **`exp` devolve uma mediana condicional.** Retransformar `ln(y)` estima a mediana, que tende a ficar abaixo da média
  da volatilidade quando o erro em log tem variância.
- **`schema_version` continua `1.0`** apesar da camada `exp` nova, o que contraria a regra do ADR 0003. É aceitável
  porque nenhuma release 1.0.0 existia; depois dela, uma mudança assim sobe a versão.
- O notebook deixa de precisar de credencial do Kaggle, o que simplifica o ambiente e o [ADR 0009](0009-repositorio-publico-e-credenciais-manuais.md).
- O monitoramento de drift cobre `credit-default-mlp` e `candlestick-cnn` e **não** cobre `volatility-mlp`
  ([ADR 0007](0007-monitoramento-de-drift-psi.md)).
