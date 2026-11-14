# ADR 0007: Monitoramento de drift com PSI sobre as métricas expostas

- Status: aceito, implementação parcial
- Data: 2026-11-10
- Origem: [PLANO](../../PLANO.md) §5.5 e §7

## Contexto

Um modelo em produção degrada quando os dados deixam de se parecer com os do treino (drift). O PLANO pediu
observabilidade de "IA em produção": métricas RED por modelo, contador de classes previstas, histogramas de 3 a 4
features com buckets vindos do manifesto, e um ADR curto sobre PSI. OpenTelemetry foi cortado (sem coletor nem
backend, ninguém leria os traces); ficaram `structlog` com `request_id` e o Prometheus do profile `observability`.

A cardinalidade tem de ser limitada por construção: nenhum rótulo de métrica pode vir de dados da requisição.

## Decisão

**O que a API expõe de fato** (porta interna `TRINET_METRICS_PORT`, padrão 9100, caminho `/metrics`, nunca publicada
pelo compose; registrador privado com o coletor de processo):

| Métrica | Tipo | Rótulos |
|---|---|---|
| `trinet_http_requests_total` | counter | `method`, `route` (modelo de rota), `model`, `status` |
| `trinet_http_request_duration_seconds` | histogram | `method`, `route`, `model` |
| `trinet_inference_duration_seconds` | histogram | `model`, `runtime` |
| `trinet_predicted_class_total` | counter | `model`, `label` (rótulos do manifesto) |
| `trinet_http_inflight_requests` | gauge | |
| `trinet_inflight_inferences` | gauge | |
| `trinet_model_info` | gauge (1 = serve, 0 = indisponível) | `model`, `version`, `status`, `runtime` |

O rótulo `model` vem só dos modelos carregados (qualquer outro valor vira `unknown`), `route` é o *template* da
rota (`/v1/models/{name}/predictions`), nunca o caminho real, e `status` é o código HTTP. Há ainda as métricas
padrão de processo.

**Como usar para drift.** O sinal disponível é o **drift de predição para classificadores**: a distribuição de
`trinet_predicted_class_total` por janela comparada com uma distribuição de referência, medida pelo PSI
(*Population Stability Index*):

```text
PSI = soma sobre as classes de (a_i - e_i) * ln(a_i / e_i)
```

onde `e_i` é a fração esperada (referência) e `a_i` a fração observada na janela, com um piso pequeno (por exemplo
`1e-4`) para classes sem contagem. A regra prática usual: abaixo de 0,10 estável, entre 0,10 e 0,25 atenção, acima
de 0,25 mudança relevante. São convenções da indústria, não limiares calibrados para este projeto.

A contagem por janela sai do Prometheus (`increase()` trata o reinício do contador na troca de contêiner):

```text
sum by (label) (increase(trinet_predicted_class_total{model="credit-default-mlp"}[1d]))
```

O cálculo do PSI roda **fora da API** (script ou alerta); o [runbook](../runbook.md#checar-drift-de-classe-prevista)
traz um exemplo. A referência é o suporte por classe no split de teste do notebook, hoje fora do contrato.

## Consequências

- Cobre `credit-default-mlp` (taxa de inadimplência prevista) e `candlestick-cnn` (distribuição dos regimes
  `baixa`, `media` e `alta`). **Não cobre** o `volatility-mlp`: a API não expõe a distribuição dos valores previstos
  da regressão, só latência e erros.
- No mercado, a mudança na distribuição prevista pode ser o **regime** mudando, e não defeito do modelo: os tercis que
  definem as classes do `candlestick-cnn` vêm do treino, então um mercado mais volátil que o de 2013-2018 desloca a
  distribuição de propósito (o teste do projeto já inclui a alta de jan-fev/2018). Em crédito, uma taxa prevista
  diferente pode ser uma carteira diferente. O PSI sinaliza que algo mudou; a causa exige olhar os dados.
- **Não há drift de entrada.** Os histogramas por feature previstos no PLANO não foram implementados: pediriam
  buckets de referência no manifesto (o contrato 1.0 não os tem) e aumentariam a cardinalidade. O log de predição
  guarda a classe prevista e o SHA-256 da entrada, nunca o conteúdo, por privacidade. O que se vê de entrada é
  indireto: a taxa de `422` por `status` na métrica HTTP.
- Drift de predição é um proxy: a distribuição de classes pode mudar sem que o modelo piore, e o modelo pode piorar
  sem mudar a distribuição de classes. Sem rótulos de produção, não há como medir acurácia online.
- A referência por classe precisa ser levada manualmente do notebook (`results.json` ou o model card). Um próximo
  passo coerente é um campo opcional no manifesto com a distribuição de referência e, para entradas, bins por
  feature, o que exige subir o `schema_version`.
- O Prometheus do compose guarda 7 dias (`--storage.tsdb.retention.time=7d`) e raspa a cada 15 s; janelas maiores
  exigem um armazenamento externo. Não há dashboard do Grafana: dependeria de nomes de métrica que o contrato ainda
  não congela.
