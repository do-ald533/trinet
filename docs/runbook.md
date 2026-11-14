# Runbook de operação

Guia para quem sobe, configura e depura a API de inferência do TriNet em um único host com Docker Compose. Os
comandos assumem a raiz do repositório. O serviço é um **projeto educacional**: nada nele é recomendação de
investimento nem decisão de crédito real (veja o [README](../README.md)).

| Componente | Onde | Observação |
|---|---|---|
| API pública | `127.0.0.1:8000` | só em loopback; um proxy reverso deve terminar o TLS para acesso remoto |
| Métricas | `api:9100/metrics` | porta **interna**, nunca publicada pelo compose |
| Prometheus | `127.0.0.1:9090` | só no profile `observability`, retenção de 7 dias |
| Chave da API | `secrets/api_key` → `/run/secrets/api_key` | arquivo, nunca variável de ambiente nem `.env` |
| Modelos | `/app/artifacts` | **dentro da imagem** (`COPY artifacts/`): trocar um modelo é gerar uma imagem nova |

## Deploy

### Local (constrói a imagem)

```bash
make up                # base python:3.13-slim-trixie fixada por digest (sem conta, sem login)
make down
```

`make up` constrói a imagem, cria `secrets/api_key` se faltar e sobe o compose (`up -d --no-build --wait`). Detalhes de bases, do
Docker Desktop fora de `$HOME` e do endurecimento do contêiner estão em [`deploy/README.md`](../deploy/README.md).

### Verificado (a imagem que o CI construiu)

`make deploy` confere as duas atestações do `actions/attest` (proveniência SLSA e SBOM CycloneDX) antes de subir:

```bash
make deploy DEPLOY_IMAGE=ghcr.io/<dono>/trinet-api@sha256:<digest> GH_REPO=<dono>/trinet
```

Ele roda `gh attestation verify` duas vezes, com `--signer-workflow <dono>/trinet/.github/workflows/cd.yml` e
`--deny-self-hosted-runners`, depois `make secrets` e `compose up -d --no-build --pull always --wait`. Se a
verificação falhar, **nada sobe**. `GH_REPO` tem como padrão `<dono da imagem>/trinet`; exige um `gh` recente, com o
subcomando `gh attestation`, autenticado.

Onde achar o digest:

- no resumo do job `publish` da execução do CD, a seção "Published image" traz a referência completa e o comando
  `make deploy` pronto;
- a partir de uma tag: `docker buildx imagetools inspect --raw ghcr.io/<dono>/trinet-api:sha-<12 caracteres> | sha256sum`
  (as tags são cópias do mesmo manifesto, então o digest é o mesmo que foi atestado).

Antes de um release, o CD só publica tags `sha-<commit>`: `latest` e as tags `vX.Y.Z` só se movem pelo CD manual com
`release=true` e `version=1.0.0`, e só se `make check-artifacts-release` passar.

### Depois de subir: confirme os modelos

O `HEALTHCHECK` da imagem (`python -m trinet_api.healthcheck`) só consulta `/health/live`, que responde 200 mesmo
quando um modelo falhou ao carregar. `compose up --wait` portanto **não prova que os modelos servem**. Confira:

```bash
curl -s http://127.0.0.1:8000/health/ready
curl -s -H "X-API-Key: $(cat secrets/api_key)" http://127.0.0.1:8000/v1/models
```

`status` deve ser `ok` e todo modelo deve ter `"available": true`. `degraded` significa que ao menos um modelo não
carregou: veja [404 model_not_available](#404-model_not_available).

## Configuração

Todas as variáveis têm o prefixo `TRINET_` e são lidas por `trinet_api.settings.Settings` (valores inválidos impedem
o serviço de subir). O compose repassa as marcadas e aceita sobrescrevê-las pelo `.env` (copie o `.env.example`).

| Variável | Padrão | No compose | Descrição |
|---|---|---|---|
| `TRINET_HOST` | `0.0.0.0` | | endereço da API (use `127.0.0.1` fora do contêiner) |
| `TRINET_PORT` | `8000` | | porta da API |
| `TRINET_METRICS_ENABLED` | `true` | | liga o servidor de métricas |
| `TRINET_METRICS_HOST` | `0.0.0.0` | | endereço do servidor de métricas (use `127.0.0.1` fora do contêiner) |
| `TRINET_METRICS_PORT` | `9100` | | porta das métricas (interna, nunca publicada) |
| `TRINET_ARTIFACTS_DIR` | `/app/artifacts` | | raiz dos artefatos |
| `TRINET_API_KEY_FILE` | `/run/secrets/api_key` | | arquivo da chave; **vence** `TRINET_API_KEY` se existir |
| `TRINET_API_KEY` | (nenhum) | | chave por variável, só para execução local sem Docker |
| `TRINET_LOG_LEVEL` | `INFO` | `TRINET_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` ou `CRITICAL` |
| `TRINET_MAX_CONCURRENCY` | `4` | `2` | inferências simultâneas (1 a 64); o excedente espera e vira 503 |
| `TRINET_QUEUE_TIMEOUT_S` | `2.0` | | espera máxima por uma vaga antes do 503 `overloaded` (0 a 60 s) |
| `TRINET_ORT_INTRA_OP_THREADS` | `1` | `1` | threads do ONNX Runtime por inferência (o runtime não respeita a cota de CPU do cgroup) |
| `TRINET_RATE_LIMIT` | `60/minute` | `60/minute` | limite do middleware próprio ([ADR 0011](adr/0011-rate-limit-proprio-em-vez-de-slowapi.md)); aceita vários, como `10/second;60/minute` |
| `TRINET_MAX_BATCH` | `256` | `256` | instâncias por requisição tabular (1 a 10.000) |
| `TRINET_MAX_JSON_BYTES` | `1048576` (1 MiB) | | corpo JSON máximo |
| `TRINET_MAX_UPLOAD_BYTES` | `6291456` (6 MiB) | | corpo `multipart` máximo; a imagem ainda respeita o `max_bytes` do manifesto |
| `TRINET_MAX_IMAGE_PIXELS` | `12000000` | | pixels máximos de uma imagem (checado pelo cabeçalho, antes de decodificar) |

Outras variáveis do compose (não são `TRINET_*` da API): `TRINET_IMAGE`, `BASE_IMAGE`, `BUILD_IMAGE`,
`TRINET_API_PORT` e `PROMETHEUS_PORT`, documentadas no [`.env.example`](../.env.example).

O limite de requisições é um middleware ASGI próprio, não o `slowapi`: requisições com a chave válida dividem um
balde, as demais têm um balde por IP, e `/health/live` e `/health/ready` são isentas. O estouro responde 429 com
`Retry-After`.

## Segredos

A chave da API é um arquivo, não uma variável de ambiente.

- **Criar:** `make secrets` cria `secrets/api_key` (aleatório, 43 caracteres, ignorado pelo git) se ainda não existir.
  O compose (fora do Swarm) monta o arquivo do host como está e **ignora `uid`, `gid` e `mode`** dos segredos, então o
  arquivo precisa ser legível pelo uid 65532 do contêiner: o alvo o cria com modo `0644` dentro de um diretório `0700`
  (outros usuários do host não atravessam o diretório).
- **Falha fechada:** sem chave, ou com menos de 16 caracteres, a API **não sobe**
  (`no API key configured...` / `the API key must have at least 16 characters`).
- **Nunca** coloque a chave em `.env`, em argumento de build, no compose ou na linha de comando. O repositório é
  público ([ADR 0009](adr/0009-repositorio-publico-e-credenciais-manuais.md)); Gitleaks e o push protection são a
  proteção preventiva.
- **Girar a chave:** a chave é lida **só na inicialização**. Escreva o novo valor no arquivo e reinicie:

  ```bash
  python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > secrets/api_key
  docker compose -f deploy/compose.yaml restart api
  ```

  Não há sobreposição entre a chave antiga e a nova: os clientes têm de trocar na mesma hora.
- **Vazou?** Trate como comprometida: gire a chave. Não reescreva o histórico do git para "apagar" o segredo.
- O CI e o CD não usam segredo algum além do `GITHUB_TOKEN` embutido ([ADR 0012](adr/0012-fluxos-gratuitos-e-sem-credenciais.md)); a única
  variável do repositório é `SECURITY_GATE`, configurada à mão no GitHub ([`.github/CICD.md`](../.github/CICD.md)).

## Health checks e métricas

| Rota | Quando responde |
|---|---|
| `GET /health/live` | o processo está de pé (200 `{"status":"ok"}`); o uvicorn só abre o socket **depois** de carregar os modelos |
| `GET /health/ready` | 200 `ok` (todos servem) ou 200 `degraded` (ao menos um serve); **503 `not_ready`** se nenhum serve; traz o mapa `models` |

O `HEALTHCHECK` da imagem roda a cada 30 s, com `start-period` de 30 s para dar tempo de carregar e aquecer os modelos.
Ambas as rotas são abertas (sem `X-API-Key`) e isentas do rate limit.

As métricas ficam em `http://api:9100/metrics`, acessível só pela rede do compose (a imagem não tem `curl` nem shell).
Para olhá-las, suba o Prometheus e use a interface em `http://127.0.0.1:9090`:

```bash
make build && make observability
```

| Métrica | Uso |
|---|---|
| `trinet_http_requests_total{method,route,model,status}` | taxa de pedidos e de erros por rota e modelo (`status` é o código HTTP) |
| `trinet_http_request_duration_seconds` | latência de ponta a ponta |
| `trinet_inference_duration_seconds{model,runtime}` | só o tempo de inferência |
| `trinet_predicted_class_total{model,label}` | contagem de classes previstas (base do [drift](#checar-drift-de-classe-prevista)) |
| `trinet_http_inflight_requests`, `trinet_inflight_inferences` | saturação (compare com `TRINET_MAX_CONCURRENCY`) |
| `trinet_model_info{model,version,status,runtime}` | `1` se o modelo serve, `0` se está indisponível |

Se o alvo `trinet-api` aparecer **down** em `/targets`, veja [falhas comuns](#falhas-comuns).

## Logs

Uma linha JSON por evento (`structlog`), com `event`, `level`, `logger`, `timestamp` e, nas requisições, `request_id`.
Os eventos úteis:

| `event` | Campos | Quando |
|---|---|---|
| `http_request` | `method`, `route`, `model`, `status`, `duration_ms` | uma por requisição |
| `prediction` | `model`, `version`, `runtime`, `batch_size`, `predicted`, `latency_ms`, `input_sha256` | uma por predição (o conteúdo nunca é registrado, só o hash) |
| `rate_limited` | `bucket` (`key` ou `ip`), `limit` | requisição barrada pelo limite |
| `unhandled_error` | traceback no log; **o cliente só recebe um 500 genérico** | exceção inesperada |

```bash
docker compose -f deploy/compose.yaml logs --no-log-prefix --since 1h api \
  | jq -c 'select(.event == "unhandled_error" or (.event == "http_request" and .status >= 500))'
```

O `request_id` da resposta (cabeçalho `X-Request-ID` e campo `request_id`) liga a queixa de um cliente à linha do log.

## 404 model_not_available

A API devolve dois 404 diferentes para `/v1/models/{name}`:

| `code` | Significa |
|---|---|
| `model_not_found` | não existe diretório `artifacts/<name>/` com uma versão válida. Confira o nome em `GET /v1/models` |
| `model_not_available` | o artefato **existe**, mas o carregador não conseguiu servi-lo. A API segue de pé e serve os outros modelos |

Como diagnosticar:

1. `GET /v1/models`: o modelo aparece com `"available": false` e um `reason`.
2. `GET /health/ready`: `degraded` e `"<modelo>": false` (ou 503 `not_ready` se nenhum serve).
3. Logs de inicialização: linhas `model <nome> <versão>: ...` em `error` ou `warning`.
4. Prometheus: `trinet_model_info{model="<nome>"} == 0`.

| `reason` | Causa | O que fazer |
|---|---|---|
| `integrity_check_failed` | arquivo citado pelo manifesto ausente, fora do diretório, acima de 256 MiB ou com **SHA-256 diferente** do manifesto | regenere o artefato inteiro pelo notebook (não edite arquivos à mão); `make check-artifacts` aponta qual arquivo |
| `invalid_manifest` | `manifest.json` ilegível, maior que 5 MiB, fora do schema, ou com `name`/`version` diferentes do diretório | `make check-artifacts`; corrija o diretório ou reexporte |
| `no_usable_runtime` | nenhum runtime da escada passou na construção e no aquecimento | veja o aviso `runtime <tipo> unusable (...)` no log |

**Escada de fallback.** O manifesto lista os runtimes em ordem de preferência (`onnx`, depois `numpy_mlp`, depois
`litert`). Para cada degrau a API confere o SHA-256, constrói o runtime e roda um lote de zeros (forma e valores
finitos; probabilidades válidas nos classificadores). O primeiro que passa serve, e `GET /v1/models` mostra qual em
`runtime`. Se um degrau falha, o log traz `runtime <tipo> unusable` em `warning` e o seguinte é tentado. Casos
típicos: ONNX com dados externos é recusado por segurança; o `litert` exige a instalação do extra
(`make build UV_SYNC_ARGS="--extra litert"`).

**Só a maior versão é carregada, e não há fallback silencioso entre versões.** Para cada modelo, vale a maior versão
semver em `artifacts/<nome>/` (uma pré-versão fica abaixo da release, então `1.0.0` vence `0.0.0-dummy`). Se a
`1.0.1` não carregar, o modelo fica **indisponível**, mesmo com a `1.0.0` íntegra ao lado. A saída é remover ou
reverter o diretório da versão ruim e reimplantar, ou voltar à imagem anterior ([rollback](#rollback-por-digest)).

## Trocar a versão de um modelo

Os artefatos vão dentro da imagem, então uma versão nova de modelo é um commit e uma imagem nova:

1. **Exporte** pelo notebook (Restart & Run All). O apêndice grava um diretório **novo**,
   `artifacts/<nome>/<semver>/`, e se recusa a escrever se o quality gate falhar. Nunca sobrescreva um diretório de
   versão já publicada.
2. **Valide** os artefatos: `make check-artifacts` (manifesto, SHA-256, golden, tamanhos) e
   `make check-artifacts-release` (a maior versão de cada modelo não é `dummy`, é >= 1.0.0 e tem o gate aprovado).
3. **Rode os testes**: `make test`. A suíte `api/tests/golden/` repete cada caso do `golden.json` pelo
   pré-processamento, por cada runtime da escada e pela API HTTP.
4. **Regenere a documentação**: `uv run --package trinet-contract python tools/gen_model_card.py` atualiza
   `docs/model-cards/` e a tabela de resultados do README (`--check` só confere, sem escrever).
5. **PR** para `main`; o único check obrigatório é o `ci-gate`.
6. **CD**: o push em `main` publica `ghcr.io/<dono>/trinet-api` com a tag `sha-<commit>`, atesta e roda o smoke.
7. **Release** (opcional): **Actions > CD > Run workflow** com `release=true` e `version=1.0.0`. O gate de release
   repete a checagem e só então move `latest` e `v1.0.0`.
8. **Implante** com `make deploy` e confirme `GET /v1/models` ([acima](#depois-de-subir-confirme-os-modelos)).

Os diretórios `0.0.0-dummy` saíram do repositório quando os artefatos reais entraram (`make dummy` os recria para
desenvolver o pipeline). A API carrega só a maior versão, e o gerador de cards e o gate de release também, então um
dummy esquecido ao lado de um `1.x` não é servido, mas pesa na imagem e na suíte: apague-o.

## Rollback por digest

Toda imagem publicada é imutável e tem um digest atestado. Para voltar:

```bash
make deploy DEPLOY_IMAGE=ghcr.io/<dono>/trinet-api@sha256:<digest anterior> GH_REPO=<dono>/trinet
```

Como os modelos estão dentro da imagem, **voltar a imagem volta também os modelos**. Para reverter só um modelo,
reverta o diretório da versão no git e deixe o CD gerar uma imagem nova. O digest anterior está na seção "Published image" do
resumo das execuções antigas do CD (ou em `docker image ls --digests ghcr.io/<dono>/trinet-api`, se a imagem ainda
estiver no host). Uma tag como `latest` não serve de ponto de retorno: ela se move.

## Interruptor SECURITY_GATE=warn

A variável de repositório `SECURITY_GATE` (padrão `block`) transforma os scanners bloqueantes (OSV-Scanner, Gitleaks,
zizmor, Hadolint e `trivy config`) em anotações enquanto um achado é triado. O `ci-gate` **continua** falhando por
qualquer outro motivo (lint, tipos, testes, drift do schema, artefatos).

```bash
gh variable set SECURITY_GATE --body warn  -R <dono>/trinet     # destrava o PR
gh variable set SECURITY_GATE --body block -R <dono>/trinet     # volta ao normal
```

Use por pouco tempo e com uma issue aberta para o achado: com `warn` um segredo ou uma dependência vulnerável pode
chegar à `main` com o check verde.

## Checar drift de classe prevista

A API não expõe histogramas por feature: o sinal disponível é a distribuição das **classes previstas**
(`trinet_predicted_class_total`) de `credit-default-mlp` e `candlestick-cnn`, comparada com uma referência pelo PSI
([ADR 0007](adr/0007-monitoramento-de-drift-psi.md)). O `volatility-mlp` não tem essa checagem. Com o Prometheus no ar
(`make observability`), este script (só biblioteca padrão) consulta a janela e calcula o PSI:

```python
import json
import math
import urllib.parse
import urllib.request

PROM = "http://127.0.0.1:9090"
MODEL = "credit-default-mlp"
WINDOW = "1d"
# Fração de cada classe PREVISTA no conjunto de teste do notebook (valores de exemplo: troque).
REFERENCE = {"no_default": 0.78, "default": 0.22}


def window_counts(prom: str, model: str, window: str) -> dict[str, float]:
    query = f'sum by (label) (increase(trinet_predicted_class_total{{model="{model}"}}[{window}]))'
    url = f"{prom}/api/v1/query?" + urllib.parse.urlencode({"query": query})
    with urllib.request.urlopen(url, timeout=5) as response:
        data = json.load(response)
    return {r["metric"]["label"]: float(r["value"][1]) for r in data["data"]["result"]}


def psi(expected: dict[str, float], observed: dict[str, float], floor: float = 1e-4) -> float:
    total_e, total_o = sum(expected.values()), sum(observed.values())
    if total_e <= 0 or total_o <= 0:
        raise ValueError("distribuição vazia: sem predições na janela?")
    value = 0.0
    for label in expected.keys() | observed.keys():
        e = max(expected.get(label, 0.0) / total_e, floor)
        a = max(observed.get(label, 0.0) / total_o, floor)
        value += (a - e) * math.log(a / e)
    return value


counts = window_counts(PROM, MODEL, WINDOW)
value = psi(REFERENCE, counts)
print(
    counts,
    f"PSI = {value:.4f}",
    "estável" if value < 0.10 else "atenção" if value < 0.25 else "mudança relevante",
)
```

Cuidados na leitura:

- Os limiares 0,10 e 0,25 são convenções da indústria, não calibrados para este projeto.
- A referência precisa vir do notebook (as classes previstas no teste), pois o contrato 1.0 não a carrega.
- Poucas predições na janela (dezenas) tornam o PSI ruidoso; o `increase()` também precisa de ao menos duas coletas do
  Prometheus dentro da janela.
- Na `candlestick-cnn`, a distribuição prevista é um **regime de mercado**: ela muda quando o mercado muda, sem que o
  modelo tenha piorado. No crédito, pode ser uma carteira diferente.

## Falhas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| A API não sobe: `no API key configured` | sem `secrets/api_key` montado nem `TRINET_API_KEY` | `make secrets` e suba de novo |
| A API não sobe: `the API key must have at least 16 characters` | chave curta | gere outra ([Segredos](#segredos)) |
| `ValidationError` de `Settings` ao subir | valor fora da faixa (por exemplo `TRINET_MAX_CONCURRENCY=100`) ou `TRINET_RATE_LIMIT` ilegível (`invalid rate limit ...`) | corrija conforme a [tabela](#configuração) |
| 401 `unauthorized` | `X-API-Key` ausente ou diferente da do arquivo; chave girada sem reiniciar | confira o cabeçalho; reinicie a API depois de girar |
| 404 `model_not_found` | nome errado (os atuais: `credit-default-mlp`, `volatility-mlp`, `candlestick-cnn`) | `GET /v1/models` |
| 404 `model_not_available` | artefato presente, mas não carregou | [seção própria](#404-model_not_available) |
| 413 `payload_too_large` | corpo acima de 1 MiB (JSON) / 6 MiB (upload) ou imagem acima do `max_bytes` do manifesto | reduza o lote ou a imagem; ajuste `TRINET_MAX_*` só se for intencional |
| 415 `unsupported_media_type` | `Content-Type` errado (tabular espera JSON; imagem espera `multipart/form-data`) ou imagem fora dos formatos do manifesto: o `candlestick-cnn` só aceita **PNG** | reenvie como PNG |
| 422 `invalid_input` | corpo que não é JSON (`invalid_json`), `instances` ausente (`missing`) ou vazio (`empty_batch`), lote acima de `TRINET_MAX_BATCH`, campo faltando (`required`), desconhecido (`unknown_field`), fora da faixa (`below_minimum`, `above_maximum`, `out_of_range`) ou do vocabulário (`not_in_vocabulary`), tipo errado (`type_error`), divisor de razão zero (`zero_denominator`) ou, em imagem, o campo `file` ausente | leia `errors[].loc`, `errors[].code` e `errors[].message` |
| 429 `rate_limited` | acima de `TRINET_RATE_LIMIT` | respeite o `Retry-After`; o balde é por chave válida ou, sem ela, por IP |
| 503 `overloaded` | todas as vagas de inferência ocupadas por mais de `TRINET_QUEUE_TIMEOUT_S` | espere o `Retry-After`; aumente `TRINET_MAX_CONCURRENCY` só se houver CPU (o compose limita a 2) |
| 503 `not_ready` em `/health/ready` | nenhum modelo serve | [404 model_not_available](#404-model_not_available) |
| 500 `internal_error` / `inference_failed` | erro inesperado; o corpo é genérico de propósito | procure o `request_id` nos logs (`unhandled_error`) |
| Contêiner `unhealthy` logo após subir | os modelos ainda carregam (o `start-period` é de 30 s) | aguarde; se persistir, veja os logs |
| `Bind for 127.0.0.1:8000 failed: port is already allocated` | porta ocupada | mude `TRINET_API_PORT` no `.env` |
| Alvo `trinet-api` **down** no Prometheus | a API não subiu, ou o Prometheus não está na mesma rede/projeto compose | `docker compose -f deploy/compose.yaml ps`; suba pelo `make observability` |
| `mounts denied` com Docker Desktop | o repositório está fora de `$HOME` e o diretório não é compartilhado | adicione a pasta em Settings > Resources > File sharing, ou use o Docker Engine |
| `docker login dhi.io` negado, ou a imagem DHI não é encontrada | você pediu `BASE=dhi` (opcional) sem conta ou token DHI | `make up` (a base padrão `slim` não precisa de login) |
| CI vermelho em `static (free-flows)` | um workflow passou a usar segredo, variável, registro, runner ou serviço fora da lista gratuita | leia a mensagem (`arquivo:linha`) e remova o uso; se for deliberado, mude o [ADR 0012](adr/0012-fluxos-gratuitos-e-sem-credenciais.md) e a lista de `tools/check_free_flows.py` |
| `make deploy` falha em `gh attestation verify` | digest que não veio do `cd.yml` deste repositório, `GH_REPO` errado, `gh` antigo ou sem login | confira a referência no resumo do CD e o `GH_REPO`; **não** contorne a verificação |
| CI vermelho em `ci-gate` | qualquer job obrigatório falhou | abra o job que falhou; para achado de scanner em triagem, veja o [interruptor](#interruptor-security_gatewarn) |
| CI vermelho em `schema-check` | `contract/schema/*.json` fora de sincronia com o contrato | `make schema` e commite os arquivos gerados |
| CI vermelho em artefatos | manifesto, SHA-256 ou tamanho inválido em `artifacts/` | `make check-artifacts` localmente |
