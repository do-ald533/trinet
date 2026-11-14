# Plano de refatoração (v2): domínios, DI e persistência assíncrona

- Status: **v2, revisada** (2026-11-14) por seis revisores independentes (arquitetura, dados e transações, tecnologia
  nova, sequenciamento, segurança, async e testes) e por um spike executável. Vira o ADR 0013 na fase R8.
- Origem: pedido do dono. Cada domínio vira uma pasta com
  `router → controller → service (facade) → creator/updater/deleter/indexer/getter → repository`, container de
  injeção de dependências, padrões de projeto e sistema assíncrono.
- Regras que continuam valendo: fluxos gratuitos e sem credencial externa
  ([ADR 0012](adr/0012-fluxos-gratuitos-e-sem-credenciais.md)); `trinet_contract` não muda; goldens, testes de contrato e
  o snapshot do OpenAPI das 5 operações atuais continuam passando (mudança só **aditiva** e consciente).

## 1. Premissas minhas (o dono pode vetar)

A API de hoje é de inferência e não tem entidade para criar, alterar ou apagar. Para a estrutura pedida ter corpo:

1. **`predictions`** é o único domínio **persistido**, com os cinco serviços (creator, updater, deleter, indexer, getter).
   Guarda um histórico das predições. **Nunca guarda a entrada.**
2. **`models`** é **somente leitura** (indexer e getter) sobre o `ModelRegistry` imutável em memória, atrás da mesma
   interface de repository. Não há `POST/PATCH/DELETE /v1/models`: os artefatos vão cozidos na imagem, a troca de versão é
   uma nova imagem (runbook), e uma API de escrita criaria uma segunda fonte da verdade que o serving ignoraria.
3. **`health`** é um domínio de sonda (router, controller, service).
4. SQLite com `aiosqlite` é o único dialeto. Postgres fica fora (nada em CI o executaria).

## 2. O que a revisão mudou em relação ao rascunho v1

| Cortado ou trocado | Motivo |
|---|---|
| Rotas de escrita de `models`, reconciler, `ModelState`, `/versions/*` | segunda fonte da verdade; só a maior versão é servida; raiz somente leitura |
| Postgres/asyncpg, gatilhos de `audit_log`, `If-Match`, loop de retenção, cursor opaco | sem feature que os use; dialeto nunca executado em CI; mais superfície |
| HMAC da entrada no banco | o banco **não guarda impressão da entrada**; o `id` é a chave de correlação |
| `UnitOfWork` e `Repository[T]` genérico em `core/base` | a `AsyncSession` já é a unidade de trabalho; o facade faz o commit |
| `wire()`/`@inject` do dependency-injector | é global ao processo (um app respondeu com o container de outro, verificado) |
| Inferência dentro da transação de escrita | com `BEGIN IMMEDIATE` serializaria todas as predições; infere primeiro, grava depois |
| `/health/ready` com `SELECT 1` retornando 503 | banco fora não pode tirar do ar a inferência; vira `degraded` |
| Fase 1 "só dependências" | quebra o deptry (DEP002); cada dependência entra com o código que a importa |
| Contratos do import-linter na fase final | um contrato que cita módulo inexistente aborta o `lint-imports`; reescrever junto com a movimentação |

Fica **mantido** (e não pode ser cortado): a rede de caracterização antes de mover código; a credencial de admin separada
com autenticação antes do parse do corpo; a inferência isolada do banco (best effort com prazo, ONNX em threads limitadas).

## 3. Decisões técnicas

| Tema | Decisão | Observação |
|---|---|---|
| ORM | **SQLModel** `~=0.0.50` sobre SQLAlchemy 2.1.8 | é o ORM que une SQLAlchemy e Pydantic; verificado no spike com `mypy --strict`. Pré-1.0, então fica isolado em `orm.py` e `repository.py`; as tabelas **não validam no `__init__`** (usar `model_validate`). Troca por SQLAlchemy puro custa só esses dois arquivos |
| DI | **dependency-injector** 4.49.1, um `Container` **por app** em `app.state.container`, **sem `wire()`/`@inject`** | providers `Singleton`, `Factory`, `Resource`, `ContextLocalResource`. Resolução por `Depends` com funções acessoras tipadas (`await container.x.async_()`) |
| Banco | SQLite WAL, dois engines: **escrita** (`BEGIN IMMEDIATE`, `pool_size=1`) e **leitura** (`BEGIN`, `query_only`) | `PRAGMA` por conexão: `foreign_keys=ON`, `busy_timeout`, `synchronous=NORMAL`, `temp_store=MEMORY`, `max_page_count` |
| Migrações | Alembic async (`render_as_batch`, `naming_convention`), executado com `anyio.to_thread` | `asyncio.run` do template falha dentro do loop; sem `fileConfig`; só migrações aditivas |
| Async | router→repository `async`; ONNX em thread com `anyio.CapacityLimiter` | handlers de erro viram `async def` |
| Testes async | plugin do `anyio` (`anyio_mode = "auto"`, `anyio_backend` de sessão = asyncio), `httpx2.AsyncClient` + `ASGITransport`, lifespan entrado à mão | sem `pytest-asyncio` |
| Entrega | ramo `refactor/domains`, **um commit por fase**, sem push (a credencial é sua) | permite bisect |

## 4. Domínios

### `predictions` (persistido, admin)

| Verbo e rota | Service | Notas |
|---|---|---|
| `POST /v1/models/{name}/predictions` | creator | **existe hoje**. Infere, solta o slot, grava o histórico (melhor esforço, prazo curto) e devolve o resultado igual ao de hoje, mais o cabeçalho opcional `X-Prediction-Id` (aditivo, documentado) |
| `GET /v1/predictions?limit&before_id&model` | indexer | `limit` 1..100 (padrão 20); sem `COUNT(*)`; pagina por `id` decrescente |
| `GET /v1/predictions/{id}` | getter | |
| `PATCH /v1/predictions/{id}` | updater | `outcome_label` **ou** `outcome_value` (finito), validado contra o `task` e as classes do manifesto; só uma vez (409 depois) |
| `DELETE /v1/predictions/{id}` e `DELETE /v1/predictions?older_than=` | deleter | exclusão física; em lote só com `older_than` |

Registro (`prediction_records`): `id INTEGER PRIMARY KEY AUTOINCREMENT`, `model_name`, `model_version`, `runtime`,
`batch_size`, `summary` (JSON de até 2 KiB, com `CHECK(length(CAST(summary AS BLOB)) <= 2048)`: contagem por rótulo e
probabilidade média no caso de classificação; mínimo, máximo e média na regressão; nunca eco por instância), `latency_ms`,
`created_at` (UTC, relógio injetado, `TypeDecorator` que rejeita data ingênua), `outcome_label`, `outcome_value`,
`outcome_at`. Sem chave estrangeira para modelos. Índices: PK, `created_at`, `(model_name, id)`.
Retenção padrão de 30 dias (`TRINET_HISTORY_RETENTION_DAYS`, máximo 365): expurgo **na subida**, em lotes de 1.000, com
checkpoint do WAL. Sem tarefa de fundo.

### `models` (somente leitura)

`GET /v1/models` (indexer) e `GET /v1/models/{name}` (getter), **respostas idênticas às de hoje**. O repository é um
adaptador sobre o `ModelRegistry`. Creator, updater e deleter não existem de propósito (documentado no ADR).

### `health`

`router → controller → service`. `/health/live` nunca toca o banco. `/health/ready` reporta `degraded` (HTTP 200) se o
banco estiver fora, com cache de 5 s; o formato da resposta não muda.

## 5. Estrutura e responsabilidades

```
api/src/trinet_api/
  __main__.py  healthcheck.py                 # ficam na raiz (Dockerfile CMD e HEALTHCHECK dependem deles)
  app.py   container.py                       # raiz de composição (fora do `core`, importam os domínios)
  core/        settings.py  errors.py  observability.py  security.py  database.py  repository.py
               http/ (middleware.py  problems.py  constants.py  openapi.py  scope.py)
  inference/   ports.py  registry.py  results.py  engine.py  limiter.py  factory.py  loading.py
               preprocessing.py  postprocessing.py  runtimes/ (onnx.py  numpy_mlp.py  litert.py)
  domains/
    health/       router.py  controller.py  schemas.py  service.py
    models/       router.py  controller.py  schemas.py  service.py  repository.py  services/ (indexer.py  getter.py)
    predictions/  router.py  controller.py  schemas.py  service.py  repository.py  orm.py
                  services/ (creator.py  updater.py  deleter.py  indexer.py  getter.py)
  migrations/    env.py  script.py.mako  versions/
```

| Camada | Faz | Não faz |
|---|---|---|
| **router** | `APIRouter`, caminho, verbo, `operationId`, tags, autenticação (`Security`), **leitura do corpo em bytes**, cabeçalhos de resposta, métricas e log `prediction`; **nunca declara parâmetro de corpo** | lógica de negócio; acessar service ou repository |
| **controller** | valida o payload com Pydantic estrito (`model_validate_json`), monta o comando, chama o facade, devolve o DTO. Livre de FastAPI e Starlette | sessão, ORM, I/O de rede |
| **service (facade)** | um método por caso de uso, **uma transação por método, o commit é dele** (antes de responder), devolve DTO, nunca objeto ORM | `select()`; HTTP |
| **services/** | classes de responsabilidade única (`creator`, `updater`, `deleter`, `indexer`, `getter`) com o repository injetado; só `flush()` | commit |
| **repository** | única camada com `select()`/`session`; `orm.py` define a tabela (SQLModel) | regra de negócio |

Detalhes que a revisão exigiu:

- O FastAPI faz o parse de um corpo declarado **antes** de resolver as dependências. Por isso os routers **não declaram
  corpo**; recebem `Request`, a dependência de auth roda primeiro, e o controller recebe bytes. Teste: sem chave e JSON
  malformado dá **401**, não 422.
- `POST predictions` é um caso de uso em dois passos (resolver o tipo de entrada pelo manifesto: 404 e 415 antes de ler o
  corpo; depois inferir). O passo 1 é uma dependência `Depends` do router; o passo 2 é o creator.
- Constantes compartilhadas (`JSON`, `MULTIPART`, `MODEL_VERSION_HEADER`, `ModelName`, caminhos de saúde) ficam em
  `core/http/constants.py`, senão `models` e `predictions` se importam. A tag `models` e o `{name}` do caminho ficam como
  estão (o rótulo `model` das métricas e o snapshot dependem deles).
- `_build_openapi` deixa de ter o caminho fixo; localiza a operação por `operationId`. `core` não importa domínios.
- Erros novos (`PredictionNotFound` 404, `OutcomeConflict` 409, `InvalidOutcome` 422, `HistoryUnavailable`) entram na
  árvore `TriNetError` em `core/errors.py` e no `_STATUS_BY_ERROR`; um teste garante que **toda** subclasse tem status.
  `IntegrityError` vira 409 genérico; erros de lock/pool/`SQLITE_FULL` viram 503 com `Retry-After`; `hide_parameters=True`.

## 6. Container e escopo por requisição (padrão testado no spike)

```python
class Container(containers.DeclarativeContainer):
    settings = providers.Singleton(Settings)                       # ou Object(settings) vindo de create_app
    metrics = providers.Singleton(Metrics)
    registry = providers.Singleton(load_registry, settings=settings)  # imutável
    limiter = providers.Singleton(AsyncConcurrencyLimiter, settings=settings)
    engine = providers.Singleton(PredictionEngine, registry=registry, limiter=limiter, clock=...)
    write_engine = providers.Resource(init_engine, settings=settings, writer=True)   # async gen, dispose no shutdown
    read_engine = providers.Resource(init_engine, settings=settings, writer=False)
    write_session = providers.ContextLocalResource(init_session, engine=write_engine)
    read_session = providers.ContextLocalResource(init_session, engine=read_engine)
    predictions_repository = providers.Factory(PredictionsRepository, session=write_session)
    predictions_creator = providers.Factory(PredictionCreator, engine=engine, repo=predictions_repository, ...)
    predictions_service = providers.Factory(PredictionsService, session=write_session, creator=..., ...)   # Facade
    predictions_controller = providers.Factory(PredictionsController, service=predictions_service)
```

Regras verificadas (cada uma tem teste no spike):

1. `container = Container()` **dentro de `create_app`**, guardado em `app.state.container`; nunca no módulo. Teste de
   isolamento com dois apps vivos.
2. `ContextLocalResource` vive num `ContextVar`: abrir e fechar na mesma task. Um middleware ASGI **puro** (nunca
   `BaseHTTPMiddleware`) chama `shutdown()` dos escopos no `finally`.
3. `init_resources(providers.Resource)` (não o padrão) no lifespan, e `shutdown_resources()` no `finally`, com um
   `AsyncExitStack`; `engine.dispose()` sempre.
4. Cadeia async inteira: só rotas `async def`. Acessor por rota: `await request.app.state.container.x.async_()`.
5. Override em teste: chamar `provider.enable_async_mode()` **antes** de `override(providers.Object(fake))`, e
   `reset_async_mode()` depois; `container.reset_override()` desfaz todos, inclusive os de fixture.
6. Engine em memória de teste: `StaticPool` com `BEGIN` simples (o `BEGIN IMMEDIATE` falha com escritas concorrentes no
   `StaticPool`); concorrência real em arquivo `tmp_path`.
7. `expire_on_commit=False`; DTO construído antes de fechar a sessão; `col()` para `in_`/`desc`; `cast(Any, ...)` só no
   `isolation_level` do DBAPI. Um commit que falha devolve erro (não 201), e a sessão fecha antes do 500.

## 7. Padrões (com onde e por quê)

| Padrão | Onde |
|---|---|
| Facade | `PredictionsService`, `ModelsService`, `HealthService` |
| Singleton (provider do container) | settings, métricas, registry, limiter, `PredictionEngine`, engines do banco (não é `__new__`) |
| Factory | providers `Factory` do grafo por requisição; `inference/factory.py::build_runtime` |
| Strategy | `ModelRuntime` (ONNX, NumPy MLP, LiteRT) e as primitivas de pré-processamento |
| Repository | `Repository` (Protocol) com duas implementações: SQL (`predictions`) e em memória (`models`) |
| Adapter | `ModelsRepository` sobre o `ModelRegistry`; mapeadores ORM ⇄ DTO |

De fora de propósito: Abstract Factory, Observer, Singleton por `__new__`, State, Chain of Responsibility explícita, CQRS.

## 8. Persistência e operação

- Default `TRINET_DATABASE_URL=sqlite+aiosqlite:////data/trinet.db`; esquema de URL restrito a `sqlite+aiosqlite`;
  vazio desliga a persistência (rotas de histórico não montadas). Banco indisponível na subida **não derruba a API**:
  inferência segue, `/health/ready` fica `degraded`, métrica `trinet_database_up`.
- Dockerfile: `install -d -o 65532 -g 65532 -m 0700 /data` antes do `USER`; compose com **volume nomeado** em `/data`
  (bind mount quebra o WAL em NTFS/exFAT/Docker Desktop); `smoke-image.sh` e `test_container.py` ganham `/data` em tmpfs.
  Instância única, declarada como invariante. `.gitignore`/`.dockerignore`: `*.db`, `*.sqlite*`, `*-wal`, `*-shm`.
- Migração automática na subida (SQLite), com `VACUUM INTO` de um instantâneo quando há revisão pendente; `python -m
  trinet_api db backup` (API de backup do SQLite) e `make backup`; restaurar com o container parado e sem `-wal`/`-shm`.
- Histórico em melhor esforço: `anyio.move_on_after(prazo, shield=True)` ao redor do insert, falha e prazo viram
  `trinet_history_write_failures_total`; **nunca** `BackgroundTasks`.
- Teto de tamanho: `max_page_count` (~256 MiB) e parada de inserção a 80% com expurgo.

## 9. Segurança e privacidade

1. `TRINET_ADMIN_API_KEY_FILE` (opcional, ≥ 32 caracteres ASCII, recusada se igual à chave de predição). Sem ela, as rotas
   de histórico **não são montadas**; o creator continua gravando. Balde próprio no rate limit. `make secrets` gera a chave.
2. Corpo de PATCH ≤ 1 KiB, `extra="forbid"`, `strict=True`, `max_length` em toda string, sem dict livre. Ordenação por
   enum→coluna; `sqlalchemy.text` proibido fora das migrações (regra do ruff).
3. Handler de validação: cortar cada `loc` em 64 caracteres e limitar a 50 itens.
4. Upload: semáforo ≤ 4 e `asyncio.timeout(10)` na leitura (o `/tmp` tem 64 MiB).
5. Reverte o corte do `PredictionRecorder` (ADR 0002). **Atualizar `SECURITY.md`, `architecture.md` e a nota LGPD do README:**
   a entrada nunca é gravada, mas a **saída resumida** passa a ser, com retenção de 30 dias.
6. **Pendência herdada, fora do escopo:** o log `prediction` guarda `input_sha256` simples (SHA-256 do corpo), que é um
   oráculo de pertencimento para linhas do dataset público. Trocar por HMAC muda o significado do campo; fica para uma decisão
   sua (registrada no ADR 0013).

## 10. import-linter (sintaxe verificada na 2.16)

- Reescrever os 6 contratos atuais **na mesma fase da movimentação**, preservando: `inference` sem fastapi/starlette/
  pydantic_settings/structlog/prometheus/uvicorn; entidades só com stdlib; HTTP nunca importa `inference.runtimes`;
  só o `loading` instancia runtimes; `trinet_contract` não importa `trinet_api`. Incluir `starlette`, `sqlalchemy`,
  `sqlmodel`, `aiosqlite`, `alembic`, `anyio` e `dependency_injector` em `root_packages`.
- Camadas por domínio (`layers`, siblings com `|`, **sem parênteses**): `router | controller | service |
  services.creator | services.updater | services.deleter | services.indexer | services.getter | repository`, com camadas
  opcionais onde o domínio não as tem (`health`, `models`). `layers` não impede pular camada: contratos `forbidden` extras
  (router não importa service/repository; controller não importa repository nem services).
- `independence` entre `domains.*` (sem `Provide["..."]` por string, porque o container fica fora dos domínios).
- Bibliotecas externas: `forbidden` com `allow_indirect_imports = true` e fontes explícitas (`domains.*.router`);
  `fastapi`/`starlette` só em `router` e `core/http`; `sqlalchemy`/`sqlmodel` só em `orm`, `repository`, `core/database`.
- Globs de ruff para `N818` (`**/errors.py`, `**/security.py`) no mesmo commit.

## 11. Testes

- **R0, antes de mover qualquer coisa** (`api/tests/characterization/`, verde na `main` atual, rodada em todo commit
  de movimentação): snapshot do OpenAPI **mais um hash do corpo cru** (o snapshot usa `sort_keys`); tabela de erros (status,
  `code`, URN, cabeçalhos; `validation_error`, `method_not_allowed` e `http_<n>` não têm teste hoje); transcrição HTTP por
  rota num subprocesso (imune a mudança de caminho); famílias e rótulos Prometheus (conjunto **de superconjunto**: métricas
  novas são permitidas, as atuais não mudam); eventos de log (`http_request`, `prediction`, `unhandled_error`,
  `rate_limited`) com o campo `logger`; ordem dos middlewares (413 × 429 × 401); goldens e entrypoints; superfície do
  `Settings`; a rede deve falhar em pelo menos 8 mutações roteirizadas.
- **Migração dos 600 testes**: 222 intactos desde que `create_app(settings, *, registry=None)` continue público (ganha
  `container=`); 363 só trocam caminho; `unit/test_service.py` é **renomeado**, não reescrito (`PredictionService` →
  `PredictionEngine` num commit isolado). `_build_runtime` continua público em `factory.py` (3 arquivos de teste o usam).
- **Pirâmide nova**: repository com SQLite real (arquivo `tmp_path` para concorrência); teste de contrato do repository (fake
  × SQL × em memória); services com repository falso; facade com espiões; controller (parse, `loc` cortado, 401 antes de
  422); router (um por rota); Hypothesis com máquina de estados do repository; teste de isolamento de dois apps;
  Schemathesis com as rotas novas (sem `with TestClient` aninhado, que cria dois lifespans em dois loops).
- **Higiene**: `filterwarnings = error::pytest.PytestUnraisableExceptionWarning`; `concurrency = ["thread", "greenlet"]`
  no coverage; `omit` de `migrations/`; `structlog.contextvars.clear_contextvars()` autouse; `git grep` por caminhos antigos
  na documentação como gate.

## 12. Fases (um commit cada; gate completo verde em todas)

Gate: `UV_LOCKED=1 make lint typecheck imports deptry free-flows cov schema-check model-cards-check
check-artifacts-release`, mais a rede de caracterização; fases que tocam o container: `make build smoke integration`.
Cobertura ≥ 95%, nenhum teste apagado. Reversão: `git reset --hard` ao commit anterior (nada foi empurrado).

| Fase | Conteúdo | Pronto quando |
|---|---|---|
| **R0** rede | testes de caracterização e lacunas (seção 11) | verde na `main`; falha em ≥ 8 mutações; estável em 3 execuções e com `-n auto` |
| **R1** mover | `git mv` + reescrita de imports (mapa abaixo), contratos do import-linter, globs do ruff, `FACTORY`, caminhos de teste | `git log --follow` funciona; diff de testes só de linhas de import; `git grep` de caminhos antigos vazio |
| **R2** dividir | `entities` → `registry` + `results`; `service` → `engine` + `limiter`; `loading` → + `factory`; `routers` e `schemas` → domínios; `interface/` some | diff do snapshot vazio |
| **R3** container | `Container` por app, escopo por requisição, `app.state.container`, `dependency-injector` declarado e usado | teste de dois apps; sem `wire(`/`@inject`; deptry verde |
| **R4** camadas | controller, facade e services **async** para as 5 operações atuais; `models` somente leitura; `health`; contratos por domínio | caracterização idêntica |
| **R5** banco | deps (`sqlalchemy[asyncio]`, `sqlmodel`, `aiosqlite`, `alembic`, `greenlet`), `core/database.py`, migrações, settings, Dockerfile, compose, smoke, política de degradação, readiness `degraded` | imagem sobe com banco desligado e com `/data`; wheel contém as migrações; `upgrade`/`downgrade` em SQLite vazio |
| **R6** histórico | `predictions` completo, chave de admin, expurgo, `X-Prediction-Id`, snapshot do OpenAPI atualizado (aditivo) | subconjunto antigo do snapshot intacto; Schemathesis verde com as rotas novas; rotas ausentes sem chave de admin |
| **R7** async nativo e defeitos | `CapacityLimiter`, handlers `async`, prazo de upload, `warnings` não thread-safe, chave ASCII, `Retry-After` do `ready` 503, `fullmatch` | testes de 503 e de limites passam **sem alteração** |
| **R8** docs | ADR 0013 (status que substitui partes dos ADRs 0002 e 0011), `architecture.md`, README, runbook, SECURITY, deploy/README, este plano marcado aceito | `git grep` de caminhos antigos vazio |

**Mapa da movimentação (R1, protótipo já verde numa cópia: 591 testes, ruff, mypy, import-linter):**
`settings.py`→`core/settings.py`; `domain/errors.py`→`core/errors.py`; `domain/{ports,entities}.py`→`inference/`;
`application/{pre,post}processing.py`→`inference/`; `application/service.py`→`inference/engine.py`;
`infrastructure/{loading.py,runtimes/*}`→`inference/`; `infrastructure/observability.py`→`core/observability.py`;
`interface/http/{middleware,problems}.py`→`core/http/`; `security.py`→`core/security.py`; `app.py` e `dependencies.py`→raiz
(`app.py`) e `core/http/` conforme a fase. `routers.py` e `schemas.py` ficam até a R2. `FACTORY` vira
`trinet_api.app:create_app`.

## 13. Invariantes (a revisão mapeou; o teste de caracterização protege)

Ordem dos middlewares (de fora para dentro): RequestContext, RateLimit, BodyLimit, rotas. Nenhum handler para `Exception`.
`BodyTooLarge` propaga sem embrulho (nada de `BaseHTTPMiddleware` ou `TaskGroup`). Cabeçalhos de segurança e `X-Request-ID`
em toda resposta. `application/problem+json` com `type=urn:trinet:problem:<code>` e os códigos atuais. Auth (401) antes da
validação de caminho e da busca do modelo; depois 404, 415 e o parse. `create_app` levanta `ConfigurationError`; sem diretório
de artefatos o app sobe com `live` 200 e `ready` 503. Loggers `trinet_api.http` e `trinet_api.predictions` mantêm o nome
(os campos `logger` de `loading`, `engine`, `app` e `onnx` mudam com a movimentação: fixar ou aceitar de forma explícita).
Operações, `operationId` e nomes de componentes do OpenAPI das 5 rotas atuais; `info.version` do metadado do pacote.

## 14. Riscos e itens em aberto

- **Para o dono decidir:** (1) `models` somente leitura está bem? (2) o histórico guarda só o resumo da saída, e a retenção
  padrão de 30 dias serve? (3) `input_sha256` simples nos logs: trocar por HMAC? (4) SQLModel 0.0.x (pedido) ou SQLAlchemy puro?
- Não verificado: desconexão do cliente no meio do commit; `BaseHTTPMiddleware` com `ContextVar` (evitado por regra);
  `ContextLocalSingleton`; Docker/`make build smoke` na cópia de teste da movimentação.
- Defeitos latentes já achados no código atual (corrigidos na R7): `warnings.catch_warnings()` não é thread-safe na 3.13;
  chave de API não-ASCII nunca autentica; `Retry-After` do `ready` 503 está documentado e não é enviado; semáforo
  `threading` dentro do pool padrão de 40 threads ignora `queue_timeout_s` (medido: 2 a 3 s em vez de 1 s).
