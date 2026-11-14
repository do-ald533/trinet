# ADR 0002: Hexagonal enxuta, dirigida pelo manifesto

- Status: aceito
- Data: 2026-11-10
- Origem: [PLANO](../../PLANO.md) §5.2 e §14

## Contexto

Os três modelos têm a mesma forma: entra um tensor, sai um tensor. A primeira versão do plano previa quatro portas,
quatro casos de uso, Template Method e um repositório de modelos. O debate apontou que isso era cerimônia sem
consumidor: cada porta teria uma só implementação e cada caso de uso repetiria o mesmo corpo.

## Decisão

Uma arquitetura hexagonal com **uma porta** e comportamento vindo dos dados:

```text
interface/http  ->  infrastructure  ->  application  ->  domain
(FastAPI, DTOs)     (loading, runtimes,   (PredictionService,  (ModelRuntime, entidades,
                     observability)        funções puras)        erros)
```

- `domain/ports.py`: o `Protocol` `ModelRuntime` com dois membros, `kind` e `predict(batch)`. O contrato do
  `predict` (float32, forma `(n, ...)`, saída finita, entrada não mutada, chamadas concorrentes seguras) é verificado
  por uma suíte compartilhada por todos os adaptadores.
- `domain/entities.py` e `errors.py`: `ModelBundle`, `UnavailableModel`, `ModelRegistry` (leitura apenas),
  resultados imutáveis e a hierarquia `TriNetError`.
- `application/`: `PredictionService` escolhe o pré e o pós-processamento pelo `manifest.input` e `manifest.output`
  (um `match`, sem subclasses); `preprocessing.py` e `postprocessing.py` são funções puras, usadas também pelos
  testes golden.
- `infrastructure/`: `loading.py` (registro dirigido por dados), `runtimes/` (ONNX, NumPy, LiteRT) e
  `observability.py`.
- `interface/http/`: `create_app()` é a raiz de composição; o `lifespan` carrega os modelos; DTOs Pydantic separados
  das entidades; erros em Problem Details (RFC 9457); API key; middlewares puros ASGI.

As fronteiras são **verificadas no CI** pelo import-linter, com 6 contratos em `pyproject.toml`: camadas
`interface > infrastructure > application > domain`; o domínio importa só biblioteca padrão, NumPy e o contrato; a
aplicação não importa web, runtime, logging nem métricas; a interface não importa os adaptadores; só o carregador
instancia adaptadores; o contrato não conhece a API.

Cortado no debate: `PredictionRecorder`, `ModelRepository`, casos de uso por modelo, Template Method e um
`TFSavedModelRuntime` (poria TensorFlow na imagem).

## Consequências

- **Um modelo novo é um artefato novo.** Sem editar código, desde que use as primitivas do contrato (OCP). É o
  exemplo para a apresentação.
- Testes sem modelo real: o `PredictionService` roda com um `FakeRuntime` que fica em `api/tests/fakes.py`, fora de
  `src/`, logo fora da imagem.
- Uma tarefa nova (por exemplo, texto) exige mudar o contrato, o serviço e o notebook, porque o `match` do serviço
  e as primitivas são fechados de propósito.
- O processo atende por completo só depois do `lifespan`: o uvicorn abre o socket depois do startup, então enquanto
  os modelos carregam as sondas não conseguem conectar. O `HEALTHCHECK` da imagem tem `start_period=30s`.
- Há uma sessão ONNX por modelo, compartilhada por todas as threads, e um limitador de concorrência
  (`TRINET_MAX_CONCURRENCY`) que devolve 503 com `Retry-After` quando a fila excede `TRINET_QUEUE_TIMEOUT_S`.
