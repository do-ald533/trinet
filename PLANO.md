# TriNet — Plano do Projeto (T2 · Aprendizado Profundo · CIC407)

> **Status:** v1.0 (consolidado após debate de 3 especialistas) · **Data:** 09/11/2026
> **Escopo decidido com o usuário:** notebook + API completos neste ciclo (até 19/11), deploy via Docker, treino em **CPU** local, CI/CD com segurança robusta (SAST, SCA, code smell, segurança da imagem).
>
> **v1.1 (10/11/2026) — decisões do usuário que alteram a v1.0:**
> - **Treino na CPU** (i7-11370H, 4C/8T, ~8 GB livres), não na GPU. Pacote `tensorflow==2.21.0` (sem `[and-cuda]`); itens de driver NVIDIA/TF32/Colab saem do caminho crítico. A CNN é dimensionada para CPU (resolução e largura medidas na 1ª época; ver §4.1).
> - **Repositório público** desde já: CodeQL, rulesets, push protection e `actions/attest` gratuitos. Nomes e RA ficam fora do repositório (só na cópia do Canvas).
> - **Credenciais** (Kaggle, `dhi.io`, segredos do GitHub) configuradas manualmente pelo usuário.
> - **Nomes e RA**: adiados.
> - `uv` atualizado localmente para 0.13.7; o contrato (`contract/`) define `manifest.json` e `golden.json` com schemas versionados em `contract/schema/`.
> Versões verificadas em **09/11/2026** no PyPI (`pypi.org/pypi/<pkg>/json`) e na API de releases do GitHub. Itens que não puderam ser confirmados estão marcados **⚠️ não verificado** e viram tarefa de validação no dia 1.

---

## 0. Resumo executivo

Um repositório (`trinet`, "três redes") com dois componentes ligados por um **contrato de artefato**:

1. **Notebook (entrega avaliada: 7 pts de código + 3 de apresentação).** Três redes construídas camada a camada com `tf.keras.layers`: classificação tabular (Dry Bean), regressão tabular (California Housing) e CNN de imagens (Intel Scenes). Inclui EDA, preparação **visível**, justificativas em células de texto, ablações curtas, treino/validação/teste, previsões em novas instâncias e as métricas exigidas. O apêndice exporta cada modelo como `model.onnx` + `manifest.json` + golden samples.
2. **API de inferência (ML Engineering).** FastAPI com hexagonal **enxuta** (uma porta, vários adaptadores), dirigida pelo manifesto. Serve os 3 modelos via ONNX Runtime, sem TensorFlow na imagem, e testa o *training-serving skew* com golden samples. Inclui observabilidade, container endurecido e CI/CD com gates de segurança e cadeia de suprimentos (SBOM + proveniência atestada).

**Princípio do plano:** o notebook é prioridade absoluta, e a API não pode roubar tempo dele. Ferramenta que não tem dono, gate ou plateia sai do plano.

---

## 1. Análise do enunciado → requisitos rastreáveis

| # | Requisito (PDF) | Onde é atendido |
|---|---|---|
| R1 | 3 problemas supervisionados: classificação, regressão, visão | Notebook §1, §2, §3 |
| R2 | TensorFlow; **sem arquiteturas prontas**; camadas via `tf.keras.layers.*` | `Input/Dense/Conv2D/MaxPooling2D/Flatten/Dropout/BatchNormalization/Normalization/Rescaling`. Proibido: `keras.applications`, transfer learning, KerasHub |
| R3.1.1 | Preditoras × alvo; **normalização/padronização obrigatória** + outras preparações | `Normalization.adapt()` em célula **visível** (mostra média/desvio antes e depois) e depois embutida no modelo; imputação, one-hot, dedup, padronização do alvo |
| R3.1.2 | Justificar ativação das ocultas, profundidade/largura, ativação de saída | Célula "Justificativa de arquitetura" por rede, com evidência de ablação |
| R3.1.3 | Justificar loss, taxa de aprendizado e métrica | Célula "Justificativa de compilação" por rede |
| R3.1.4 | Treino/validação/teste | Tabular 70/15/15 estratificado; Intel `seg_train` → treino/val (mesmo `seed`), `seg_test` → teste. O teste é tocado **uma única vez** |
| R3.1.5 | Previsões em **novas instâncias** | Linhas **reais** reservadas antes de tudo + cenários "e se"; Intel: `seg_pred` (sem rótulo) |
| R3.1.6 | Previsões no teste | `model.predict(test)` |
| R3.2 | RMSE, MAE, margem relativa (MAE/média, RMSE/média) | Em dólares, média do **teste**; também nas linhas abaixo do teto do alvo |
| R3.3 | Matriz de confusão (acertos por classe) + precision/recall/F1 | Heatmap com diagonal destacada + `classification_report` |
| R3.4 | CNN: acurácia | Acurácia + (extra) matriz de confusão e exemplos de erro |
| R4 | `.ipynb` com nomes completos e RA; Canvas até 19/11 | Capa; checklist de entrega |
| R5 | Apresentação oral (semana de 16/11), todos dominam | Divisão em §11 + simulado de perguntas em 15/11 |

**Pesos:** classificação 2 · regressão 2 · CNN 3 · apresentação 3.
**Calendário:** a apresentação pode cair **antes** de 19/11, então o notebook precisa estar apresentável até **15/11**.

---

## 2. Datasets

| Problema | Dataset | Tamanho | Por que serve à rubrica | Verificação |
|---|---|---|---|---|
| Classificação | **Dry Bean** (UCI #602) | 13.611 × 16 features numéricas, 7 classes | Multiclasse desbalanceada (justifica F1-macro e ablação de `class_weight`); matriz 7×7 rica; padronização essencial | ✅ CC BY 4.0, DOI 10.24432/C50S4B · ⚠️ duplicatas (estudo secundário cita 68) → `df.duplicated().sum()` |
| Regressão | **California Housing Prices** (Kaggle `camnugent/california-housing-prices`) | 20.640 × 10 | 207 NaN em `total_bedrooms` (imputação), categórica `ocean_proximity` (one-hot), alvo censurado no teto | ✅ 20.640×10 e 207 NaN · ⚠️ licença da página Kaggle · ⚠️ teto (~500k) a medir na EDA |
| Visão | **Intel Image Classification** (Kaggle `puneet6060/intel-image-classification`) | ~25k imagens 150×150, 6 classes | Viável em 4 GB; `seg_pred` sem rótulo atende R3.1.5; pares confundíveis (glacier×mountain, buildings×street) | ✅ classes e ~25k · ⚠️ contagem por pasta → contar após download · ⚠️ licença "Other": uso acadêmico, **não redistribuir imagens** |

- **Download:** `kagglehub 1.0.3` (token Kaggle **só local**, nunca no CI nem no repo) e o zip direto da UCI.
- **Fallback do housing:** o `fetch_california_housing` do sklearn **não serve**, porque não tem NaN nem `ocean_proximity` e mataria o argumento das "outras preparações". Fallback: CSV do repositório `ageron/data` (⚠️ conteúdo e licença a verificar).
- **Fallbacks gerais:** classificação → Bank Marketing (UCI); imagens → CIFAR-10.
- O CI **nunca** baixa dataset.

---

## 3. Arquitetura geral

```
trinet/
├─ notebook/  (projeto uv PRÓPRIO, lock próprio; Python 3.13 + TF 2.21 GPU)
│    T2_TriNet.ipynb ─► treino fp32 ─► avaliação ─► apêndice de export
│                                                     │
│                                                     ▼
├─ artifacts/<modelo>/<versão>/   model.onnx · manifest.json · golden.json · results.json
│                                                     │  (commitados no git; *.onnx binary)
├─ contract/  trinet-contract (só Pydantic) ◄─────────┤  valida o manifesto dos dois lados
│                                                     ▼
├─ api/       FastAPI ─► PredictionService ─► porta ModelRuntime ─► OnnxRuntime | NumpyMlp | LiteRT
│               │
│               └─► logs JSON (structlog) · /metrics em porta interna (Prometheus)
│
├─ deploy/    Dockerfile multi-stage (DHI) · compose (api + profile observability)
└─ .github/   CI de PR (qualidade+testes+segurança) · CD em main (GHCR por digest + Trivy + SBOM + attest)
```

**Por que ONNX Runtime na API?**
- Imagem sem TF: menor superfície de ataque, build e cold start mais rápidos.
- A API **nunca desserializa `.keras`/`.h5`**. Houve vários CVEs de execução de código no carregamento de modelos Keras, inclusive bypass de `safe_mode`: CVE-2025-1550, CVE-2025-8747, CVE-2025-9905 e GHSA-c9rc-mg46-23w3. O `.keras` nem entra no git.
- A API também **não importa o pacote `onnx`** (CVE-2026-34447, leitura de arquivo via `external_data`, afeta só `onnx` < 1.21.0) e recusa modelos com dados externos.

**Riscos do export (consenso do debate):**
- `Model.export(format="onnx")` exige política de dtype **float32**, conforme o docstring de `keras/src/export/onnx.py` v3.15.1.
- No backend TF, o export usa tf2onnx, que tem o patch "TODO: Remove this once tf2onnx supports numpy 2".
- O README do tf2onnx diz "looking for a new maintainer" e testa só TF 2.13–2.15 e opset 14–18.

Daí a **escada de fallback** da §5.4.

---

## 4. Componente 1: Notebook

### 4.1 Ambiente
- **Python 3.13**: TF 2.21.0 tem wheels cp310–cp313 e não tem 3.14.
- Pacotes: `tensorflow[and-cuda]==2.21.0` (CUDA 12 via wheels NVIDIA), `keras==3.15.1`, `scikit-learn==1.9.1`, `pandas==3.1.1`, `numpy==2.5.4`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `jupyterlab==4.6.6`, `ipykernel==7.4.0`, `kagglehub==1.0.3`.
- Ligação com o contrato: `[tool.uv.sources] trinet-contract = { path = "../contract" }`, importado **só** no apêndice de export, dentro de `try/except ImportError`. Assim o notebook continua autocontido para o professor.
- **GPU:** RTX 3050 Mobile, 4 GB.
  - ⚠️ Hoje o `nvidia-smi` falha: o `nvidia-driver-580-open` está instalado, mas o módulo do kernel não carrega (provável Secure Boot/MOK ou reboot pendente). **Corrigir no dia 1** (ação do usuário).
  - Plano B da CNN: Colab/Kaggle com T4.
- **Precisão: float32, sem `mixed_float16`.** O TF32 já vem ligado por padrão em Ampere para matmul/conv (doc `enable_tensor_float_32_execution`). Os golden outputs são gerados em CPU, ou com TF32 desligado, para não contaminar a paridade.
- `tf.data` com `cache()` e `prefetch(AUTOTUNE)`; `set_memory_growth`.
- **Medir a 1ª época da CNN no dia 1.** Se passar de ~90 s, usar 128×128 ou T4.
- **Reprodutibilidade:** `keras.utils.set_random_seed` antes de cada modelo, resultados em média ± desvio de 3–5 seeds e célula final com versões e hardware. `enable_op_determinism` **não** é usado na CNN: serializa o `tf.data` e não garante reprodução entre hardwares.
- Outputs enxutos (`verbose=0` nos loops de ablação). Teste final em ambiente limpo com **Restart & Run All**.

### 4.2 Estrutura (mesmo esqueleto nas 3 seções)
0. Capa (nomes completos + RA), objetivo, setup e versões.
1. **Dry Bean.**
   - Reservar 5 linhas reais por classe como "novas instâncias" (antes de dedup e split).
   - Dedup → EDA (classes, correlação, outliers) → split estratificado → `Normalization.adapt(train)` visível.
   - Ablação → arquitetura final → justificativas → treino (curvas).
   - Teste: matriz de confusão + `classification_report` → novas instâncias.
2. **California Housing.**
   - Reservar 5 linhas reais + 5 cenários "e se" (editados a partir de linhas reais).
   - EDA (teto do alvo, NaN, categórica) → split estratificado por faixas de `median_income`.
   - Imputação (mediana **do treino**) + one-hot (vocabulário do treino) → padronização de X e de **y** → baselines (média do treino e regressão linear).
   - Ablação → modelo final.
   - RMSE, MAE, MAE/média, RMSE/média em dólares, também abaixo do teto. Gráficos previsto × real e resíduos.
3. **Intel Scenes.**
   - `image_dataset_from_directory`: treino/val com o mesmo `seed`; **teste com `shuffle=False`**, senão a matriz de confusão sai desalinhada.
   - Amostras por classe → ablação de 4 configs → CNN final.
   - Acurácia no teste (+ matriz e erros) → previsões em 12 imagens de `seg_pred` com classe e probabilidade.
4. Conclusões, comparação com os baselines, limitações.
5. **Apêndice (fora da área avaliada):** export + manifesto + golden + quality gate.

### 4.3 Arquiteturas iniciais e ablações

| Rede | Ponto de partida | Saída / Loss / Métricas | Ablação (substitui KerasTuner) |
|---|---|---|---|
| Dry Bean (MLP) | `Input(16)` → `Normalization` → `Dense(64, relu)` → `Dropout(0.2)` → `Dense(32, relu)` → `Dense(7, softmax)` | `categorical_crossentropy` (rótulos one-hot, para usar `keras.metrics.F1Score(average="macro")` no `compile` e no `monitor`) · Adam 1e-3 | Grade largura × profundidade (3 seeds, heatmap) + com/sem `class_weight`, decidido pelo F1-macro na validação. Teto esperado ~92–93% de acurácia (literatura secundária) |
| Housing (MLP) | `Input(n)` → `Normalization` → `Dense(128, relu)` → `Dense(64, relu)` → `Dense(32, relu)` → `Dense(1)` → `Normalization(invert=True)` (devolve dólares) | linear · **Huber** com `delta` ≈ 0,3–0,5 desvio do alvo padronizado **vs MSE** (comparar) · RMSE, MAE · Adam 1e-3 + `ReduceLROnPlateau` | Grade pequena + Huber×MSE + baselines |
| Intel (CNN) | `Input(150,150,3)` → `Rescaling(1/255)` → aug (`RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)`) → 4 blocos [`Conv2D(f,3,relu,same)` ×2 (+`BatchNormalization`) → `MaxPooling2D`], f = 32/64/128/128 → `Flatten` (10.368) → `Dense(128, relu)` → `Dropout(0.5)` → `Dense(6, softmax)` | `sparse_categorical_crossentropy` · accuracy · Adam 1e-3 + `EarlyStopping(restore_best_weights)` | 4 configs: base · +augmentation · +dropout · mais rasa. Observação: o `Dense(128)` pós-`Flatten` concentra ~70% dos ~1,9M parâmetros e é o principal vetor de overfitting. Acurácia esperada ~85–90% ⚠️ não verificado |

### 4.4 Template das justificativas
```
### Justificativa — Arquitetura (Rede X)
- Ativação das ocultas: ReLU porque … (evidência: curva/ablação na célula N)
- Profundidade × largura: [64, 32] porque … (heatmap da grade, overfitting na validação)
- Saída: softmax (classes exclusivas → distribuição) | linear (+ Normalization invertida → dólares)
### Justificativa — Compilação
- Loss … · Taxa de aprendizado … (curvas, ReduceLROnPlateau) · Métrica … (desbalanceamento → F1-macro)
```

### 4.5 Export: o contrato com a API
- **Clone de inferência em float32**, sem as camadas de augmentation, com pesos copiados. Depois `model.export(".../model.onnx", format="onnx", opset_version=17)`.
- **`manifest.json`** (modelo Pydantic `trinet-contract`, `schema_version`, `extra="forbid"`) contém:
  - nome, versão semver, git SHA, `task` (`classification | regression | image_classification`);
  - `tensor_spec` (nomes, ordem, dtype, shape);
  - `preprocessing` com **3 primitivas fechadas**: imputar constante, one-hot por vocabulário, razão A/B. Constantes só do treino. Se as features derivadas do housing custarem mais de ~1 h, cortá-las: imputação + one-hot já cobrem R3.1.1;
  - rótulos, métricas de validação e teste, `serving_skew`, SHA-256 do `.onnx` (detecta corrupção; adulteração é coberta pelo digest + atestação, §8), hash dos dados, versões de TF/Keras/opset, `status` (`dummy | candidate | release`).
- **Sem `StringLookup` no grafo.** O vocabulário de `ocean_proximity` vai no manifesto. Exportar string ops ponta a ponta ⚠️ não foi verificado.
- **Golden samples:**
  - tabulares: ≥200 registros **crus** por modelo (teste + novas instâncias, `null` para NaN), com a saída esperada do pipeline completo do notebook, gerada em CPU;
  - CNN: ~50 imagens reais de `seg_test` em PNG 150×150.
- **Quality gate no export:** o apêndice **recusa escrever o artefato** se a métrica ficar abaixo do limiar. O limiar é a validação menos uma margem, definido depois do treino. Um `eval_sample` rotulado (~200 linhas) é re-pontuado no CI. Ele sai do split de teste, logo é checagem de regressão, não estimativa independente.
- **Skew da CNN:**
  - o notebook mede, no `seg_test` inteiro, o caminho de serving (decode e resize com Pillow) contra o caminho Keras;
  - exige top-1 igual em ≥99,5% e |Δacc| ≤ 0,3 pp, gravado em `manifest.serving_skew`;
  - opcional: treinar com a mesma função Pillow, o que dá skew zero.
- **MLflow saiu do notebook** (consenso). As métricas vão para `results.json`. Um `tools/log_to_mlflow.py` opcional importa para o MLflow depois (stretch).
- **Artefato dummy até 10/11** (mesmo schema, `status: "dummy"`), para o pipeline da API e da imagem não esperar o freeze. O CD recusa release com dummy.

---

## 5. Componente 2: API de inferência

### 5.1 Stack (versões verificadas em 09/11/2026)

| Camada | Ferramenta | Versão |
|---|---|---|
| Linguagem / pacotes | Python · **uv** (workspace + lock) | 3.13 · 0.13.7 |
| Web | **FastAPI** · Starlette · Uvicorn | 0.162.2 · 1.10.0 · 0.54.0 |
| Schemas / config | **Pydantic** · pydantic-settings | 2.14.3 · 2.16.2 |
| Inferência | **onnxruntime** (CPU) · numpy · Pillow | 1.31.1 · 2.5.4 · 13.0.0 |
| Fallback CNN | ai-edge-litert (wheel cp313 manylinux x86_64) | 2.3.0 |
| Upload | python-multipart | 0.0.33 |
| Logs | structlog (JSON) | 26.1.0 |
| Métricas | prometheus-client | 0.26.0 |
| Rate limit | slowapi (condicional, ver §5.5) | 0.1.10 |

**Mudanças recentes do FastAPI que o código respeita:**
- `lifespan` em vez de `on_event`;
- checagem estrita de `Content-Type` JSON (0.132.0);
- falha de autenticação retorna 401, não 403 (0.122.0);
- `null` explícito ≠ campo omitido (0.151.0);
- `**extra` removido dos parâmetros → `json_schema_extra` (0.155.0);
- `extra` dos parameter models vale para campos irmãos (0.160.0);
- só Pydantic v2.

**Cortado no debate:** OpenTelemetry. Sem collector nem backend não há quem leia os traces, e `instrumentation-fastapi 0.66b2` é pré-release. `structlog` com `request_id` cobre a correlação.

### 5.2 Arquitetura: hexagonal enxuta, dirigida pelo manifesto

Os 3 modelos têm a mesma forma (tensor entra, tensor sai). Por isso há **uma porta** e o comportamento vem do manifesto, não de herança.

```
api/src/trinet_api/
├── domain/
│   ├── ports.py          # Protocol ModelRuntime: predict(batch) -> ndarray; metadata
│   ├── entities.py       # ModelBundle (manifest + runtime), Prediction (imutáveis)
│   └── errors.py         # ModelNotFound, ModelNotAvailable, InvalidInput, Overloaded
├── application/
│   ├── service.py        # PredictionService: escolhe pré/pós-processamento pelo manifest.task
│   ├── preprocessing.py  # funções puras: to_vector(record, spec) com as 3 primitivas; decode_image()
│   └── postprocessing.py # funções puras: softmax→top-k, regressão→valor
├── infrastructure/
│   ├── runtimes/{onnx.py, numpy_mlp.py, litert.py}   # adaptadores (Strategy)
│   ├── loading.py        # load_bundles(): lê artifacts/, valida manifest (trinet-contract), SHA-256, warmup
│   └── observability.py  # structlog, métricas Prometheus
└── interface/http/
    ├── app.py            # create_app() + lifespan = composition root
    ├── dependencies.py   # Depends + Annotated; override nos testes
    ├── routers.py        # health, models, predictions
    ├── schemas.py        # DTOs Pydantic (≠ entidades)
    ├── problems.py       # handlers RFC 9457 (application/problem+json), ~30 linhas
    └── security.py       # APIKeyHeader + secrets.compare_digest
tests/fakes.py            # FakeRuntime (fora de src/, não vai para a imagem)
```

- **import-linter 2.16** com 1–2 contratos de camada (domain não importa nada de fora; interface não pula a application).
- **Novo modelo = novo artefato com manifesto, sem código novo.** É o exemplo de OCP para a apresentação.

| Padrão | Onde | Motivo |
|---|---|---|
| Ports & Adapters / DIP | `ModelRuntime` | Domínio não conhece ONNX/LiteRT/FastAPI |
| Strategy | ONNX · NumpyMlp · LiteRT · Fake | Escada de fallback e testes sem modelo real |
| Registry dirigido por dados | `load_bundles()` → dict por nome | Extensão por artefato (OCP) |
| Factory | `create_app()` | Composição explícita e testável |
| Dependency Injection | `Depends` + lifespan | Sem singletons globais; `dependency_overrides` nos testes |
| Funções puras (pipeline por composição) | pré e pós-processamento | Substitui o Template Method da v0.1: mais idiomático e fácil de testar |

**SOLID na prática:**
- **S:** service × funções puras × adaptadores × DTOs, cada um com uma responsabilidade.
- **O:** extensão por artefato, sem editar código.
- **L:** todo `ModelRuntime` passa na **mesma suíte de contrato**.
- **I:** a porta tem 2 membros.
- **D:** o service depende do `Protocol`, não das classes concretas.

**Cortado no debate:** `PredictionRecorder` e `ModelRepository` (sem consumidor ou com origem única), use cases por modelo, Template Method, `TFSavedModelRuntime` (poria TF na imagem).

### 5.3 Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health/live` | processo de pé (usado no `HEALTHCHECK`) |
| GET | `/health/ready` | modelos carregados, SHA ok e aquecidos |
| GET | `/v1/models` | lista com versão, métricas e status |
| GET | `/v1/models/{name}` | manifesto público do modelo |
| POST | `/v1/models/{name}/predictions` | tabular: `{"instances":[{...campos crus nomeados...}]}` → `{"predictions":[...]}`, lote com `max_length` · imagem: `multipart/form-data`, 1 arquivo ≤ 5 MB → classe + top-k |

- Respostas trazem `model_name`, `model_version` e `request_id`, além do header `X-Model-Version`.
- Erros em Problem Details, declarados em `responses=` no OpenAPI para o Schemathesis não acusar falso positivo.
- Modelo indisponível → `404 model_not_available`; os outros continuam no ar.
- `/metrics` **não** fica na porta pública: `prometheus_client.start_http_server(9100)` no lifespan, porta não publicada.

### 5.4 Serving, concorrência e escada de fallback
- **1 processo uvicorn** e **1 sessão ORT compartilhada por modelo**. Verificado: `InferenceSession.run` solta o GIL (comentário no pybind do ORT); a thread-safety vem de fontes secundárias.
- Endpoints `def` rodam no threadpool do FastAPI. Um **limitador de concorrência** (2–4 inferências) responde `503` com `Retry-After` quando a espera excede o timeout.
- `intra_op_num_threads` **explícito**: container com `cpus: 2`, começar em `intra_op=1`, `inter_op=1`, `ORT_SEQUENTIAL`, e calibrar com `docker stats`. Se o ORT respeita a cota do cgroup ⚠️ não foi verificado, por isso fixar.
- Escala por **réplicas de container**, não por `--workers` (evita N× memória e o modo multiprocess do Prometheus).
- Limite de corpo **antes** do parse do multipart; `tmpfs: /tmp` (o Starlette grava em disco uploads acima de 1 MB).
- **Imagem:** decode com Pillow → `convert("RGB")` → `resize((150,150), BILINEAR)` se precisar → float32 0–255, com o `Rescaling` dentro do grafo. Validação por magic bytes, `Image.MAX_IMAGE_PIXELS`, formatos permitidos.
- **Escada de fallback** (decisão até 11/11 ao meio-dia; spike com no máximo 2 h e as **3 redes finais**):
  1. ONNX a partir do clone float32, opset 17.
  2. MLPs: `NumpyMlpRuntime` (só Dense + ReLU + softmax/linear; pesos em `.npz` e camadas declaradas no manifesto).
  3. CNN: `LiteRTRuntime` (`format="litert"`). O spike precisa **rodar** o `.tflite` no `Interpreter`, não só converter: o exportador do Keras habilita `SELECT_TF_OPS` e o suporte a ops Flex no runtime leve ⚠️ não foi verificado. O interpretador é tratado como não thread-safe (lock).
  4. Se tudo falhar na CNN: `404 model_not_available` para ela, e os tabulares seguem.

### 5.5 Segurança e requisitos não funcionais
- **API key:** `APIKeyHeader` + `secrets.compare_digest`, segredo via compose `secrets` (arquivo, não env). `/health/*` fica aberto.
- **Rate limit:** slowapi em memória (há 1 processo), com chave pela API key, e teste de 429 no PR. A compatibilidade do 0.1.10 com Starlette 1.10 ⚠️ não foi verificada; se quebrar, limiter próprio.
- Limites de corpo, lote e pixels; timeouts; nenhum stack trace na resposta; security headers. Um pytest garante headers e ausência de stack trace.
- Logs estruturados de predição: `request_id`, modelo/versão, classe, latência e **hash** da entrada, nunca o payload cru.
- **Observabilidade "IA em produção":**
  - RED por modelo, contador de classes previstas e histogramas de 3–4 features, com buckets do manifesto e cardinalidade limitada;
  - Prometheus no profile `observability`; Grafana só se o dashboard provisionado ficar pronto até 17/11;
  - ADR curto explicando PSI/drift.
- **Model card** gerado a partir do manifesto (~1 h, alto valor na apresentação).
- Configuração 12-factor com `pydantic-settings` e `SecretStr`.
- Referências: OWASP Top 10:2025 e OWASP API Security Top 10 (2023, edição vigente encontrada).

---

## 6. Testes automatizados (por gatilho)

| Gatilho | Testes | Ferramentas |
|---|---|---|
| **PR, bloqueante, alvo < 5 min** | Unitário do `PredictionService` com `FakeRuntime` · **suíte de contrato do `ModelRuntime`** (ORT, NumpyMlp, Fake) · Hypothesis nos validadores (NaN/inf, faixas, tamanho de lote; perfil CI com 50 exemplos) e "softmax soma 1" no ORT real · `TestClient` com artefatos reais · **golden/skew**: `to_vector` da API reproduz a matriz X do notebook (atol 1e-6) e as saídas batem (prob. atol 1e-4; top-1 só com margem > ε; imagens PNG estritas; 5 imagens fora de 150×150 com top-1 igual e atol 0,05) · re-score do `eval_sample` · **Schemathesis in-process** (`schemathesis.openapi.from_asgi`, ~50 exemplos por operação) · 429, 401 e headers de segurança · cobertura ≥ 85% global e ≥ 90% em domain/application | pytest 9.2.0 · pytest-cov 7.1.0 · pytest-xdist 3.8.0 · Hypothesis 6.170.15 · Schemathesis 4.33.1 · inline-snapshot 0.36.2 |
| **`main`** | Smoke com **Testcontainers** contra o **digest** publicado: ready, 1 predição por modelo, 401 sem chave | Testcontainers 4.15.0 |
| **Semanal / manual** | Schemathesis longo · teste metamórfico estatístico da CNN (flip horizontal em ~50 imagens reais, concordância mínima) · skew completo do `seg_test` (no export) | — |
| **Cortados** | mutmut, Locust/k6, vulture, Hypothesis stateful, ZAP | — |

---

## 7. Qualidade de código (dono: Arquiteto)
- **Bloqueantes:**
  - **ruff 0.17.4** com `E,F,I,B,UP,S,SIM,RUF,C4,PT,N,ASYNC` + `C90` (complexidade máx. 10). `PL`, `D`, `ANN` e `TRY` saem por ruído;
  - **mypy 2.4.1 `--strict`** + plugin Pydantic (um único type checker; `pyrefly 1.4.1` é stable desde 1.0, e `ty 0.0.100` ainda é beta: fica no editor);
  - **import-linter 2.16**;
  - **deptry 0.25.1**.
- **Consultivo:** SonarQube Cloud (§8).
- **Hooks locais:** `prek 0.5.10` (compatível com `.pre-commit-config.yaml`; fallback `pre-commit 4.7.0`) rodando ruff, mypy, gitleaks e hadolint.
- **Notebook no CI:** só lint estático (ruff/nbQA) num job separado com path filter, sem TF. O "Restart & Run All" é local.
- Conventional Commits. Dependabot com `cooldown` para uv, docker e actions.

---

## 8. CI/CD (GitHub Actions)

### 8.1 Pipeline de PR (`ci.yml`)
Um job agregador **`ci-gate`** (`if: always()`, `needs: [...]`) é o **único required check** do ruleset. Isso evita checks travados em "Expected" por path filters.

| Categoria pedida | Ferramenta (versão) | Modo |
|---|---|---|
| **SAST** | ruff regras `S` (bandit) + **CodeQL** default setup (python + actions; bundle v2.27.4) | bloqueante (CodeQL High+ via ruleset) |
| **SCA / dependency check** | **OSV-Scanner v2.7.1** sobre o `uv.lock` da raiz (API + contrato) | bloqueante |
| | `actions/dependency-review-action` (CVE + licença no diff) | consultivo (⚠️ verificar no dia 1 se o grafo do GitHub lê `uv.lock`; se não ler, sai) |
| | `uv audit` (preview, jun/2026) e `UV_MALWARE_CHECK=1` (preview) | local / informativo |
| **Segredos** | push protection do GitHub + **Gitleaks v8.30.1** (pre-commit e CLI no CI; a `gitleaks-action` exige licença em organizações) | bloqueante |
| **Code smell** | ruff + mypy strict + import-linter + deptry | bloqueante |
| | **SonarQube Cloud** (Free: analisa o branch principal e PRs para `main`; o "Sonar way" não é customizável e falha na 1ª análise) | consultivo, fora do `ci-gate`; excluir `notebook/**`; importar cobertura |
| **Dockerfile** | **Hadolint v2.15.1** + Trivy misconfig | bloqueante |
| **Segurança da imagem** | build com `load: true` (sem push) + **Trivy v0.76.2** `--severity HIGH,CRITICAL --ignore-unfixed --exit-code 1`, DB do espelho `public.ecr.aws/aquasecurity/trivy-db` + cache, `--vex repo` para o VEX da DHI | bloqueante |
| **Segurança do próprio CI** | **zizmor v1.32.0** + todas as actions fixadas por **SHA completo** + `permissions:` mínimas por job | bloqueante |
| Testes | §6 (linha PR) | bloqueante |
| Contrato | drift de `contract/schema.json` (`git diff --exit-code`) + guard de tamanho em `artifacts/` | bloqueante |

### 8.2 Pipeline de entrega (`cd.yml`, em `main`)
1. **Build único**, depois push no **GHCR por digest**. A imagem escaneada é exatamente a publicada.
2. Trivy no digest; **SBOM CycloneDX gerado pelo Trivy** (um formato só; Syft cortado).
3. **`actions/attest`** (v4, embrulha `attest-build-provenance`): proveniência SLSA + SBOM atestados.
   - É uma action de primeira parte, então não há `cosign-installer` de terceiros para fixar, o que pesa depois do caso Trivy.
   - O `provenance`/`sbom` do buildx fica desligado para não duplicar atestações.
   - **cosign foi descartado**: um mecanismo só, e as flags de bundle do v3 mudaram.
4. Smoke com Testcontainers contra o digest.
5. `harden-runner v2.22.4` em modo **audit**, só no job de publish.
6. **Deploy:** `make deploy` na máquina do grupo roda `gh attestation verify oci://ghcr.io/<owner>/trinet-api@sha256:… -R <owner>/trinet` e só então `docker compose up`.
   - Instalar `gh` pelo repositório oficial da GitHub: o apt do Ubuntu oferece 2.45.0, possivelmente antigo para `attestation` ⚠️ não verificado.
   - Sem nuvem e sem Kubernetes.
7. Release recusada se algum manifesto tiver `status: "dummy"`.

### 8.3 Semanal (`security-weekly.yml`, abre issue, não bloqueia)
OSV-Scanner completo (inclusive o lock do notebook), rescan da imagem publicada e das imagens de observabilidade, Schemathesis longo.

### 8.4 Políticas
- **Gates que quebram por causa externa:** um CVE novo publicado de madrugada não pode avermelhar um PR que não mexeu em nada. Daí o `--ignore-unfixed` e o scan completo só no semanal.
- **`SECURITY_GATE=warn` a partir de 16/11** (variável do repo), para o pipeline não travar a entrega.
- **Lição de 2026:** em março, o `aquasecurity/trivy-action` teve 75–76 tags reescritas para commits maliciosos (CVE-2026-33634), com encadeamento em pacotes do PyPI. Consequências no plano: pin por SHA, Dependabot atualizando SHAs, zizmor, preferência por actions de primeira parte.
- **CODEOWNERS** em `artifacts/` e `contract/`.
- **Cortado no debate:** Semgrep (o CE analisa só dentro de função/arquivo; taint entre arquivos é Pro), pip-audit (não lê `uv.lock`), Grype e Docker Scout (segunda opinião sem ação possível), Syft, Scorecard, cosign, ZAP, OTel.

---

## 9. Docker e operação
- **Multi-stage com uv** (doc oficial):
  - `COPY --from=ghcr.io/astral-sh/uv:0.13.7` fixado por digest;
  - `RUN --mount=type=cache` com `uv sync --package trinet-api --locked --no-install-project --no-editable`, depois o projeto;
  - `UV_COMPILE_BYTECODE=1`, `UV_LINK_MODE=copy`, `UV_NO_DEV=1`, `UV_PYTHON_DOWNLOADS=0`;
  - o estágio final copia só o `.venv`, e os artefatos entram na **última camada**.
- **Base:** `ARG BASE_IMAGE=dhi.io/python:3.13`, com a variante `-dev` no estágio de build. São Docker Hardened Images, gratuitas e Apache 2.0 desde dez/2025.
  - `docker login dhi.io` só no job de build.
  - Token read-only duplicado em *Dependabot secrets* (PRs do Dependabot não recebem secrets normais).
  - Digest fixado manualmente.
  - **Fallback documentado:** `--build-arg BASE_IMAGE=python:3.13-slim-trixie@sha256:…`.
  - ⚠️ usuário não-root padrão e matriz de tags da DHI: verificar no dia 1.
- **Container:** não-root, `read_only: true` + `tmpfs: /tmp`, `cap_drop: [ALL]`, `no-new-privileges`, `cpus: 2`, `mem_limit: 1g`, porta publicada em `127.0.0.1`, `HEALTHCHECK` via script Python em `/health/live` com `start_period`, labels OCI, `.dockerignore` estrito (exclui `notebook/`, `*.keras`, `data/`, `mlruns/`).
- **compose:** serviço `api` (padrão) + profile `observability` (Prometheus v3.16.0; Grafana v13.3.0 opcional), com imagens fixadas por digest. Serviço MLflow cortado.

---

## 10. Estrutura do repositório

```
trinet/
├── pyproject.toml          # [tool.uv.workspace] members = ["contract", "api"], exclude = ["notebook"]
├── uv.lock                 # lock de produção (API + contrato)
├── contract/               # trinet-contract: só pydantic; manifest.py + schema.json gerado
├── api/                    # trinet-api (depende de trinet-contract via workspace)
│   ├── src/trinet_api/...
│   └── tests/{unit,contract,property,api,golden,integration}/ + fakes.py
├── notebook/               # projeto uv independente, uv.lock próprio
│   ├── pyproject.toml      # trinet-contract = { path = "../contract" }
│   └── T2_TriNet.ipynb
├── artifacts/<modelo>/<versão>/{model.onnx, manifest.json, golden.json, results.json}
├── deploy/{Dockerfile, compose.yaml, prometheus.yml, grafana/}
├── tools/{gen_model_card.py, log_to_mlflow.py (opcional)}
├── docs/{adr/, model-cards/, runbook.md}
├── .github/{workflows/{ci,cd,security-weekly}.yml, dependabot.yml, CODEOWNERS}
├── .gitattributes          # *.onnx binary
├── .pre-commit-config.yaml · .importlinter · sonar-project.properties · Makefile · README.md
```

- A documentação do uv diz que um workspace tem lock único e não serve para membros com requisitos conflitantes (TF com CUDA × API). Por isso o notebook é um projeto independente ligado por path. Lock separado do projeto excluído ⚠️ confirmar com `uv lock` no dia 1.
- ADRs curtos: ONNX na API e escada de fallback; hexagonal enxuta; contrato e manifesto; base DHI; attest × cosign; drift/PSI.

---

## 11. Divisão do grupo e cronograma

**Divisão (contraproposta do Cientista, aceita):**
- **4 pessoas no notebook:** 2 na CNN, 1 no Dry Bean e 1 no Housing. Ficam **fora** de API/CI, exceto para revisar PRs.
- **2 pessoas de plataforma (API e CI/Docker):** são donas das células de **novas instâncias, golden e export** dos dois tabulares. Isso as obriga a ler a EDA e a preparação, e elas apresentam os MLPs. Os donos da CNN apresentam a CNN.
- Revisão cruzada obrigatória. **Simulado de perguntas em 15/11.**
- Apresentação com o notebook já executado (nada de treino ao vivo), uma "folha de defesa" por rede (escolha, motivo, célula de evidência) e no máximo 1 slide de API/CI.

| Dia | Notebook | API / Plataforma |
|---|---|---|
| **09–10/11** | Corrigir driver da GPU · download · reservar novas instâncias · EDA · **spike de export (≤ 2 h)** · medir época da CNN | Repo, workspace (`contract` + `api`), notebook excluído, **artefato dummy**, CI núcleo (ruff, mypy, pytest, CodeQL, OSV, gitleaks, zizmor, Hadolint, Trivy, `ci-gate`) |
| **11/11** | **Decisão da escada de fallback (meio-dia)** · preparação + baselines | Domínio, service, `FakeRuntime`, suíte de contrato, adaptador ORT |
| **12–13/11** | Primeiras arquiteturas + ablações + justificativas | Endpoints, Problem Details, API key, limitador de concorrência, `to_vector` + testes golden, Schemathesis |
| **14/11** | Ablação da CNN, seeds, avaliação, novas instâncias | Dockerfile DHI endurecido, compose, Trivy da imagem |
| **15/11** | **Freeze apresentável** · export real (substitui o dummy) · `serving_skew` · **simulado** | Integração dos artefatos reais; ajustes de tolerância |
| **16–17/11** | Revisão das justificativas · ensaio | `SECURITY_GATE=warn` · CD (GHCR digest, Trivy, SBOM, `actions/attest`, smoke) · Sonar consultivo · métricas/Prometheus · model cards · ADRs |
| **18/11** | Restart & Run All em ambiente limpo · revisão final | README, runbook, release v1.0.0 |
| **19/11** | **Entrega no Canvas** | — |

---

## 12. Riscos e mitigação

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Driver NVIDIA não carregado | Alta (hoje) | Alto | Corrigir no dia 1; CNN no Colab/Kaggle T4 |
| Export ONNX falhar (tf2onnx sem mantenedor, fora da matriz testada) | Média | Médio | Spike com as redes finais em 11/11; escada NumpyMlp/LiteRT; 404 isolado |
| Skew de pré-processamento (tabular e imagem) | Média | Alto (produção) | Primitivas fechadas no manifesto; golden crus; `serving_skew` medido |
| Escopo grande × prazo | Alta | Alto | 4 pessoas blindadas no notebook; freeze 15/11; cortes do debate; `SECURITY_GATE=warn` em 16/11 |
| Integrante não dominar a parte do outro | Média | Alto (3 pts) | Divisão com plataforma dona de células do notebook; simulado 15/11 |
| Repositório público: cópia por outros grupos / detector de plágio / RA exposto | Média | Alto | **Decisão do usuário (§15)**; capa com RA só na cópia do Canvas |
| Gate de segurança avermelhado por CVE externo | Média | Médio | `--ignore-unfixed`, scan completo semanal, modo warn |
| Supply chain no CI | Baixa | Alto | Pin por SHA, zizmor, actions de primeira parte, harden-runner audit |
| Licença de dataset incerta | Baixa | Baixo | Uso acadêmico, citação, não redistribuir imagens |

---

## 13. Skills selecionadas (via `/find-skills`): 20, em dois níveis

**Núcleo (uso diário):**

| # | Skill | Origem | Domínio |
|---|---|---|---|
| 1 | `anthropic-skills:exploratory-data-analysis` | instalada | EDA dos 3 datasets |
| 2 | `scikit-learn` | instalada | split, métricas, matriz de confusão |
| 3 | `computer-vision-expert` | instalada | CNN |
| 4 | `ml-engineer` | instalada | serving e monitoramento |
| 5 | `fastapi-pro` | instalada | FastAPI + Pydantic v2 |
| 6 | `python-pro` | instalada | Python moderno, uv, ruff |
| 7 | `software-framework:hexagonal` | instalada | ports & adapters |
| 8 | `software-framework:solid` | instalada | SOLID |
| 9 | `backend-design:testing-with-discernment` | instalada | estratégia de testes |
| 10 | `docker-expert` | instalada | imagem endurecida |
| 11 | `software-framework:owasp-security` | instalada | segurança da API |
| 12 | `software-framework:cicd-basics` | instalada | pipeline |

**Apoio (pontual):**

| # | Skill | Origem | Domínio |
|---|---|---|---|
| 13 | `mlops-engineer` | instalada | contrato de artefato, model card |
| 14 | `machine-learning-ops-ml-pipeline` | instalada | fluxo notebook → artefato → API |
| 15 | `software-framework:twelve-factor` | instalada | configuração |
| 16 | `software-framework:observability` | instalada | métricas e logs |
| 17 | `fastapi/fastapi@fastapi` | skills.sh, oficial, 14,9K instalações | padrões atuais do FastAPI |
| 18 | `astral-sh/agent-plugins@uv` | skills.sh, oficial, 5,4K | uv workspace, Docker |
| 19 | `wshobson/agents@python-testing-patterns` | skills.sh, 40K | pytest, Hypothesis |
| 20 | `github/awesome-copilot@github-actions-hardening` | skills.sh, 2,6K | pin por SHA, permissões |

- Opcionais: `pydantic/skills@pydantic` (oficial, 2,6K), `github/awesome-copilot@multi-stage-dockerfile` (31,5K).
- Não há skill de TensorFlow/Keras com reputação suficiente (a mais instalada tem 114 instalações). Para TF/Keras, usar a documentação oficial via Context7.
- Instalação das 4 externas: `npx skills add <owner/repo@skill> -g -y` (aguarda aprovação do usuário).

---

## 14. Debate dos especialistas (Sonnet, 2 rodadas)

**Painel:** Cientista de ML & Rigor Acadêmico · Arquiteto de Software & ML Engineer · DevSecOps & Plataforma. Cada um leu o plano v0.1, verificou fatos com fontes próprias (código do Keras, PyPI, docs do uv/FastAPI/Starlette/GitHub) e criticou só o seu domínio (rodada 1). Na rodada 2, cada um leu as posições dos outros dois, respondeu às perguntas recebidas e fechou as decisões em disputa.

### 14.1 Achados que mudaram o plano

| Achado | Quem levantou | Efeito |
|---|---|---|
| `mixed_float16` impede o export ONNX (o docstring do Keras exige política float32) | Arquiteto + Cientista | Treino em fp32 (TF32 já acelera em Ampere); export de clone float32 |
| tf2onnx "looking for a new maintainer", testado só em TF 2.13–2.15 e opset 14–18; o Keras carrega patch de numpy 2 | Cientista + Arquiteto | Spike com as redes finais e escada de fallback concreta |
| "Sem skew" era falso para o housing (imputação e one-hot fora do grafo) | Arquiteto | Manifesto com 3 primitivas fechadas + golden crus + teste de paridade da matriz X |
| Alvo da regressão sem escala torna o Huber `delta=1` um L1 de fato | Cientista | `Normalization(invert=True)` na saída; delta calibrado; comparação com MSE |
| Teto do alvo é censura, não outlier | Cientista | Métricas também abaixo do teto; baselines |
| Instâncias sintéticas do Dry Bean caem fora da distribuição | Cientista | Linhas reais reservadas antes do split; cenários "e se" no housing |
| `shuffle=True` padrão desalinha a matriz de confusão da Intel | Cientista | `shuffle=False` no teste |
| Hexagonal da v0.1 era cerimônia (4 portas, 4 use cases, Template Method) | Arquiteto | Uma porta + service dirigido por manifesto + funções puras |
| Um único `uv.lock` com TF faria CVEs do notebook bloquearem PRs da API, e o uv não recomenda workspace com requisitos conflitantes | DevSecOps + Arquiteto | Notebook fora do workspace, lock próprio, contrato por path |
| Imagem construída duas vezes: a escaneada não era a publicada | DevSecOps | Build único, tudo por digest |
| Ferramentas sobrepostas (Semgrep CE intra-arquivo, pip-audit sem `uv.lock`, Grype, Syft, cosign + attest) | DevSecOps | Uma ferramenta por categoria, com gate claro |
| Required checks com path filter travam em "Expected" | DevSecOps | Job agregador `ci-gate` |
| SonarQube Free não customiza o gate e falha na 1ª análise | DevSecOps | Sonar consultivo |
| OTel sem collector nem backend é enfeite | DevSecOps + Arquiteto | Cortado; structlog + Prometheus |
| Read-only FS quebra upload multipart > 1 MB | DevSecOps + Arquiteto | `tmpfs: /tmp` |
| Repo público expõe RA e facilita cópia/plágio; privado perde CodeQL/rulesets/attest grátis | DevSecOps + Cientista | **Pergunta ao usuário** |
| 2 pessoas só em API/CI não dominariam a oral | Cientista | Plataforma dona de células do notebook e oradora dos MLPs |

### 14.2 Decisões em disputa e resolução

| Tema | Posições | Resolução |
|---|---|---|
| Contrato de entrada | Arquiteto: registros crus + preprocessing declarativo · Cientista: vetor pós-FE | **Híbrido** (aceito pelos dois): endpoint recebe campos crus nomeados; o tensor do modelo é um vetor descrito no manifesto; a API converte com `to_vector` e 3 primitivas fechadas |
| Plano B do runtime | TF SavedModel (v0.1) · LiteRT · NumPy | **Escada:** ONNX → NumpyMlp (MLPs) → LiteRT (CNN, rodar no Interpreter) → 404 isolado. TF SavedModel rejeitado |
| Workspace | Arquiteto: `trinet-contract` no workspace · DevSecOps: notebook fora | **Ambos:** workspace = contrato + API; notebook independente com path dependency |
| Resize de imagem | no grafo × na API | **Na API, com Pillow**; tolerâncias + `serving_skew` medido no notebook |
| KerasTuner | v0.1: Hyperband | **Cortado** → grade manual nos MLPs + 4 configs na CNN |
| MLflow | v0.1: tracking + registry + compose | **Cortado** do notebook e do compose; `results.json`; script opcional |
| Escopo de CI vs prazo | Cientista: adiar cosign/SLSA/Sonar · usuário: CI robusto agora | **Núcleo bloqueante nos dias 1–2**; supply chain e Sonar só em `main`, em 16–17/11; modo warn a partir de 16/11 |
| Assinatura | cosign × `actions/attest` | **`actions/attest`** (primeira parte, aba Attestations) |
| Imagem base | DHI × slim | **DHI** com `ARG` de fallback para `python:3.13-slim-trixie` |
| slowapi | Arquiteto (R1): cortar · DevSecOps: cortar · Arquiteto (R2): manter | **Mantido condicionalmente** (requisito de segurança), com teste de 429 e fallback próprio |
| ZAP / Locust / mutmut | v0.1: incluídos | **Cortados**: Schemathesis in-process + teste de headers cobrem o essencial |

---

## 15. Pendências e decisões do usuário
1. **Visibilidade do repositório.**
   - Público desde já: CodeQL, rulesets e atestações grátis, mas expõe o trabalho antes da entrega e pode disparar detector de plágio.
   - Privado até 19/11 e público depois: os gates de segurança "premium" só ligam após a entrega.
   - Recomendação: consultar o professor. Se liberado, público com o RA só na cópia do Canvas.
2. **Driver NVIDIA:** carregar o módulo (Secure Boot/MOK ou reboot) antes do treino.
3. **Credenciais locais:** token Kaggle (`~/.kaggle/`) e Docker ID para `dhi.io`.
4. **Instalar as 4 skills externas** (§13).
5. **Nomes e RA** do grupo para a capa.

---

## 16. Fontes
- PyPI JSON API e GitHub Releases API (todas as versões citadas), consultadas em 09/11/2026.
- [What's new in TensorFlow 2.21](https://developers.googleblog.com/whats-new-in-tensorflow-221/) · [TensorFlow: install GPU](https://tensorflow.org/install/gpu) · [TF32 em Ampere](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_tensor_float_32_execution) · [enable_op_determinism](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism)
- [Keras: model export](https://keras.io/api/models/model_saving_apis/export) · [Keras export ONNX (código v3.15.1)](https://github.com/keras-team/keras/blob/v3.15.1/keras/src/export/onnx.py) · [Keras export LiteRT (código v3.15.1)](https://github.com/keras-team/keras/blob/v3.15.1/keras/src/export/litert.py) · [Keras: LiteRT guide](https://keras.io/guides/litert_export/) · [Keras: métricas de classificação](https://keras.io/api/metrics/classification_metrics) · [tf2onnx README](https://github.com/onnx/tensorflow-onnx)
- [ONNX Runtime pybind (release do GIL)](https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/python/onnxruntime_pybind_state.cc) · [CVE-2026-34447 (onnx)](https://advisories.gitlab.com/pypi/onnx/CVE-2026-34447/)
- [FastAPI release notes](https://fastapi.tiangolo.com/release-notes/) (+ releases 0.132.0, 0.150.0, 0.151.0, 0.155.0, 0.160.0, 0.162.0 no GitHub) · [FastAPI: async/threadpool](https://fastapi.tiangolo.com/async/)
- [uv: Docker](https://docs.astral.sh/uv/guides/integration/docker/) · [uv: workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/) · [uv audit e malware check (Astral)](https://astral.sh/blog/uv-audit) · [uv + Dependabot](https://docs.astral.sh/uv/guides/integration/dependabot/)
- [Aqua: Trivy supply chain attack](https://www.aquasec.com/blog/trivy-supply-chain-attack-what-you-need-to-know/) · [Snyk: trivy-action compromise](https://snyk.io/articles/trivy-github-actions-supply-chain-compromise/) · [Socket: Trivy under attack again](https://socket.dev/blog/trivy-under-attack-again-github-actions-compromise) · [Trivy DB rate limit (discussion 7538)](https://github.com/aquasecurity/trivy/discussions/7538)
- [CVE-2025-1550](https://www.sentinelone.com/vulnerability-database/cve-2025-1550/) · [CVE-2025-9905](https://advisories.gitlab.com/pkg/pypi/keras/CVE-2025-9905/) · [CVE-2025-8747](https://cvefeed.io/vuln/detail/CVE-2025-8747) · [GHSA-c9rc-mg46-23w3](https://cvepremium.circl.lu/vuln/ghsa-c9rc-mg46-23w3)
- [Docker Hardened Images grátis (SecurityWeek)](https://www.securityweek.com/docker-makes-1000-hardened-images-free-and-open-source/) · [DHI: use an image](https://docs.docker.com/dhi/how-to/use/) · [DHI quickstart](https://docs.docker.com/dhi/get-started/) · [DHI VEX walkthrough](https://docs.docker.com/guides/dhi-vex-walkthrough/)
- [SonarQube Cloud: planos](https://docs.sonarsource.com/sonarcloud/administering-sonarcloud/managing-subscription/subscription-plans) · [Sonar free tier](https://www.sonarsource.com/blog/the-new-sonarqube-free-tier-is-here/)
- [OSV-Scanner: lockfiles suportados](https://google.github.io/osv-scanner/supported-languages-and-lockfiles/) · [OSV-Scanner v2](https://blog.google/security/announcing-osv-scanner-v2-vulnerability/) · [Semgrep CE vs Pro](https://semgrep.dev/products/semgrep-vs-ce) · [pip-audit](https://github.com/pypa/pip-audit) · [Gitleaks action](https://github.com/marketplace/actions/gitleaks) · [zaproxy/action-api-scan](https://github.com/zaproxy/action-api-scan)
- [actions/attest-build-provenance](https://github.com/actions/attest-build-provenance) · [harden-runner](https://github.com/step-security/harden-runner) · [cosign 3.0](https://blog.sigstore.dev/cosign-3-0-available/)
- [ty beta (pydevtools)](https://pydevtools.com/blog/ty-beta/) · [Pyrefly 1.0](https://github.com/facebook/pyrefly/releases/tag/1.0.0)
- [MLflow Keras 3 (v3.18.0)](https://github.com/mlflow/mlflow/blob/v3.18.0/docs/docs/classic-ml/deep-learning/keras/index.mdx)
- [Dry Bean: UCI](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) · [Dry Bean: duplicatas (JCU)](https://researchonline.jcu.edu.au/86689/7/86689.pdf) · [Dry Bean: benchmarks (arXiv)](https://arxiv.org/pdf/2408.01244) · [Intel Image Classification (HF loader)](https://huggingface.co/datasets/sfarrukhm/intel-image-classification/blob/main/dataset.py) · [hyper.ai: Intel](https://hyper.ai/en/datasets/43968)
- [OWASP Top 10:2025](https://owasp.org/Top10/2025/fa/) · [GitHub: storage e bandwidth do LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-storage-and-bandwidth-usage)
- [skills.sh](https://skills.sh/) (busca via `npx skills find`)
