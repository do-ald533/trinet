# ADR 0003: Contrato de artefato (manifest, golden e três primitivas)

- Status: aceito
- Data: 2026-11-10
- Origem: [PLANO](../../PLANO.md) §4.5, §5.2 e §14

## Contexto

O notebook (produtor) e a API (consumidora) evoluem separados. Sem um contrato explícito, o *training-serving skew*
aparece em silêncio: imputação, one-hot e redimensionamento de imagem feitos de um jeito no notebook e de outro na
API. O debate achou falso o argumento de "sem skew" para os modelos tabulares (no plano original, o de regressão
imobiliária; hoje `credit-default-mlp` e `volatility-mlp`), porque imputação, one-hot e razões ficam fora do grafo.
Os dois lados precisam implementar exatamente a mesma coisa, e isso só é verificável se o conjunto de operações for
pequeno e fechado.

## Decisão

1. O pacote `trinet-contract` (só Pydantic) define o `Manifest` e o `Golden`. Os modelos são imutáveis e usam
   `extra="forbid"`; `schema_version` é a constante `"1.0"`. Os JSON Schemas ficam em `contract/schema/` e o CI
   regenera e falha se houver diferença (`make schema-check`).
2. `artifacts/<modelo>/<versão>/` contém `manifest.json`, os arquivos de runtime que ele cita, `golden.json` (e
   `golden/*.png` para imagens) e `results.json` (extras, não validado pela API). O nome do modelo é um slug e a
   versão é semver; nomes de arquivo são simples, sem separador nem travessia de diretório.
3. O manifesto traz: `task` (`classification`, `regression`, `image_classification`), `input`, `output`, tensores de
   entrada e saída, `runtimes` em ordem de preferência (cada um com SHA-256), `metrics` por split, `quality_gate`,
   `serving_skew`, `training` (dataset, SHA-256 dos dados, seed, versões de Python, TensorFlow e Keras, hardware),
   `golden` (arquivo e SHA-256), `status` (`dummy`, `candidate`, `release`) e `git_sha`.
4. **Três primitivas fechadas** para o que acontece fora do grafo, só com constantes do treino:
   - **imputação por constante** (`impute` do campo, usada para `null` ou ausência);
   - **one-hot por vocabulário** (`one_hot`, na ordem do vocabulário);
   - **razão A/B** (`ratio`, numerador e denominador numéricos).

   `passthrough` só seleciona o valor de um campo numérico como coluna. Padronização (`Normalization`) e
   reescalonamento de imagem ficam **dentro do grafo**, assim como o `exp` que devolve a volatilidade à unidade
   natural ([ADR 0010](0010-pivo-para-o-mercado-financeiro.md)); imagens têm decodificação, `convert("RGB")` e resize bilinear
   com Pillow na API, descritos por `ImageInput`. Cada campo numérico pode ter `minimum` e `maximum`, validados na
   entrada.
5. O **golden** guarda entradas cruas e a saída esperada do pipeline completo do notebook: para tabulares, `record`
   → `features` → `output` (com `label` nos casos usados como amostra rotulada); para imagens, PNG → `output`. Há
   tolerâncias (`features_atol`, `output_atol`, `top1_margin`). A suíte `api/tests/golden/` reproduz, para cada
   caso, a matriz de features, a saída de cada runtime da escada e a resposta HTTP, e refaz o quality gate.
6. O **quality gate** é gravado no manifesto e o notebook se recusa a escrever o artefato se falhar; um manifesto
   `release` com gate reprovado é inválido. `serving_skew` registra a concordância Keras x runtime. O CD recusa
   release com artefato `dummy` ou versão abaixo de 1.0.0 (`tools/check_artifacts.py --release`).
7. `CODEOWNERS` cobre `artifacts/` e `contract/`.

## Consequências

- Manifesto inválido, SHA-256 divergente ou campo desconhecido tornam o modelo indisponível, nunca meio carregado.
- O conjunto fechado limita a engenharia de atributos **feita pela API** às três primitivas. As razões em uso são
  `bill_amt1 / limit_bal` (`credit-default-mlp`) e `rv_5 / rv_20` (`volatility-mlp`); todo divisor tem de ter
  `minimum > 0` no manifesto, senão a API devolve 422 `zero_denominator`. As features de séries temporais não cabem
  nas primitivas e são do chamador (ADR 0010).
- Acrescentar uma primitiva, um tipo de entrada ou um formato de imagem é uma mudança de contrato (`schema_version`),
  do notebook e da API ao mesmo tempo.
- O contrato 1.0 **não** carrega distribuições de referência por classe ou por feature; isso limita o que a API
  consegue monitorar ([ADR 0007](0007-monitoramento-de-drift-psi.md)).
- Os JSON Schemas ficam em `contract/schema/manifest.schema.json` e `golden.schema.json`. O PLANO §8.1 e §10 fala
  em um `contract/schema.json` único, mas o cabeçalho da v1.1 já cita a pasta `contract/schema/`.
- Os golden de imagem são sintéticos (passeios aleatórios geométricos renderizados em gráfico), porque o CSV do S&P
  500 tem licença não verificada e não é redistribuído; a acurácia em gráficos reais vem do `serving_skew` medido no
  notebook.
- A v1.2 acrescentou ao contrato a camada `exp` do runtime NumPy (`ExpLayer`, `op: "exp"`) **sem subir o
  `schema_version`**, que continua `"1.0"`. Isso contraria a regra acima (mudança de contrato sobe a versão) e só é
  aceitável porque nenhuma release 1.0.0 existia: a próxima mudança desse tipo, depois da primeira release, sobe a
  versão.
