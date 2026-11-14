# ADR 0004: Notebook fora do workspace uv

- Status: aceito
- Data: 2026-11-10
- Origem: [PLANO](../../PLANO.md) §10 e §14

## Contexto

O workspace do uv tem **um único `uv.lock`** e não serve para membros com requisitos conflitantes. O notebook
precisa de TensorFlow 2.21 e Keras 3.15, Jupyter, pandas e scikit-learn; a API precisa de FastAPI, onnxruntime e
Pillow. Com um lock só, qualquer CVE numa dependência do notebook (pilha do TensorFlow) bloquearia o PR de uma
mudança na API, e a imagem de produção carregaria a árvore de resolução do treino. O debate também registrou que o
`trinet-contract` precisa ser importável pelos dois lados.

## Decisão

- O **workspace** (`pyproject.toml` da raiz) tem só `contract` e `api` como membros e declara
  `exclude = ["notebook"]`. Seu `uv.lock` é o lock de produção.
- O **notebook** é um projeto uv independente (`notebook/pyproject.toml`, `package = false`, `notebook/uv.lock`
  próprio), com `tensorflow==2.21.0`, `keras==3.15.1` e demais versões fixadas.
- A ligação com o contrato é uma dependência por caminho:
  `trinet-contract = { path = "../contract", editable = true }`. O PLANO prevê usá-lo só no apêndice de exportação,
  dentro de `try/except ImportError`, para o notebook continuar autocontido para quem o avalia.
- Tudo que o repositório automatiza ignora o notebook, exceto onde há um motivo:
  - `ruff` (`extend-exclude`) e `mypy` (`exclude`) o ignoram;
  - o CI só faz lint estático do notebook (nbQA + ruff, regras `E9,F`), em job com filtro de caminho, sem instalar
    TensorFlow;
  - o OSV-Scanner do PR olha o `uv.lock` da raiz; o semanal olha também o `notebook/uv.lock`;
  - o Dependabot atualiza o notebook uma vez por mês e **não** atualiza `tensorflow` nem `keras`;
  - o hook de pre-commit `uv lock --check --directory notebook` garante que o lock do notebook acompanhe o
    `pyproject.toml`.
- O "Restart & Run All" do notebook é local; o CI nunca baixa dataset nem treina.

## Consequências

- Dois locks para manter. As bibliotecas que importam para a paridade (numpy, Pillow, onnxruntime) são fixadas nas
  duas pontas e precisam ser mantidas alinhadas à mão.
- CVEs do notebook aparecem na varredura semanal (issue `security-weekly`), não no gate do PR.
- A API e a imagem não enxergam o TensorFlow; o `.dockerignore` mantém `notebook/` fora do contexto de build.
- Quem só quer avaliar o notebook não precisa instalar a API, e quem só quer rodar a API não instala o
  TensorFlow (`make install` sincroniza apenas o workspace).
- O PLANO deixou como pendente confirmar que o `uv lock` funciona com o notebook excluído. O `notebook/uv.lock`
  existe; a checagem `uv lock --check --directory notebook` roda só como hook local, não há job de CI para ela.
