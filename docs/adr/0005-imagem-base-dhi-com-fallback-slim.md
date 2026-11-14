# ADR 0005: Imagem base DHI com fallback slim

- Status: substituído em parte pelo [ADR 0012](0012-fluxos-gratuitos-e-sem-credenciais.md) (o padrão passou a ser `slim`; DHI é opcional e local)
- Data: 2026-11-10
- Origem: [PLANO](../../PLANO.md) §9 e §14

## Contexto

A imagem de inferência é a superfície de ataque exposta. As Docker Hardened Images (DHI, `dhi.io`) são gratuitas e
Apache 2.0 desde dez/2025, mínimas, sem shell nem gerenciador de pacotes no runtime e com usuário não-root. Mas
exigem `docker login dhi.io`, e as credenciais são configuradas à mão pelo dono do repositório (PLANO v1.1): um
fork, um PR do Dependabot ou uma máquina sem login não consegue puxá-las.

## Decisão

- O `deploy/Dockerfile` é multi-stage e recebe `BASE_IMAGE` (runtime) e `BUILD_IMAGE` (estágio de build, variante
  `-dev`). Os padrões são `dhi.io/python:3.13` e `dhi.io/python:3.13-dev`.
- Existe um **estágio nomeado `slim`** com `python:3.13-slim-trixie` fixado por digest. Passar
  `BASE_IMAGE=slim BUILD_IMAGE=slim` (ou `make build BASE=slim`) resolve para ele sem ida ao registro, e o
  Dependabot mantém o digest fresco. O estágio remove `pip`, `setuptools` e `wheel` e aplica as atualizações de
  segurança do Debian publicadas depois do digest.
- O CI e o CD escolhem a base conforme haja credencial: com `DHI_USERNAME` e `DHI_TOKEN`, DHI (com as referências
  das variáveis `DHI_BASE_IMAGE` e `DHI_BUILD_IMAGE`, fixadas por digest); sem elas, `slim`, com um aviso.
- Os digests da DHI **não** podem ser mantidos pelo Dependabot, porque ficam em argumento de build e variável de
  repositório. O comando `make dhi-digests` imprime as referências fixadas, e a atualização é manual.
- Em qualquer base: usuário numérico `65532:65532`, `ENV TRINET_*` sem segredos, `HEALTHCHECK` só com a biblioteca
  padrão (`python -m trinet_api.healthcheck`), só o `.venv` e `artifacts/` entram no estágio final, e os artefatos
  vão na última camada.

## Consequências

- O caminho de produção é a DHI. O `slim` é um recurso para desenvolvimento e CI sem credenciais: ele traz shell e
  apt, e como aplica `apt-get upgrade` o build não é reprodutível byte a byte.
- Duas bases significam dois resultados de scan. Exceções vão em `.trivyignore.yaml`, com validade e justificativa.
  O PLANO previa também `--vex repo` para o VEX da DHI; o CI atual não o usa.
- Os dois estágios (build e runtime) precisam vir da mesma construção de SO/Python, porque o `.venv` guarda o
  caminho do interpretador.
- O PLANO deixou como `⚠️ verificar`: usuário não-root padrão e matriz de tags da DHI. O Dockerfile fixa `USER
  65532:65532` de forma explícita, o que funciona com as duas bases.
- Quem não tem conta DHI ainda roda o projeto inteiro: `make up BASE=slim`.
