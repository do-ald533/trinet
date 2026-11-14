# ADR 0012: Fluxos gratuitos e sem credenciais

- Status: aceito (substitui a parte de DHI do [ADR 0005](0005-imagem-base-dhi-com-fallback-slim.md) e a parte de
  `DHI_*` e `SONAR_TOKEN` do [ADR 0009](0009-repositorio-publico-e-credenciais-manuais.md))
- Data: 2026-11-14
- Origem: pedido do dono do repositório ("todos os nossos fluxos têm que ser gratuitos"); [PLANO](../../PLANO.md) §8 e §9

## Contexto

O PLANO v1.1 usava dois serviços que custam zero, mas dependem de conta e de credencial externa: as Docker Hardened
Images (`dhi.io`, com `DHI_USERNAME` e `DHI_TOKEN`) e o SonarQube Cloud (`SONAR_TOKEN`, plano gratuito só para
repositório público). "Gratuito hoje" não é uma propriedade do projeto: o plano de um terceiro muda, e um fork, um PR
do Dependabot ou uma máquina sem login cai no caminho de erro.

## Decisão

- **Regra.** Todo fluxo (CI, CD, varredura semanal, `make`, deploy local) roda de graça, sem conta, token ou plano
  além do `GITHUB_TOKEN` embutido do próprio GitHub Actions, num repositório público.
- **Base da imagem.** O padrão de tudo (Dockerfile, compose, Makefile, CI e CD) é o estágio `slim`: o
  `python:3.13-slim-trixie` oficial fixado por digest, mantido pelo Dependabot. As DHI continuam possíveis só como
  opção local (`make build BASE=dhi`, exige `docker login dhi.io`) e nenhum workflow as usa.
- **Sonar removido.** O job `sonar`, o `sonar-project.properties` e a exceção do Gitleaks para ele saíram. A análise
  estática de segurança fica com o CodeQL (configuração padrão do GitHub, gratuita em repositório público), mais
  `ruff`, `mypy --strict`, `import-linter` e `deptry`.
- **Trava automática.** `tools/check_free_flows.py` (`make free-flows`, parte da matriz `static` do `ci-gate`) falha se
  um workflow ler um segredo que não seja `GITHUB_TOKEN`, uma variável fora de `SECURITY_GATE`, usar outro registro
  além do `ghcr.io`, outro runner além do Ubuntu padrão do GitHub, citar um serviço que exige conta ou plano (Sonar,
  Codecov, Snyk, `dhi.io` e afins) ou se o compose voltar a depender de login. Os testes estão em
  `api/tests/unit/test_free_flows.py`.
- **O que continua e por que é gratuito.** Runners `ubuntu-24.04`, GHCR, `actions/attest`, CodeQL, secret scanning com
  push protection, Dependabot e rulesets são gratuitos em repositório público (ADR 0009). Trivy, hadolint,
  Gitleaks (CLI), zizmor, actionlint, OSV-Scanner, Schemathesis e Testcontainers são software livre; os CLIs vêm de
  releases fixadas com checksum (`.github/scripts/install-cli.sh`).

## Consequências

- Não sobram segredos para configurar: some o passo manual de `DHI_USERNAME`, `DHI_TOKEN`, `SONAR_TOKEN`,
  `DHI_BASE_IMAGE` e `DHI_BUILD_IMAGE`. Sobra só a variável opcional `SECURITY_GATE`.
- A imagem perde o ganho das DHI: a base `slim` traz shell e `apt`, e o `pip` é removido no Dockerfile. O
  usuário 65532, o sistema de arquivos somente leitura, `cap_drop: [ALL]`, o Trivy bloqueante e a atestação continuam.
  Imagem sem shell fica como evolução opcional, não como requisito.
- O digest da base agora é atualizado pelo Dependabot (literal `FROM`), sem o passo manual `make dhi-digests`.
- Sem o painel do Sonar, não há métrica de code smell fora do CodeQL, do `ruff` e das regras de arquitetura.
- Dois riscos conhecidos, não verificados em runner real: o limite de pulls anônimos do Docker Hub para o
  `python:3.13-slim-trixie` e a política de minutos do GitHub Actions, que vale para repositório público; se o
  repositório virar privado, runners, atestações e varredura de segredos passam a depender do plano da conta.
- O `step-security/harden-runner` (modo `audit`, camada comunitária, sem conta) é o único serviço de terceiros que
  resta no CD. Ele não exige credencial; se o plano gratuito mudar, remova o passo, que é só observação.
- A regra vale para o futuro: um workflow novo que quebre a trava só passa se o ADR e a lista de permissões do
  `check_free_flows.py` mudarem no mesmo PR.
