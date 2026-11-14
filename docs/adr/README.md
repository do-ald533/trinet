# Decisões arquiteturais (ADR)

Cada ADR registra uma decisão, o contexto que a motivou e o que ela custa. O formato é curto: *Status*, *Data*,
*Origem* (a seção do [PLANO](../../PLANO.md) que a originou), *Contexto*, *Decisão* e *Consequências*. Uma decisão
nova que contradiz um ADR aceito ganha um ADR novo; o antigo só é editado para corrigir fatos ou acompanhar uma
mudança de domínio (como as referências a modelos, trocadas depois do ADR 0010).

| ADR | Decisão | Status |
|---|---|---|
| [0001](0001-onnx-na-api-e-escada-de-fallback.md) | ONNX Runtime na API e escada de fallback (ONNX, NumPy MLP, LiteRT) | aceito |
| [0002](0002-hexagonal-enxuta-dirigida-pelo-manifesto.md) | Hexagonal enxuta, com uma porta e comportamento dirigido pelo manifesto | aceito |
| [0003](0003-contrato-de-artefato.md) | Contrato de artefato: manifest, golden e três primitivas de pré-processamento | aceito |
| [0004](0004-notebook-fora-do-workspace-uv.md) | Notebook fora do workspace uv, com lock próprio | aceito |
| [0005](0005-imagem-base-dhi-com-fallback-slim.md) | Imagem base DHI, com fallback `python:3.13-slim` fixado por digest | substituído em parte pelo 0012 |
| [0006](0006-actions-attest-em-vez-de-cosign.md) | `actions/attest` (proveniência e SBOM) em vez de cosign | aceito |
| [0007](0007-monitoramento-de-drift-psi.md) | Monitoramento de drift com PSI sobre as métricas realmente expostas | aceito, implementação parcial |
| [0008](0008-treino-em-cpu.md) | Treino em CPU (altera a v1.0 do PLANO) | aceito |
| [0009](0009-repositorio-publico-e-credenciais-manuais.md) | Repositório público, sem nomes e RA, credenciais manuais | aceito |
| [0010](0010-pivo-para-o-mercado-financeiro.md) | Pivô para o mercado financeiro: três modelos novos, camada `exp`, features do chamador, divisão temporal com purge | aceito |
| [0011](0011-rate-limit-proprio-em-vez-de-slowapi.md) | Rate limit próprio (middleware ASGI sobre `limits`) em vez de `slowapi` | aceito |
| [0012](0012-fluxos-gratuitos-e-sem-credenciais.md) | Fluxos gratuitos e sem credenciais: base `slim`, sem Sonar, trava `make free-flows` | aceito |

## Relações entre os ADRs

- O [0010](0010-pivo-para-o-mercado-financeiro.md) troca os datasets e os nomes dos modelos citados na v1.0 do PLANO;
  os ADRs 0001 a 0009 foram escritos antes dele e, onde falam de um modelo, valem para o equivalente atual
  (`credit-default-mlp`, `volatility-mlp`, `candlestick-cnn`).
- O [0011](0011-rate-limit-proprio-em-vez-de-slowapi.md) substitui a menção ao `slowapi` do PLANO §5.5.
- O [0012](0012-fluxos-gratuitos-e-sem-credenciais.md) tira DHI e SonarQube do caminho padrão e do CI/CD: onde o PLANO, o 0005 e o 0009 falam de `DHI_*` ou `SONAR_TOKEN`, vale o 0012.
- Divergências conhecidas entre o PLANO e a implementação estão nas seções "Consequências" de cada ADR.
