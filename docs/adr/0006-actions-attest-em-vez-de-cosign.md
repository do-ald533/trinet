# ADR 0006: `actions/attest` em vez de cosign

- Status: aceito
- Data: 2026-11-10
- Origem: [PLANO](../../PLANO.md) §8.2 e §14

## Contexto

A imagem precisa de proveniência verificável antes de entrar em produção. Em março de 2026, 75 a 76 tags do
`aquasecurity/trivy-action` foram reescritas para commits maliciosos (CVE-2026-33634). A lição do plano: preferir
actions de primeira parte, fixar tudo por SHA e não somar mecanismos redundantes. Cosign e `actions/attest`
resolveriam o mesmo problema; as flags de bundle do cosign v3 mudaram, e o cosign exigiria um `cosign-installer` de
terceiros para fixar.

## Decisão

- O job `publish` do `cd.yml` constrói **uma única vez**, empurra para o GHCR **por digest**, escaneia o digest com
  o Trivy e gera um SBOM CycloneDX também pelo Trivy (um formato só; o Syft foi cortado).
- Duas atestações com `actions/attest` (fixada por SHA completo), com `push-to-registry: true`: **proveniência de
  build SLSA** e **SBOM CycloneDX**. O `provenance` e o `sbom` do buildx ficam desligados para não duplicar.
- O job tem `id-token: write` e `attestations: write`; o resto dos workflows parte de `permissions: {}`. O
  `harden-runner` roda em modo `audit`, só no job de publicação.
- O próprio CD verifica o que acabou de atestar, com as mesmas flags do `make deploy`:

  ```bash
  gh attestation verify oci://<imagem>@sha256:<digest> -R <dono>/trinet \
    --signer-workflow <dono>/trinet/.github/workflows/cd.yml --deny-self-hosted-runners
  # e, para o SBOM, o mesmo comando com:
  #   --predicate-type https://cyclonedx.org/bom
  ```

- O `make deploy` roda esses dois comandos e só então sobe o compose, com `--pull always` e a imagem sempre
  referenciada por digest.
- `create-storage-record` está `false`: registros de armazenamento só existem para repositórios de organização.
  Vira `true` (com a permissão `artifact-metadata`) se o repositório mudar para uma organização.
- O **cosign foi descartado**: um mecanismo só.

## Consequências

- Nenhuma ação de terceiros extra para atestar; a aba *Attestations* do GitHub mostra o resultado.
- A verificação depende do GitHub (API de atestações) e do `gh`, que precisa ser recente: o subcomando
  `gh attestation` não existe em versões antigas, e o PLANO nota que o `apt` do Ubuntu oferecia o `gh` 2.45.0.
  Instale pelo repositório oficial da GitHub.
- A confiança está em "esta imagem foi construída por `cd.yml` deste repositório numa runner hospedada pelo
  GitHub", não em uma chave que a equipe guarde. Quem controlar o workflow controla o que é atestado: por isso
  `.github/` tem `CODEOWNERS` (com a revisão de Code Owners exigida no ruleset, passo manual do `CICD.md`) e as
  actions são fixadas por SHA.
- A tag `latest` e a de versão só se movem por `docker buildx imagetools create --prefer-index=false` no mesmo
  digest, para não invalidar as atestações.
