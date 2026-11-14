## What and why

<!-- One or two sentences. Link the issue or the PLANO section. -->

## Checklist

- [ ] `make lint typecheck imports deptry cov` pass locally (or the `ci-gate` check is green)
- [ ] Contract changed? `make schema` run and `contract/schema/*.json` committed
- [ ] Artifacts changed? `make check-artifacts` passes (manifest + SHA-256 + golden + size guard)
- [ ] No secrets, tokens, datasets or `.keras`/`.h5` files added
- [ ] Workflow or Dockerfile changed? Actions pinned by full SHA with a `# vX.Y.Z` comment
