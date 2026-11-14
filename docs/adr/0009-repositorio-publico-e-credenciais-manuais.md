# ADR 0009: Repositório público, sem identificação dos alunos e credenciais manuais

- Status: aceito (altera a v1.0 do PLANO)
- Data: 2026-11-10
- Origem: [PLANO](../../PLANO.md), cabeçalho v1.1 e §15

## Contexto

A v1.0 deixou aberta a visibilidade do repositório. Público libera de graça o CodeQL, os rulesets, o push
protection e as atestações do `actions/attest`, mas expõe o trabalho antes da entrega e pode disparar detector de
plágio. Privado até 19/11 adia os gates de segurança "premium". Havia ainda o risco de expor nomes e RA.

## Decisão

- O repositório é **público desde já**.
- **Nomes e RA ficam fora do repositório**: só aparecem na cópia do notebook entregue no Canvas. Os arquivos do
  repositório usam "TriNet contributors" como titular da licença MIT.
- **As credenciais são configuradas manualmente pelo usuário**: token do Kaggle (`~/.kaggle/`, só na máquina local; hoje dispensável, porque o download do S&P 500 via
  `kagglehub` é anônimo),
  e as variáveis e regras do GitHub (o login em `dhi.io` e o `SONAR_TOKEN` saíram, [ADR 0012](0012-fluxos-gratuitos-e-sem-credenciais.md)). Nada disso entra no repositório; o `.gitignore` bloqueia `kaggle.json`, `.env*` (exceto
  `.env.example`), `secrets/`, `*.pem` e `*.key`, e o Gitleaks roda no pre-commit e no CI sobre o histórico inteiro.
- Os datasets não são redistribuídos e o CI nunca os baixa. O CSV do S&P 500 (Kaggle `camnugent/sandp500`) tem licença
  e origem **não verificadas**, e o *Default of Credit Card Clients* (UCI #350, CC BY 4.0) é baixado por quem roda o
  notebook. Por isso as imagens golden são sintéticas (gráficos de passeios aleatórios, nunca derivados do CSV).
  Desde a v1.2 os dois downloads são anônimos, então o projeto não precisa mais de credencial do Kaggle
  ([ADR 0010](0010-pivo-para-o-mercado-financeiro.md)).
- Os passos do GitHub que o repositório não consegue automatizar (rulesets, CodeQL, push protection, variáveis e
  segredos) estão listados em [`.github/CICD.md`](../../.github/CICD.md).

## Consequências

- Quem clona o repositório vê código, contratos, artefatos dummy ou exportados e a documentação, mas não os dados
  nem as credenciais.
- Segurança e visibilidade andam juntas: um commit com segredo fica público. A proteção é preventiva (push
  protection, hooks, Gitleaks no histórico) e a resposta é rotacionar a credencial, não reescrever o histórico
  ([runbook](../runbook.md#segredos)).
- A política de reporte de vulnerabilidades usa o *private vulnerability reporting* do GitHub
  ([`SECURITY.md`](../../SECURITY.md)); isso também é uma configuração manual.
- A marcação do handle em `.github/CODEOWNERS` é um marcador a substituir pelo dono (assinalado com `TODO(user)`).
