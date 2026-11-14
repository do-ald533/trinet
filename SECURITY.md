# Política de segurança

O TriNet é um **projeto educacional** (trabalho de curso, CIC407) que serve três modelos de risco financeiro por uma API
de inferência. Nada nele é recomendação de investimento nem decisão de crédito real. Mesmo assim, o repositório é
público e a API foi construída como se fosse para produção, então vulnerabilidades reais são bem-vindas.

## Como reportar uma vulnerabilidade

**Use o relato privado de vulnerabilidades do GitHub:** na página do repositório, abra a aba **Security** e clique em
**Report a vulnerability**. O relato fica visível só para quem mantém o repositório.

- **Não** abra uma issue pública, um pull request nem um comentário com detalhes da falha.
- Se o botão não estiver disponível (o recurso é uma configuração manual do repositório, listada em
  [`.github/CICD.md`](.github/CICD.md)), abra uma issue pública **sem detalhes técnicos** pedindo um canal privado.
- Inclua: o componente afetado (API, contrato, imagem, workflow), a versão ou o commit, os passos para reproduzir, o
  impacto que você vê e, se tiver, uma sugestão de correção.
- **Nunca** inclua segredos reais (chaves de API, tokens, credenciais) no relato.

Este é um projeto de curso mantido por estudantes: a resposta é **por melhor esforço, sem prazo garantido**. Em
compensação, tratamos o relato com divulgação coordenada: corrigimos primeiro e só então descrevemos a falha
publicamente, com crédito a quem a relatou, se a pessoa quiser.

## Versões suportadas

| Versão | Suporte |
|---|---|
| `main` | recebe correções de segurança |
| última release `1.x` (imagem `ghcr.io/<dono>/trinet-api`, tag `vX.Y.Z` ou `latest`) | recebe correções de segurança, a partir da primeira release |
| releases `1.x` anteriores e imagens `sha-<commit>` antigas | sem suporte: atualize |

Ainda **não existe release `1.x`**: os artefatos de modelo `1.0.0` versionados têm status `candidate` (modelos reais, não
promovidos) e o CD só move `latest` num release manual, depois que o gate de release passa. Até lá, vale apenas a `main`.

## Escopo

Dentro do escopo: o código da API (`api/`), o contrato (`contract/`), o `Dockerfile` e o compose (`deploy/`), os
workflows e scripts (`.github/`), as ferramentas (`tools/`) e o tratamento de artefatos de modelo.

Fora do escopo: vulnerabilidades das imagens base ou das dependências que ainda não têm correção publicada (relate ao
projeto de origem; o repositório as acompanha pelo OSV-Scanner e pelo Trivy), a qualidade preditiva dos modelos,
e o uso dos dados de terceiros (os datasets não são redistribuídos aqui).

## Resumo dos controles de segurança

| Área | Controle |
|---|---|
| Autenticação | cabeçalho `X-API-Key` comparado em tempo constante; **falha fechada**: sem chave configurada, ou com menos de 16 caracteres, a API não sobe |
| Segredo | a chave vem de um arquivo (`/run/secrets/api_key`, que vence a variável `TRINET_API_KEY`), nunca de `.env` nem de argumento de build; a configuração a trata como `SecretStr` e os logs da API não a registram |
| Abuso | rate limit próprio (middleware ASGI sobre a biblioteca `limits`, janela móvel): balde por chave válida e, para as demais requisições, por IP; limite de corpo (1 MiB JSON, 6 MiB upload, antes de qualquer parse), de lote (256), de pixels (12 milhões, checado pelo cabeçalho antes de decodificar) e de concorrência com 503 |
| Entrada | validação estrita por manifesto (campos desconhecidos, não finitos, faixas e vocabulário); imagens validadas pelos **magic bytes**, nunca pelo tipo declarado; só PNG no `candlestick-cnn` |
| Respostas | erros em Problem Details (RFC 9457) sem stack trace nem valor da requisição; cabeçalhos `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Content-Security-Policy`, `Cross-Origin-Resource-Policy`, `Strict-Transport-Security` e `Cache-Control: no-store` |
| Privacidade | os logs guardam só o SHA-256 da entrada, nunca o conteúdo; `/docs` e `/redoc` ficam desligados |
| Modelos | a API nunca desserializa `.keras`, `.h5` ou pickle; `.npz` é lido com `allow_pickle=False`; ONNX com dados externos é recusado por varredura do protobuf, sem importar o pacote `onnx`; SHA-256 de cada arquivo conferido antes de carregar; só a maior versão é servida |
| Contêiner | usuário 65532, sistema de arquivos somente leitura, `/tmp` em tmpfs, `cap_drop: [ALL]`, `no-new-privileges`, limites de CPU, memória e processos, porta só em loopback, métricas em porta interna não publicada, `pip` removido da imagem |
| Supply chain | actions fixadas por SHA completo, `permissions: {}` por padrão, zizmor, Dependabot com *cooldown*, bases fixadas por digest, SBOM CycloneDX e proveniência SLSA atestados com `actions/attest` e verificados no `make deploy` |
| Varreduras | OSV-Scanner, Gitleaks (pre-commit e CI, histórico inteiro), Hadolint, `trivy config`, Trivy na imagem (HIGH e CRITICAL com correção) e uma varredura semanal que abre uma issue; CodeQL e *push protection* dependem de configuração manual no GitHub |

Os detalhes de cada controle estão na [arquitetura](docs/architecture.md), no [runbook](docs/runbook.md) e em
[`.github/CICD.md`](.github/CICD.md). As decisões estão nos ADRs
[0006](docs/adr/0006-actions-attest-em-vez-de-cosign.md), [0009](docs/adr/0009-repositorio-publico-e-credenciais-manuais.md)
e [0011](docs/adr/0011-rate-limit-proprio-em-vez-de-slowapi.md).

## Limitações conhecidas

- **Uma única chave de API**, sem escopos e sem rotação com sobreposição: girar a chave exige reiniciar a API e trocar o
  valor nos clientes ao mesmo tempo ([runbook](docs/runbook.md#segredos)).
- **Sem TLS próprio:** a porta fica em loopback; para acesso remoto, termine o TLS em um proxy reverso (que também deve
  aplicar o próprio limite por cliente, porque o balde por IP da API enxerga o endereço do proxy).
- **Rate limit em memória e por processo:** cada réplica teria o próprio contador.
- **Instância única** e sem autorização por modelo: quem tem a chave acessa todos os modelos.
- **Gates por interruptor:** a variável `SECURITY_GATE=warn` rebaixa scanners bloqueantes a avisos; deve ser usada só
  durante a triagem de um achado ([runbook](docs/runbook.md#interruptor-security_gatewarn)).
- **O repositório é público:** um segredo commitado fica público. A resposta é **girar a credencial**, não reescrever o
  histórico.
