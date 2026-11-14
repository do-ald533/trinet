# ADR 0011: Rate limit próprio em vez de slowapi

- Status: aceito (substitui a decisão "slowapi, condicional" do PLANO §5.5 e §14.2)
- Data: 2026-11-14
- Origem: [PLANO](../../PLANO.md) §5.1, §5.5 e §14.2

## Contexto

O PLANO pedia um rate limit em memória (a API roda em um único processo), com chave pela API key, e manteve o
`slowapi` 0.1.10 **condicionalmente**: a compatibilidade com o Starlette 1.10 não estava verificada, e a regra era "se
quebrar, limiter próprio", com um teste de 429 no PR para provar que o limite funciona.

Na integração com o FastAPI 0.162 (travado em 0.162.4 no `uv.lock`), o `slowapi` 0.1.10 ficou **desativado em silêncio**
pelo `include_router`: nenhum erro, nenhum aviso, e o limite não era aplicado. Reproduzido em 14/11/2026 com um app
mínimo (FastAPI 0.162.2, slowapi 0.1.10, `SlowAPIMiddleware` com `default_limits=["2/minute"]`): uma rota registrada
direto no `app` devolveu `200, 200, 429, 429`, e a mesma rota declarada num `APIRouter` incluído com `include_router`
devolveu `200, 200, 200, 200`. Um limitador que parece ligado e não limita é pior que nenhum, porque é um controle de
segurança que ninguém percebe ter falhado.

## Decisão

- O limite é um **middleware ASGI puro**, `RateLimitMiddleware`, em `api/src/trinet_api/interface/http/middleware.py`,
  sobre a biblioteca `limits` (`MovingWindowRateLimiter` com `MemoryStorage`). O `slowapi` saiu das dependências, do
  `import-linter` e dos overrides do mypy; `limits>=3.13.0` ficou como dependência direta.
- A ordem dos middlewares, de fora para dentro, é `RequestContextMiddleware` > `RateLimitMiddleware` >
  `BodyLimitMiddleware` > rotas. Como é uma camada ASGI e não depende do roteador, o limite vale para **toda** rota,
  inclusive 404 e 405, e não pode ser desligado por uma mudança de roteamento.
- **Baldes.** Requisição com a chave válida usa um balde `key:<digest>` (início do SHA-256 da chave). Qualquer outra
  (chave ausente ou errada) usa um balde `ip:<endereço do cliente>`. Assim uma enxurrada de chaves ruins não consome
  o orçamento de quem tem a chave certa, e a memória fica limitada ao número de endereços. `/health/live` e
  `/health/ready` são isentas.
- Ao estourar, a resposta é `429` em `application/problem+json` com `code: "rate_limited"` e o cabeçalho `Retry-After`
  (segundos até a janela liberar). O estouro vira um log `rate_limited` com o tipo de balde, nunca com a chave.
- A configuração é `TRINET_RATE_LIMIT` (padrão `60/minute`), validada na inicialização com `limits.parse_many`, que
  aceita vários limites (por exemplo `10/second;60/minute`); um valor inválido impede o serviço de subir.
- O teste de 429 continua: `api/tests/unit/test_rate_limit.py` exercita o middleware como componente ASGI, sem
  FastAPI, e a suíte `api/tests/api/` confirma o 429 na aplicação inteira.

## Consequências

- O controle não depende de detalhes internos do FastAPI ou do Starlette, que foi o que quebrou o `slowapi`, e o seu
  teste não monta uma aplicação.
- O projeto passa a manter cerca de 50 linhas de código de segurança, em troca de uma dependência a menos. A biblioteca
  `limits` continua sendo a dona da aritmética da janela.
- O limite é **por processo e em memória**: uma segunda réplica teria o próprio contador, e um reinício o zera. O
  PLANO já fixava uma única instância; passar disso exigiria um armazenamento compartilhado (a `limits` suporta).
- O endereço usado é o do cliente TCP visto pelo uvicorn. Atrás de um proxy reverso é o do proxy (o projeto não
  configura os cabeçalhos de proxy do uvicorn), então todos os clientes sem chave válida dividiriam um balde. O proxy
  deve aplicar o próprio limite por cliente.
- O limite roda **antes** da autenticação: requisições sem chave também gastam o balde do próprio IP, o que é
  desejado para frear tentativas de adivinhar a chave.
- Trocar de implementação de novo é trocar o middleware: a variável `TRINET_RATE_LIMIT`, o código de erro e os testes
  de comportamento permanecem.
