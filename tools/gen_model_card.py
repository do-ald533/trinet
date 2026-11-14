"""Gera os model cards (pt-BR) a partir do artefato mais recente de cada modelo.

Le ``artifacts/<modelo>/<versao>/manifest.json`` (validado por ``trinet_contract.load_manifest``)
e, se existir, ``results.json`` e escreve:

* ``docs/model-cards/<modelo>.md``: o card do modelo;
* ``docs/model-cards/README.md``: indice com a tabela de resultados;
* o bloco entre ``<!-- results:start -->`` e ``<!-- results:end -->`` do ``README.md`` da raiz
  (ignorado, com aviso, se os marcadores nao existirem).

A saida e deterministica (nenhuma data de execucao entra nos arquivos), entao rodar de novo nao
gera diff. Artefatos ``dummy`` sao marcados como tal no topo do card, e o README so mostra
numeros de artefatos reais.

Uso::

    uv run --package trinet-contract python tools/gen_model_card.py            # escreve
    uv run --package trinet-contract python tools/gen_model_card.py --check    # exit 1 se mudaria

So usa a biblioteca padrao e ``trinet-contract``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trinet_contract import (
    CategoricalField,
    ClassificationOutput,
    DenseLayer,
    ExpLayer,
    ImageGolden,
    ImageInput,
    Manifest,
    NumericField,
    NumpyMlpRuntimeSpec,
    OneHotFeature,
    OnnxRuntimeSpec,
    PassthroughFeature,
    Status,
    TabularInput,
    Task,
    load_golden,
    load_manifest,
)
from trinet_contract.manifest import MlpLayer, RuntimeSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_OUT = REPO_ROOT / "docs" / "model-cards"
DEFAULT_README = REPO_ROOT / "README.md"

README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
REGENERATE = "uv run --package trinet-contract python tools/gen_model_card.py"
ADR_FALLBACK = "../adr/0001-onnx-na-api-e-escada-de-fallback.md"
ADR_DRIFT = "../adr/0007-monitoramento-de-drift-psi.md"
ADR_PIVOT = "../adr/0010-pivo-para-o-mercado-financeiro.md"
MAX_ROWS = 40
MAX_JSON_LINES = 80

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")
_RESULTS_KNOWN = frozenset(
    {"model", "version", "status", "metrics", "quality_gate", "serving_skew"}
)

TASK_PT: dict[Task, str] = {
    Task.CLASSIFICATION: "classificação tabular",
    Task.REGRESSION: "regressão tabular",
    Task.IMAGE_CLASSIFICATION: "classificação de imagens",
}


# --------------------------------------------------------------------------- textos por modelo


@dataclass(frozen=True, slots=True)
class Notes:
    """Texto curado (pt-BR) que o manifesto nao carrega: uso, dados, limites e etica."""

    disclaimer: str
    intended_use: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    data: tuple[str, ...]
    caller: tuple[str, ...]
    limitations: tuple[str, ...]
    ethics: tuple[str, ...]


EDU_MARKET = (
    "**Uso educacional.** Trabalho de curso (CIC407, T2). Nada neste modelo ou na API é "
    "recomendação de investimento, sinal de compra ou venda, nem base para decisões "
    "financeiras reais."
)
EDU_CREDIT = (
    "**Uso educacional.** Trabalho de curso (CIC407, T2). Este modelo **não é** um modelo de "
    "decisão de crédito real e nada aqui é recomendação financeira: os dados são de Taiwan, "
    "2005, e o limiar de decisão usa custos ilustrativos."
)

SP500_DATA = (
    "S&P 500, 5 anos (Kaggle `camnugent/sandp500`): 619.040 linhas e 7 colunas (`date`, "
    "`open`, `high`, `low`, `close`, `volume`, `Name`), 505 tickers, de 2013-02-08 a "
    "2018-02-07 (1.259 pregões). Preços ajustados a splits.",
    "A licença e a origem do arquivo **não foram verificadas**: o CSV nunca é versionado nem "
    "redistribuído (o download via `kagglehub` é anônimo, feito por quem roda o notebook).",
    "Limpeza (PLANO v1.2, V3): remove linhas com NaN em OHLC, volume 0, `high < low` e `close` "
    "fora de [`low`, `high`]. Uma amostra em `t` exige a janela [t-59, t+5] contígua e nenhum "
    "|retorno log| > 0,4 nela (descarta artefatos de ações corporativas e 3 eventos reais).",
    "Divisão cronológica por data (todos os tickers juntos): 70% / 15% / 15% dos pregões, com "
    "embargo (purge) de 5 pregões em cada fronteira, porque o alvo olha 5 dias à frente. "
    "O teste é tocado uma vez.",
)

SP500_BIASES = (
    "Viés de sobrevivência: só entram as ações que eram constituintes do índice em fev/2018. "
    "Empresas que saíram, faliram ou foram compradas antes estão ausentes.",
    "Período curto: cinco anos de um único regime de mercado, com a alta de volatilidade de "
    "jan-fev/2018 **dentro do teste**. Ela é um evento fora do que o treino viu, então o "
    "erro nesse trecho é esperado e deve ser lido separado (o notebook mostra o erro por "
    "sub-período).",
    "O preço de fechamento ajustado vem de uma fonte não auditada; 14 retornos com |log| > 0,4 "
    "indicam ajustes de desdobramento e cisão incompletos (alguns eventos são reais).",
)

NOTES_BY_NAME: dict[str, Notes] = {
    "credit-default-mlp": Notes(
        disclaimer=EDU_CREDIT,
        intended_use=(
            "Estimar a probabilidade de inadimplência no mês seguinte de um cliente de cartão "
            "de crédito a partir do limite, da escolaridade, da idade, do histórico de atraso "
            "(`pay_1` a `pay_6`) e dos valores de fatura e de pagamento dos últimos 6 meses.",
            "Demonstração didática da disciplina CIC407 (T2) e exemplo de serviço de inferência.",
        ),
        out_of_scope=(
            "Concessão, limite, preço ou cobrança de crédito reais.",
            "Qualquer população fora do conjunto de treino: clientes de cartão em Taiwan, 2005.",
            "Uso de sexo ou estado civil como insumo: eles não são entradas do modelo nem da API.",
        ),
        data=(
            "Default of Credit Card Clients (UCI Machine Learning Repository #350): 30.000 "
            "clientes, 23 atributos e o alvo `Y` (inadimplência no mês seguinte). Sem NaN; "
            "inadimplência de 22,12% (6.636 clientes).",
            "Licença CC BY 4.0, DOI 10.24432/C55S3H. Citação: Yeh, I. (2009). Default of "
            "Credit Card Clients [Dataset]. UCI Machine Learning Repository.",
            "Problemas verificados em 14/11/2026: 35 linhas duplicadas (sem o ID), removidas "
            "antes do split; `EDUCATION` com os códigos 0, 5 e 6 não documentados (14, 280 e "
            "51 linhas); `MARRIAGE = 0` (54 linhas); `PAY_*` com os valores -2 e 0 sem "
            "documentação; 590 `BILL_AMT1` negativos.",
            "Divisão 70% / 15% / 15% estratificada e sem duplicatas. O conjunto é baixado da "
            "UCI por quem roda o notebook e não é versionado.",
        ),
        caller=(
            "`education` é categórica na API: mapeie os códigos da UCI (1 = `graduate`, "
            "2 = `university`, 3 = `high_school`; 0, 4, 5 e 6 = `other`) antes de chamar.",
            "`limit_bal` precisa ser >= 1: a utilização `bill_amt1 / limit_bal` não aceita "
            "divisor zero, e a API responde 422 (`below_minimum`) a valores menores.",
        ),
        limitations=(
            "Classes desbalanceadas (22,12% de inadimplentes): a acurácia sozinha engana. O "
            "notebook reporta ROC-AUC, PR-AUC, KS, Brier e calibração, contra baselines "
            "(classe majoritária e regressão logística).",
            "O limiar de decisão foi escolhido na validação por custo (um falso negativo custa "
            "5 falsos positivos). A premissa é ilustrativa, não calibrada com dados de um "
            "credor, e o limiar de 0,5 é reportado para comparação.",
            "Dados de um único mercado e de 2005: o comportamento de crédito, os produtos e o "
            "contexto regulatório mudaram.",
            "Códigos sem documentação (`EDUCATION` 0, 5, 6; `PAY_*` -2 e 0) foram mantidos ou "
            "agrupados como descrito no notebook, sem confirmação do significado original.",
        ),
        ethics=(
            "**Atributos protegidos.** `SEX` e `MARRIAGE` ficam fora das features (princípio de "
            "não discriminação em crédito: a Regulation B / ECOA, nos EUA, trata sexo e estado "
            "civil como insumos vedados). Excluí-los não elimina o risco: idade, escolaridade "
            "e o histórico de pagamento podem funcionar como proxy de grupos protegidos.",
            "**Diagnóstico de equidade.** O notebook mede, no conjunto de teste e por `SEX`, a "
            "taxa de aprovação e a taxa de verdadeiros positivos de cada grupo. É um "
            "diagnóstico de limitações, não uma garantia de equidade, e os dados são de "
            "Taiwan, 2005.",
            "**Decisão automatizada.** A LGPD (Lei 13.709/2018), art. 20, prevê o direito do "
            "titular de pedir a revisão de decisões tomadas unicamente com base em tratamento "
            "automatizado, inclusive as de perfil de crédito. É um resumo do projeto: confirme "
            "o texto vigente em planalto.gov.br antes de citá-lo. Um uso real exigiria "
            "explicação, revisão humana e canal de contestação, que este serviço não oferece.",
            "**Privacidade.** O serviço não guarda as entradas: os logs registram apenas o "
            "SHA-256 do corpo da requisição.",
            "**Não é uma decisão de crédito.** O resultado é uma probabilidade estimada por um "
            "modelo didático e nunca deve ser usado para aprovar ou negar crédito a uma pessoa.",
        ),
    ),
    "volatility-mlp": Notes(
        disclaimer=EDU_MARKET,
        intended_use=(
            "Estimar a volatilidade realizada dos próximos 5 pregões de uma ação do S&P 500, "
            "anualizada e em fração (0,20 = 20% ao ano), a partir de 20 features calculadas "
            "pelo chamador no fechamento do dia `t`.",
            "Demonstração didática da disciplina CIC407 (T2). Compara-se com três baselines: a "
            "média do treino, a persistência (`rv_5`) e um HAR linear.",
        ),
        out_of_scope=(
            "Recomendação de investimento, sinal de compra ou venda, dimensionamento de posição "
            "ou precificação de opções.",
            "Prever a direção do preço: o modelo estima a magnitude das variações, não o sinal.",
            "Ativos fora do S&P 500, ou regimes de mercado muito diferentes de 2013-2018.",
        ),
        data=SP500_DATA,
        caller=(
            "A API recebe **features já calculadas**: a engenharia de séries temporais é do "
            f"chamador (escolha registrada no [ADR 0010]({ADR_PIVOT})).",
            "Toda feature em `t` usa só dados até o fechamento de `t`. Definições: "
            "`ret_1d` a `ret_5d` são log-retornos defasados; `rv_5`, `rv_10`, `rv_20` e `rv_60` "
            "são volatilidades realizadas anualizadas; `pk_5` e `pk_20`, estimadores de "
            "Parkinson (high/low); `range_5`, a média de (high - low) / close; `volume_ratio`, "
            "o volume sobre a média de 20 dias; `gap`, open / close anterior - 1; `mkt_rv_20`, "
            "a volatilidade média do mercado (proxy de VIX sobre os 505 tickers). A fonte de "
            "verdade das fórmulas é a rede 2 do notebook (`notebook/parts/20_volatility.py`).",
            "A razão `rv_5 / rv_20` exige `rv_20 >= 1e-6`: abaixo disso (inclusive zero) a API "
            "responde 422 (`below_minimum`).",
        ),
        limitations=(
            *SP500_BIASES,
            "Se o chamador calcular uma feature de forma diferente da do notebook, surge "
            "*training-serving skew* que o golden da API **não detecta**: o golden cobre só o "
            "caminho da API (validação, razão, one-hot e grafo).",
            "O alvo de treino é ln(y) padronizado e a saída passa por `exp`: isso estima a "
            "mediana condicional da volatilidade, que tende a ficar abaixo da média quando o "
            "erro em log tem variância.",
            "Não há classe de rejeição nem intervalo de confiança: a saída é um número único.",
        ),
        ethics=(
            "**Não é recomendação.** A volatilidade prevista não é o risco total de uma "
            "posição e não indica retorno futuro. Não use o resultado para operar.",
            "**Vieses declarados.** Sobrevivência e regime único (veja as limitações) tendem a "
            "tornar as métricas otimistas para qualquer uso fora da amostra.",
            "**Dados.** Não há dados pessoais. Os preços não são redistribuídos por terem "
            "origem e licença não verificadas.",
        ),
    ),
    "candlestick-cnn": Notes(
        disclaimer=EDU_MARKET,
        intended_use=(
            "Classificar o regime de volatilidade dos próximos 5 pregões (`baixa`, `media` ou "
            "`alta`) a partir de um gráfico renderizado de 20 pregões: barras OHLC, média móvel "
            "de 20 dias e volume, em 64x64x3 pixels.",
            "Responde à pergunta de conclusão do projeto: o gráfico acrescenta algo além das "
            "features numéricas do `volatility-mlp`? A ideia vem de Jiang, Kelly e Xiu (2023), "
            "*(Re-)Imag(in)ing Price Trends*, Journal of Finance.",
            "Demonstração didática da disciplina CIC407 (T2) e exemplo de serviço de inferência "
            "com imagens.",
        ),
        out_of_scope=(
            "Recomendação de investimento ou previsão da direção do preço: o alvo é regime de "
            "volatilidade, e a direção foi descartada por ser quase aleatória.",
            "Gráficos que não sejam o produto de `tools/render_chart.py` (capturas de tela, "
            "outras plataformas, outra escala ou outro tamanho): a API aceita qualquer PNG "
            "válido, mas fora da rasterização do treino o resultado não tem significado.",
        ),
        data=(
            *SP500_DATA,
            "Rótulos: tercis de `y` (volatilidade realizada futura de 5 pregões) calculados só "
            "no treino. Chance = 33,3%. O treino usa em torno de 60 mil imagens, por "
            "amostragem com passo (PLANO v1.2, V4).",
            "Os golden samples do repositório são gráficos **sintéticos** (passeio aleatório "
            "geométrico), nunca imagens derivadas do CSV de preços.",
        ),
        caller=(
            "O chamador renderiza a janela de 20 pregões com a mesma função do notebook: "
            "`python tools/render_chart.py acao.csv -o grafico.png` (NumPy puro, sem "
            "antialiasing, pixels 0 ou 255). Um teste do apêndice do notebook garante que as "
            "duas cópias da função produzem arrays idênticos.",
            "Só PNG: JPEG destruiria o traço de 1 pixel, e a API responde 415 a ele.",
        ),
        limitations=(
            *SP500_BIASES,
            "O rótulo é função de `y`, e a volatilidade recente já prediz a futura (persistência): "
            "a amplitude das barras no gráfico carrega essa informação. A CNN pode ter só "
            "reaprendido a persistência, por isso o notebook a compara com o baseline tabular "
            "para o mesmo alvo de 3 classes.",
            "Os tercis vêm do treino. No teste, que inclui a alta de jan-fev/2018, a "
            "distribuição de classes não é uniforme.",
            "Resolução fixa e uma única escolha de rasterização (canais, escala vertical e "
            "janela); outras escolhas mudariam o resultado.",
        ),
        ethics=(
            "**Não é recomendação.** O artigo que inspira a ideia reporta ganhos em portfólio, "
            "e uma réplica menor concluiu que a CNN não passa de chute em acertos direcionais; "
            "este modelo não prevê direção e não deve embasar operações.",
            "**Vieses declarados.** Sobrevivência e regime único (veja as limitações).",
            "**Dados.** Não há dados pessoais. As imagens de mercado reais não são "
            "redistribuídas; o serviço não as guarda (os logs registram apenas o SHA-256 do "
            "corpo da requisição).",
        ),
    ),
}

NOTES_BY_TASK: dict[Task, Notes] = {
    task: Notes(
        disclaimer=EDU_MARKET,
        intended_use=(
            f"Modelo de {label} servido pela API TriNet. Veja `description` no manifesto.",
        ),
        out_of_scope=("Qualquer uso fora do domínio dos dados de treino.",),
        data=("Veja `training.dataset` e `training.data_sha256` na seção de proveniência.",),
        caller=(),
        limitations=("Sem texto curado para este modelo: documente limites e vieses do conjunto.",),
        ethics=("Avalie vieses e impactos do conjunto de dados antes de qualquer uso real.",),
    )
    for task, label in TASK_PT.items()
}


def notes_for(manifest: Manifest) -> Notes:
    return NOTES_BY_NAME.get(manifest.name) or NOTES_BY_TASK[manifest.task]


# --------------------------------------------------------------------------- descoberta


@dataclass(frozen=True, slots=True)
class Artifact:
    directory: Path
    manifest: Manifest
    results: Mapping[str, object]


def semver_key(version: str) -> tuple[int, int, int, int, str] | None:
    """Chave de ordenação semver: uma pré-versão fica abaixo da sua release."""
    match = _SEMVER.match(version)
    if match is None:
        return None
    major, minor, patch, pre = match.groups()
    return int(major), int(minor), int(patch), 0 if pre else 1, pre or ""


def load_results(path: Path) -> dict[str, object]:
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(key): value for key, value in data.items()} if isinstance(data, dict) else {}


def latest_version_dir(model_dir: Path) -> Path | None:
    candidates = [
        (key, child)
        for child in model_dir.iterdir()
        if child.is_dir() and not child.is_symlink() and (key := semver_key(child.name))
    ]
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def discover(root: Path) -> tuple[list[Artifact], list[str]]:
    """O artefato de maior versão de cada modelo, e os erros encontrados no caminho."""
    if not root.is_dir():
        return [], [f"{root}: diretório de artefatos não existe"]
    artifacts: list[Artifact] = []
    errors: list[str] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()):
        latest = latest_version_dir(model_dir)
        if latest is None:
            continue
        where = f"{model_dir.name}/{latest.name}"
        try:
            manifest = load_manifest(latest / "manifest.json")
        except (OSError, ValueError) as exc:
            first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
            errors.append(f"{where}: manifest inválido ({first})")
            continue
        if manifest.name != model_dir.name or manifest.version != latest.name:
            errors.append(f"{where}: o manifesto diz {manifest.name}@{manifest.version}")
            continue
        artifacts.append(Artifact(latest, manifest, load_results(latest / "results.json")))
    return artifacts, errors


# --------------------------------------------------------------------------- markdown


def cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(cell(c) for c in row) + " |" for row in rows)
    return lines


def bullets(items: Iterable[str]) -> list[str]:
    return [f"- {item}" for item in items]


def fmt(value: float) -> str:
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.2f}" if abs(value) >= 100 else f"{value:.4f}"


def shape_text(shape: Sequence[int | None]) -> str:
    return "(" + ", ".join("batch" if dim is None else str(dim) for dim in shape) + ")"


def scalar_text(value: object) -> str:
    if isinstance(value, float):
        return fmt(value)
    return str(value)


# --------------------------------------------------------------------------- seções do card


def banner(manifest: Manifest) -> list[str]:
    match manifest.status:
        case Status.DUMMY:
            return [
                "> **ARTEFATO DUMMY.** Pesos aleatórios gerados por "
                "`tools/make_dummy_artifacts.py` só para exercitar a API, a imagem e o CI. "
                "As métricas, o quality gate e o skew abaixo **não medem desempenho real** e "
                "não devem ser citados. O card é regenerado quando o notebook exportar o "
                "artefato real.",
                "",
            ]
        case Status.CANDIDATE:
            return [
                "> **Candidato.** Exportado pelo notebook, mas ainda não promovido a `release`. "
                "A release exige versão >= 1.0.0 e quality gate aprovado "
                "(`make check-artifacts-release`).",
                "",
            ]
        case Status.RELEASE:
            return ["> **Release.** Aprovado no quality gate de exportação.", ""]


def section_identity(a: Artifact) -> list[str]:
    m = a.manifest
    rows = [
        ("Versão", f"`{m.version}`"),
        ("Status", f"`{m.status.value}`"),
        ("Tarefa", f"{TASK_PT[m.task]} (`{m.task.value}`)"),
        ("Criado em", f"`{m.created_at.isoformat()}`"),
        ("Artefato", f"`artifacts/{m.name}/{m.version}/`"),
        ("Schema do contrato", f"`{m.schema_version}`"),
    ]
    lines = [f"# Model card: {m.name}", "", f"> {notes_for(m).disclaimer}", "", *banner(m)]
    if m.description:
        lines += [m.description, ""]
    return [*lines, *table(("Campo", "Valor"), rows), ""]


def section_use(a: Artifact) -> list[str]:
    notes = notes_for(a.manifest)
    return [
        "## 1. Uso pretendido",
        "",
        *bullets(notes.intended_use),
        "",
        "**Fora do escopo**",
        "",
        *bullets(notes.out_of_scope),
        "",
    ]


def section_data(a: Artifact) -> list[str]:
    m, notes = a.manifest, notes_for(a.manifest)
    lines = ["## 2. Dados", ""]
    if m.status is Status.DUMMY:
        lines += [
            f"Os dados deste artefato são sintéticos (`{m.training.dataset}`). "
            "Referência dos dados do modelo real:",
            "",
        ]
    lines += [*bullets(notes.data), ""]
    lines += [f"Hash dos dados de treino (SHA-256): `{m.training.data_sha256}`", ""]
    return lines


def field_rows(spec: TabularInput) -> list[tuple[str, str, str, str]]:
    rows = []
    for field in spec.fields:
        if isinstance(field, NumericField):
            bounds = []
            if field.minimum is not None:
                bounds.append(f">= {field.minimum:g}")
            if field.maximum is not None:
                bounds.append(f"<= {field.maximum:g}")
            valid, kind = " e ".join(bounds) or "qualquer número finito", "numérico"
        elif isinstance(field, CategoricalField):
            valid = "uma de: " + ", ".join(f"`{v}`" for v in field.vocabulary)
            kind = "categórico"
        missing = "obrigatório (HTTP 422)" if field.impute is None else f"`{field.impute}`"
        rows.append((f"`{field.name}`", kind, valid, missing))
    return rows


def feature_rows(spec: TabularInput) -> list[tuple[str, str, str]]:
    rows = []
    position = 0
    for feature in spec.features:
        if isinstance(feature, PassthroughFeature):
            width, kind = 1, "passthrough"
            text = f"valor numérico de `{feature.source}` (já imputado)"
        elif isinstance(feature, OneHotFeature):
            field = spec.field(feature.source)
            values = field.vocabulary if isinstance(field, CategoricalField) else ()
            width, kind = len(values), "one_hot"
            text = f"one-hot de `{feature.source}`, na ordem: " + ", ".join(
                f"`{v}`" for v in values
            )
        else:
            width, kind = 1, "ratio"
            text = f"`{feature.numerator} / {feature.denominator}` (HTTP 422 se o divisor for 0)"
        span = str(position) if width == 1 else f"{position}-{position + width - 1}"
        rows.append((span, kind, text))
        position += width
    return rows


def preprocessing_tabular(m: Manifest, spec: TabularInput) -> list[str]:
    lines = [
        "A API converte campos crus nomeados em um vetor float32 usando só as três primitivas "
        "fechadas do contrato (imputação por constante, one-hot por vocabulário e razão A/B). "
        "Campos desconhecidos, `null` sem imputação, valores não finitos e fora da faixa são "
        "rejeitados com HTTP 422.",
        "",
        "**Campos de entrada**",
        "",
        *table(("Campo", "Tipo", "Valores válidos", "Se ausente ou null"), field_rows(spec)),
        "",
        f"**Vetor de entrada do modelo** ({spec.width} colunas, nesta ordem)",
        "",
        *table(("Colunas", "Primitiva", "Definição"), feature_rows(spec)),
        "",
        "A padronização das entradas fica dentro do grafo do modelo, não na API.",
    ]
    if m.output.kind == "regression":
        lines.append("")
        lines.append(
            "A saída já sai desnormalizada, na unidade do alvo"
            + (
                " (o alvo de treino é ln(y) padronizado e o modelo termina em `exp`)."
                if has_exp_layer(m)
                else "."
            )
        )
    return lines


def preprocessing_image(spec: ImageInput) -> list[str]:
    low, high = spec.value_range
    return [
        "Pipeline da API (o mesmo que o notebook mede em `serving_skew`):",
        "",
        *bullets(
            (
                f"formatos aceitos: {', '.join(spec.formats)}, validados pelos magic bytes "
                f"(o tipo declarado pelo cliente não é confiado); até {spec.max_bytes} bytes "
                f"({spec.max_bytes / 1048576:g} MiB);",
                "checagem do tamanho pelo cabeçalho antes de decodificar (limite de pixels da "
                "API, `TRINET_MAX_IMAGE_PIXELS`);",
                "decodificação, conversão para RGB e, se o tamanho for diferente, "
                f"redimensionamento {spec.resample} (Pillow) para {spec.width}x{spec.height};",
                f"float32 na faixa [{low:g}, {high:g}] (o reescalonamento fica dentro do grafo);",
                "a orientação EXIF é ignorada de propósito.",
            )
        ),
    ]


def section_preprocessing(a: Artifact) -> list[str]:
    spec = a.manifest.input
    body = (
        preprocessing_tabular(a.manifest, spec)
        if isinstance(spec, TabularInput)
        else preprocessing_image(spec)
    )
    caller = notes_for(a.manifest).caller
    extra = ["", "**Responsabilidade do chamador**", "", *bullets(caller)] if caller else []
    return ["## 3. Pré-processamento", "", *body, *extra, ""]


def layer_text(layer: MlpLayer) -> str:
    if isinstance(layer, DenseLayer):
        return f"{layer.op}({layer.key}, {layer.activation})"
    if isinstance(layer, ExpLayer):  # sem pesos: não tem `key`
        return layer.op
    return f"{layer.op}({layer.key})"


def has_exp_layer(manifest: Manifest) -> bool:
    return any(
        isinstance(layer, ExpLayer)
        for runtime in manifest.runtimes
        if isinstance(runtime, NumpyMlpRuntimeSpec)
        for layer in runtime.layers
    )


def runtime_detail(runtime: RuntimeSpec) -> str:
    if isinstance(runtime, OnnxRuntimeSpec):
        return f"opset {runtime.opset}"
    if isinstance(runtime, NumpyMlpRuntimeSpec):
        return " > ".join(layer_text(layer) for layer in runtime.layers)
    return ""


def section_architecture(a: Artifact) -> list[str]:
    m = a.manifest
    tensor_in, tensor_out = m.inputs[0], m.outputs[0]
    if isinstance(m.output, ClassificationOutput):
        output_text = f"{len(m.output.labels)} classes: " + ", ".join(
            f"`{x}`" for x in m.output.labels
        )
    else:
        output_text = f"valor de `{m.output.target}` em {m.output.unit or 'unidade do alvo'}"
    rows = [
        (i + 1, f"`{r.kind}`", f"`{r.file}`", f"`{r.sha256[:12]}`", runtime_detail(r))
        for i, r in enumerate(m.runtimes)
    ]
    return [
        "## 4. Arquitetura e runtimes",
        "",
        f"- Entrada: tensor `{tensor_in.name}` {tensor_in.dtype}, forma "
        f"{shape_text(tensor_in.shape)}.",
        f"- Saída: tensor `{tensor_out.name}` {tensor_out.dtype}, forma "
        f"{shape_text(tensor_out.shape)}: {output_text}.",
        "",
        "Os runtimes seguem a ordem de preferência do manifesto (escada de fallback): a API usa o "
        "primeiro que passa na verificação de SHA-256 e no warm-up. "
        f"Veja o [ADR 0001]({ADR_FALLBACK}).",
        "",
        *table(("Ordem", "Runtime", "Arquivo", "SHA-256", "Detalhe"), rows),
        "",
        "A estrutura das camadas treináveis (larguras, profundidade) está no notebook, em "
        "`notebook/`, e, quando exportada, em `results.json`.",
        "",
    ]


def section_metrics(a: Artifact) -> list[str]:
    metrics = a.manifest.metrics
    lines = ["## 5. Métricas por split", ""]
    if not metrics:
        return [*lines, "Nenhuma métrica registrada no manifesto.", ""]
    columns = list(dict.fromkeys(name for split in metrics.values() for name in split))
    rows = [
        (f"`{split}`", *[fmt(values[c]) if c in values else "-" for c in columns])
        for split, values in metrics.items()
    ]
    return [*lines, *table(("Split", *columns), rows), ""]


def section_gate(a: Artifact) -> list[str]:
    gate = a.manifest.quality_gate
    rule = ">=" if gate.higher_is_better else "<="
    verdict = "aprovado" if gate.passed else "REPROVADO"
    return [
        "## 6. Quality gate de exportação",
        "",
        "O notebook se recusa a escrever o artefato se o gate falhar; o limiar é a métrica de "
        "validação com uma margem definida depois do treino.",
        "",
        *table(
            ("Métrica", "Split", "Regra", "Valor", "Resultado"),
            [
                (
                    f"`{gate.metric}`",
                    f"`{gate.split}`",
                    f"{rule} {fmt(gate.threshold)}",
                    fmt(gate.value),
                    verdict,
                )
            ],
        ),
        "",
    ]


def optional_value(value: float | None, render: str = "{:.4f}") -> str:
    return "não medido" if value is None else render.format(value)


def section_skew(a: Artifact) -> list[str]:
    m, skew = a.manifest, a.manifest.serving_skew
    lines = ["## 7. Training/serving skew", ""]
    if skew is None:
        return [*lines, "Não medido neste artefato.", ""]
    regression = m.task is Task.REGRESSION
    not_applicable = "não se aplica (regressão)"
    rows = [
        ("Amostras comparadas", skew.samples),
        ("Maior diferença absoluta (Keras x runtime)", f"{skew.max_abs_diff:.3e}"),
        (
            "Concordância do top-1",
            not_applicable if regression else optional_value(skew.top1_agreement),
        ),
        (
            "Variação de acurácia (pontos percentuais)",
            not_applicable if regression else optional_value(skew.accuracy_delta_pp, "{:+.2f}"),
        ),
    ]
    lines += table(("Medida", "Valor"), rows)
    if m.task is Task.IMAGE_CLASSIFICATION:
        ok = (
            skew.top1_agreement is not None
            and skew.top1_agreement >= 0.995
            and abs(skew.accuracy_delta_pp or 0.0) <= 0.3
        )
        lines += [
            "",
            "Critério do projeto para imagens (PLANO 4.5): top-1 igual em pelo menos 99,5% e "
            "|variação de acurácia| <= 0,3 pp. Situação: "
            + ("atendido." if ok else "NÃO atendido ou incompleto."),
        ]
    return [*lines, ""]


def section_limitations(a: Artifact) -> list[str]:
    notes = notes_for(a.manifest)
    generic = (
        "As métricas valem para a distribuição dos dados de treino e teste; fora dela o "
        "desempenho não é garantido.",
        "A API expõe contadores por classe prevista, mas não histogramas por feature; o que dá e "
        f"o que não dá para detectar de drift está no [ADR 0007]({ADR_DRIFT}).",
    )
    return [
        "## 8. Limitações",
        "",
        *bullets((*notes.limitations, *generic)),
        "",
        "## 9. Considerações éticas e de uso",
        "",
        *bullets(notes.ethics),
        "",
        f"Contexto da escolha dos dados e dos alvos: [ADR 0010]({ADR_PIVOT}).",
        "",
    ]


def golden_summary(a: Artifact) -> str:
    ref = a.manifest.golden
    if ref is None:
        return "sem golden samples"
    try:
        golden = load_golden(a.directory / ref.file)
    except (OSError, ValueError):
        return f"`{ref.file}` (ilegível)"
    labelled = sum(1 for case in golden.cases if case.label is not None)
    kind = "imagens PNG" if isinstance(golden, ImageGolden) else "registros crus"
    return f"`{ref.file}`: {len(golden.cases)} casos ({kind}), {labelled} rotulados"


def section_provenance(a: Artifact) -> list[str]:
    m = a.manifest
    t = m.training
    rows = [
        ("Git SHA", f"`{m.git_sha}`" if m.git_sha else "não registrado"),
        ("Conjunto de dados", t.dataset),
        ("SHA-256 dos dados", f"`{t.data_sha256}`"),
        ("Seed", t.seed),
        ("Python", t.python),
        ("TensorFlow", t.tensorflow),
        ("Keras", t.keras),
        ("Hardware", t.hardware or "não registrado"),
        ("Golden samples", golden_summary(a)),
        *[(f"SHA-256 de `{r.file}`", f"`{r.sha256}`") for r in m.runtimes],
    ]
    return ["## 10. Proveniência", "", *table(("Campo", "Valor"), rows), ""]


# --------------------------------------------------------------------------- results.json


def is_scalar(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def render_extra(key: str, value: object) -> list[str]:
    heading = [f"### `{key}`", ""]
    if isinstance(value, dict) and value and all(is_scalar(v) for v in value.values()):
        pairs = [(f"`{k}`", scalar_text(v)) for k, v in list(value.items())[:MAX_ROWS]]
        return [*heading, *table(("Chave", "Valor"), pairs), ""]
    if isinstance(value, list) and value and all(isinstance(r, dict) for r in value):
        columns = list(dict.fromkeys(str(k) for row in value[:MAX_ROWS] for k in row))
        rows = [[scalar_text(row.get(c, "")) for c in columns] for row in value[:MAX_ROWS]]
        return [*heading, *table(columns, rows), ""]
    if is_scalar(value):
        return [*heading, scalar_text(value), ""]
    return [*heading, "```json", *json_lines(value)[:MAX_JSON_LINES], "```", ""]


def json_lines(value: object) -> list[str]:
    """JSON legível; matrizes (como a de confusão) ficam com uma linha por linha da matriz."""
    if isinstance(value, list) and value and all(isinstance(r, list) for r in value):
        rows = [json.dumps(row, ensure_ascii=False) for row in value]
        return ["[", *[f"  {row}," for row in rows[:-1]], f"  {rows[-1]}", "]"]
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True).splitlines()


def section_extras(a: Artifact) -> list[str]:
    skip = _RESULTS_KNOWN | ({"note"} if a.manifest.status is Status.DUMMY else set())
    extras = {k: v for k, v in a.results.items() if k not in skip}
    if not extras:
        return []
    lines = ["## 11. Resultados adicionais (`results.json`)", ""]
    for key in sorted(extras):
        lines += render_extra(key, extras[key])
    return lines


def render_card(a: Artifact) -> str:
    m = a.manifest
    sections = [
        section_identity(a),
        section_use(a),
        section_data(a),
        section_preprocessing(a),
        section_architecture(a),
        section_metrics(a),
        section_gate(a),
        section_skew(a),
        section_limitations(a),
        section_provenance(a),
        section_extras(a),
        [
            "---",
            "",
            f"Gerado por `tools/gen_model_card.py` a partir de "
            f"`artifacts/{m.name}/{m.version}/manifest.json`. Não edite à mão: rode "
            f"`{REGENERATE}`.",
        ],
    ]
    return "\n".join(line for section in sections for line in section).rstrip() + "\n"


# --------------------------------------------------------------------------- índice e README


def metrics_on_test(m: Manifest) -> str:
    values = m.metrics.get("test", {})
    return "; ".join(f"{name} {fmt(value)}" for name, value in values.items()) or "-"


def gate_text(m: Manifest) -> str:
    gate = m.quality_gate
    rule = ">=" if gate.higher_is_better else "<="
    return f"{gate.metric} {rule} {fmt(gate.threshold)} ({'ok' if gate.passed else 'falhou'})"


def results_table(artifacts: Sequence[Artifact], prefix: str) -> list[str]:
    rows = []
    for a in artifacts:
        m = a.manifest
        dummy = m.status is Status.DUMMY
        rows.append(
            (
                f"`{m.name}`",
                TASK_PT[m.task],
                f"`{m.version}`",
                f"`{m.status.value}`",
                "pendente (artefato dummy)" if dummy else metrics_on_test(m),
                "pendente" if dummy else gate_text(m),
                f"[card]({prefix}{m.name}.md)",
            )
        )
    header = ("Modelo", "Tarefa", "Versão", "Status", "Métricas no teste", "Quality gate", "Card")
    return table(header, rows)


def pending_note(artifacts: Sequence[Artifact]) -> list[str]:
    if not artifacts:
        return [
            "**Pendente.** Nenhum artefato encontrado em `artifacts/`.",
            "",
        ]
    if all(a.manifest.status is Status.DUMMY for a in artifacts):
        return [
            "**Resultados pendentes.** Os artefatos versionados em `artifacts/` ainda são "
            "`dummy` (pesos aleatórios de `make dummy`, só para exercitar a API e o CI). Os "
            "números reais aparecem aqui depois que o notebook exportar os artefatos `1.0.0` e "
            f"`{REGENERATE}` for executado.",
            "",
        ]
    return []


def readme_block(artifacts: Sequence[Artifact]) -> str:
    lines = [
        README_START,
        f"<!-- Gerado por tools/gen_model_card.py ({REGENERATE}); não edite aqui -->",
        "",
        *pending_note(artifacts),
    ]
    if artifacts:
        lines += [*results_table(artifacts, "docs/model-cards/"), ""]
    lines.append(README_END)
    return "\n".join(lines)


def render_index(artifacts: Sequence[Artifact]) -> str:
    lines = [
        "# Model cards",
        "",
        "Um card por modelo, gerado a partir do artefato de maior versão em `artifacts/`.",
        "",
        *pending_note(artifacts),
    ]
    if artifacts:
        lines += [*results_table(artifacts, ""), ""]
    lines += [
        "---",
        "",
        f"Gerado por `tools/gen_model_card.py`. Rode `{REGENERATE}` para atualizar.",
    ]
    return "\n".join(lines) + "\n"


def replace_block(text: str, block: str) -> str | None:
    start, end = text.find(README_START), text.find(README_END)
    if start == -1 or end == -1 or end < start:
        return None
    return text[:start] + block + text[end + len(README_END) :]


# --------------------------------------------------------------------------- CLI


def build_outputs(
    artifacts: Sequence[Artifact], out_dir: Path, readme: Path | None
) -> tuple[dict[Path, str], list[str]]:
    outputs = {out_dir / f"{a.manifest.name}.md": render_card(a) for a in artifacts}
    outputs[out_dir / "README.md"] = render_index(artifacts)
    warnings: list[str] = []
    if readme is not None:
        original = readme.read_text(encoding="utf-8") if readme.is_file() else ""
        updated = replace_block(original, readme_block(artifacts))
        if updated is None:
            warnings.append(f"{readme}: marcadores {README_START} / {README_END} ausentes; pulado")
        else:
            outputs[readme] = updated
    return outputs, warnings


def changed(outputs: Mapping[Path, str]) -> list[Path]:
    def current(path: Path) -> str | None:
        return path.read_text(encoding="utf-8") if path.is_file() else None

    return [path for path, content in outputs.items() if current(path) != content]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument(
        "--artifacts", type=Path, default=DEFAULT_ARTIFACTS, help="raiz dos artefatos"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="pasta dos cards")
    parser.add_argument("--readme", type=Path, default=DEFAULT_README, help="README com marcadores")
    parser.add_argument("--no-readme", action="store_true", help="não atualiza o README da raiz")
    parser.add_argument("--check", action="store_true", help="falha se algum arquivo mudaria")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts, errors = discover(args.artifacts)
    outputs, warnings = build_outputs(artifacts, args.out, None if args.no_readme else args.readme)
    for message in warnings:
        print(f"aviso: {message}", file=sys.stderr)
    for message in errors:
        print(f"erro: {message}", file=sys.stderr)
    stale = changed(outputs)
    if args.check:
        for path in stale:
            print(f"desatualizado: {path}", file=sys.stderr)
        return 1 if stale or errors else 0
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(outputs[path], encoding="utf-8", newline="\n")
        print(f"escrito: {path}")
    if not stale:
        print("nada a atualizar")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
