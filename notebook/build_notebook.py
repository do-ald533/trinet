"""Monta ``T2_TriNet.ipynb`` a partir de ``parts/*.py`` (formato "percent" do jupytext, sem depender dele).

Uso (a partir de ``notebook/``)::

    python build_notebook.py                      # monta o notebook inteiro (partes em ordem de nome de arquivo)
    python build_notebook.py --only 00,10         # monta so as partes cujo prefixo numerico esta na lista
    python build_notebook.py --script run.py      # emite um unico .py (celulas viram comentarios) para rodar rapido
    python build_notebook.py --execute            # monta e executa (nbclient), gravando as saidas no proprio .ipynb
    python build_notebook.py --only 00,10,15 --execute --out /tmp/parcial.ipynb
    python build_notebook.py --sync-markdown      # reescreve so o TEXTO de um .ipynb ja executado, mantendo as saidas
    python build_notebook.py --sync-markdown --integrantes "Fulano (RA 1); Beltrano (RA 2)" --drop-parts 95,96 --out T2_TriNet_entrega.ipynb

Formato de cada parte: ``# %% [markdown]`` (linhas ``# texto``) e ``# %%`` (codigo). Tudo antes do primeiro marcador
(docstring do modulo) e ignorado. Ordem: 00, 10, 15, 20, 30 (CNN), 90, 95 e 96 (apendices), pelo nome do arquivo.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS_DIR = HERE / "parts"
DEFAULT_OUT = HERE / "T2_TriNet.ipynb"
MARKER = re.compile(r"^# %%(?P<rest>.*)$")
PLACEHOLDER = "*a preencher*"  # nomes e RA ficam fora do repositorio publico: entram so na copia entregue


def part_files(only: set[str] | None = None) -> list[Path]:
    files = sorted(p for p in PARTS_DIR.glob("[0-9][0-9]_*.py"))
    if only is not None:
        files = [p for p in files if p.name[:2] in only]
        missing = only - {p.name[:2] for p in files}
        if missing:
            print(f"aviso: partes sem arquivo (ainda): {sorted(missing)}", file=sys.stderr)
    return files


def parse_part(path: Path) -> list[tuple[str, str]]:
    """Devolve ``[(tipo, fonte)]`` com tipo ``markdown`` ou ``code``."""
    cells: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = MARKER.match(line)
        if m:
            current = ("markdown" if "[markdown]" in m.group("rest") else "code", [])
            cells.append(current)
        elif current is not None:
            current[1].append(line)
    out: list[tuple[str, str]] = []
    for kind, lines in cells:
        if kind == "markdown":
            text = "\n".join(ln[2:] if ln.startswith("# ") else ln[1:] if ln.startswith("#") else ln for ln in lines)
        else:
            text = "\n".join(lines)
        text = text.strip("\n")
        if text.strip():
            out.append((kind, text))
    return out


def check_code(path: Path, index: int, source: str) -> None:
    try:  # magics (%..., !...) nao sao Python valido: aqui viram erro de sintaxe
        ast.parse(source)
    except SyntaxError as exc:
        raise SystemExit(f"{path.name} celula {index}: erro de sintaxe: {exc}") from exc


def collect(only: set[str] | None) -> list[tuple[Path, str, str]]:
    cells: list[tuple[Path, str, str]] = []
    for path in part_files(only):
        for i, (kind, text) in enumerate(parse_part(path), 1):
            if kind == "code":
                check_code(path, i, text)
            cells.append((path, kind, text))
    return cells


def cell_id(path: Path, kind: str, text: str, seen: dict[str, int]) -> str:
    digest = hashlib.sha1(f"{path.name}:{kind}:{text}".encode()).hexdigest()[:8]
    seen[digest] = seen.get(digest, 0) + 1
    return f"{path.name[:2]}-{digest}-{seen[digest]}"


def apply_integrantes(cells: list[tuple[Path, str, str]], integrantes: str | None) -> list[tuple[Path, str, str]]:
    """Troca o marcador da capa pelos nomes e RA (so na copia entregue; ``parts/`` nunca guarda identificacao)."""
    if not integrantes:
        return cells
    if not any(kind == "markdown" and PLACEHOLDER in text for _, kind, text in cells):
        raise SystemExit(f"marcador {PLACEHOLDER!r} nao encontrado na capa")
    return [(p, k, t.replace(PLACEHOLDER, integrantes) if k == "markdown" else t) for p, k, t in cells]


def build_notebook(cells: list[tuple[Path, str, str]]):
    import nbformat
    from nbformat import v4

    nb = v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["title"] = "TriNet - T2 Aprendizado Profundo (CIC407)"
    seen: dict[str, int] = {}
    for path, kind, text in cells:
        new = v4.new_markdown_cell(text) if kind == "markdown" else v4.new_code_cell(text)
        new["id"] = cell_id(path, kind, text, seen)
        nb.cells.append(new)
    nbformat.validate(nb)
    return nb


def sync_markdown(src: Path, out: Path, cells: list[tuple[Path, str, str]], drop: set[str] | None = None) -> int:
    """Reescreve so as celulas de TEXTO de um .ipynb ja executado a partir de ``parts/``, preservando as saidas.

    Recusa se o numero, a ordem ou o codigo das celulas divergir de ``parts/`` (nesse caso o notebook precisa ser
    reexecutado com ``--execute``). Usa so ``json``: o formato gravado e o mesmo do nbformat (indent=1, chaves ordenadas).
    ``drop`` remove da copia gravada as celulas das partes listadas (ex.: ``{"95", "96"}``, os apendices de exportacao).
    Devolve quantas celulas de texto mudaram.
    """
    nb = json.loads(src.read_text(encoding="utf-8"))
    old = nb["cells"]
    if len(old) != len(cells):
        raise SystemExit(f"{src.name} tem {len(old)} celulas e parts/ tem {len(cells)}: reexecute com --execute")
    seen: dict[str, int] = {}
    changed = 0
    for i, (cell, (path, kind, text)) in enumerate(zip(old, cells, strict=True)):
        if cell["cell_type"] != kind:
            raise SystemExit(f"celula {i}: tipo mudou ({cell['cell_type']} -> {kind}): reexecute com --execute")
        if kind == "code" and "".join(cell["source"]) != text:
            raise SystemExit(f"celula {i} ({path.name}): o codigo mudou: reexecute com --execute para regerar as saidas")
        if kind == "markdown" and "".join(cell["source"]) != text:
            cell["source"] = text.splitlines(keepends=True)
            changed += 1
        cell["id"] = cell_id(path, kind, text, seen)
    if drop:
        nb["cells"] = [c for c, (path, _, _) in zip(old, cells, strict=True) if path.name[:2] not in drop]
    dumped = json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    out.write_text(dumped.replace(str(HERE.parent), "<repo>").replace(str(Path.home()), "~"), encoding="utf-8")
    return changed


def write_script(cells: list[tuple[Path, str, str]], out: Path) -> None:
    chunks = ["# gerado por build_notebook.py --script (nao editar)\n"]
    for path, kind, text in cells:
        if kind == "markdown":
            chunks.append("\n".join("# " + ln if ln else "#" for ln in text.splitlines()))
        else:
            chunks.append(f"# --- {path.name}\n{text}")
    out.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")


def write_scrubbed(nb, out: Path) -> None:
    """Grava o .ipynb trocando caminhos absolutos desta maquina (repositorio e HOME) por marcadores.

    As saidas das celulas mostram caminhos de artefatos e de ambientes virtuais; o notebook vai para um repositorio
    publico, entao o nome de usuario e a estrutura de pastas locais nao devem ir junto.
    """
    import nbformat

    text = nbformat.writes(nb)
    text = text.replace(str(HERE.parent), "<repo>").replace(str(Path.home()), "~")
    out.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def execute(nb, out: Path, timeout: int) -> None:
    import nbformat
    from nbclient import NotebookClient

    t0 = time.time()
    state = {"n": 0}
    total = sum(c.cell_type == "code" for c in nb.cells)

    def done(cell, cell_index, execute_reply):  # noqa: ARG001
        state["n"] += 1
        first = cell.source.strip().splitlines()[0][:70] if cell.source.strip() else ""
        print(f"[{time.time() - t0:7.0f}s] celula {state['n']}/{total}: {first}", flush=True)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
        allow_errors=False,
    )
    client.on_cell_executed = done
    try:
        client.execute()
    finally:
        write_scrubbed(nb, out)  # grava mesmo se falhar, para inspecionar a saida parcial
    print(f"execucao total: {(time.time() - t0) / 60:.1f} min -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="prefixos das partes, separados por virgula (ex.: 00,10,15)")
    ap.add_argument("--script", type=Path, help="emite um unico .py em vez do notebook")
    ap.add_argument("--execute", action="store_true", help="executa o notebook (nbclient) e grava as saidas")
    ap.add_argument("--timeout", type=int, default=14400, help="timeout por celula, em segundos (padrao 4 h)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="notebook de saida")
    ap.add_argument("--sync-markdown", action="store_true", help="atualiza so o texto de um .ipynb ja executado (le DEFAULT_OUT ou --src)")
    ap.add_argument("--src", type=Path, default=DEFAULT_OUT, help="com --sync-markdown: notebook executado de origem")
    ap.add_argument("--drop-parts", help="com --sync-markdown: prefixos das partes a omitir da copia (ex.: 95,96 = apendices de exportacao)")
    ap.add_argument("--integrantes", help="nomes completos e RA, no lugar do marcador da capa (ex.: 'Fulano (RA 1); Beltrano (RA 2)')")
    args = ap.parse_args()

    only = {p.strip() for p in args.only.split(",")} if args.only else None
    cells = apply_integrantes(collect(only), args.integrantes)
    if not cells:
        raise SystemExit("nenhuma celula encontrada")
    if args.sync_markdown:
        if only is not None:
            raise SystemExit("--sync-markdown nao combina com --only")
        drop = {p.strip() for p in args.drop_parts.split(",")} if args.drop_parts else None
        changed = sync_markdown(args.src, args.out, cells, drop)
        print(f"{changed} celulas de texto atualizadas, saidas preservadas -> {args.out}")
        return
    if args.script:
        write_script(cells, args.script)
        print(f"{len(cells)} celulas -> {args.script}")
        return
    import nbformat

    nb = build_notebook(cells)
    nbformat.write(nb, args.out)
    print(f"{len(cells)} celulas ({sum(c.cell_type == 'code' for c in nb.cells)} de codigo) -> {args.out}")
    if args.execute:
        execute(nb, args.out, args.timeout)


if __name__ == "__main__":
    main()
