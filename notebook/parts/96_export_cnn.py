"""Parte 96 - apendice: exporta a Rede 3 (``candlestick-cnn``) para ``artifacts/candlestick-cnn/1.0.0/`` (PLANO 4.5).

Fora da area avaliada. Depende da parte 30 (``cnn_model``, ``cnn_edges``, ``img_x_val/test``, ``cnn_*`` metricas) e dos
modulos ``trinet_export`` / ``trinet_export_image`` (ao lado do notebook) + ``trinet-contract``, importados dentro de
``try/except ImportError`` para o notebook continuar autocontido.
"""

# %% [markdown]
# # Apêndice · Exportação da CNN para a API (`candlestick-cnn`)
#
# > **Fora da área avaliada.** Transforma o modelo treinado em um **artefato** consumido pela API de inferência
# > (`artifacts/candlestick-cnn/1.0.0/`): `model.onnx`, `manifest.json`, `golden.json` + PNGs de referência e `results.json`.
#
# O que a célula de exportação faz, na ordem (qualquer falha **interrompe antes de escrever** o artefato):
#
# 1. **Teste de igualdade da rasterização:** a cópia inline do notebook e `tools/render_chart.py` devem gerar arrays e PNGs
#    **idênticos bit a bit** (e a CLI da ferramenta deve reproduzir o mesmo gráfico a partir de um CSV).
# 2. **Clone de inferência em float32:** sem `Dropout` e sem augmentation, pesos copiados, entrada `float32` em 0–255
#    (o `Rescaling(1/255)` continua *dentro* do grafo).
# 3. **ONNX** com `Model.export(format="onnx", opset_version=17)`; nomes e formas dos tensores lidos do ONNX Runtime.
# 4. **Training/serving skew no conjunto de teste inteiro:** caminho Keras (array `uint8` do treino) contra o caminho de
#    produção da API (gráfico codificado em **PNG**, decodificado com **Pillow** → RGB → `float32` 0–255 → ONNX Runtime).
#    Exige concordância do top-1 ≥ 99,5% e |Δ acurácia| ≤ 0,3 pp.
# 5. **Quality gate:** acurácia de validação *pelo caminho de serving* ≥ (acurácia de validação do Keras − margem de 3 pp).
# 6. **Golden samples:** 20 gráficos **sintéticos** (passeio aleatório geométrico, nunca derivados do CSV do Kaggle), 4 deles em
#    tamanhos diferentes de 64×64 para exercitar o *resize* da API; saídas esperadas do modelo Keras com o pré-processamento da API.
# 7. `manifest.json` + `golden.json` + `results.json`, validados com `trinet-contract` e relidos antes de publicar (escrita atômica).

# %%
# os, platform, sys, time, Path, np e tf vêm da parte 0; json e tempfile, da parte 95 (notebook montado na ordem das partes).
_t_export = time.time()
try:  # so o apendice depende destes modulos
    import onnx
    import onnxruntime as ort
    import trinet_export as tex
    import trinet_export_image as tei

    from trinet_contract import load_golden, load_manifest

    EXPORT_OK = True
except ImportError as _exc:
    EXPORT_OK = False
    print(f"Apêndice de exportação desativado (dependência ausente: {_exc}). O restante do notebook não é afetado.")


def records(table) -> list[dict]:
    """Linhas de uma tabela do pandas como dicts JSON validos (NaN -> null)."""
    return json.loads(table.reset_index().to_json(orient="records", force_ascii=False))


def _find_repo_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "tools" / "render_chart.py").is_file():
            return candidate
    return None


REPO_ROOT = _find_repo_root(Path.cwd().resolve())
ARTIFACT_ROOT = Path(os.environ.get("TRINET_ARTIFACTS_DIR") or (REPO_ROOT / "artifacts" if REPO_ROOT else "artifacts")).resolve()
CNN_VERSION = "1.0.0"
CNN_GATE_MARGIN = 0.03  # margem do quality gate sobre a acuracia de validacao
CNN_SKEW_MIN_TOP1, CNN_SKEW_MAX_DACC_PP = 0.995, 0.3
print(f"exportação ativa: {EXPORT_OK} | raiz do repositório: {REPO_ROOT} | artefatos em: {ARTIFACT_ROOT}")


# %%
def export_candlestick_cnn() -> Path:
    """Executa os passos 1 a 7 e devolve a pasta do artefato escrito."""
    import kagglehub
    from sklearn.metrics import f1_score

    if REPO_ROOT is None:
        raise RuntimeError("tools/render_chart.py nao encontrado: rode o notebook a partir do repositorio")
    nb_render = {"render_windows": render_windows, "render_chart": render_chart, "png_bytes": png_bytes}

    # 1) igualdade da rasterizacao: notebook x tools/render_chart.py
    tool = tei.load_tool_renderer(REPO_ROOT)
    probe, _ = tei.synthetic_ohlcv(6, seed=1)
    renderer_check = tei.assert_same_renderer(nb_render, tool, probe)
    renderer_check["cli_matches"] = tei.check_cli(tool, probe[0])
    assert renderer_check["cli_matches"], "a CLI de tools/render_chart.py nao reproduziu o grafico"
    print("rasterização idêntica (notebook × tools/render_chart.py):", renderer_check["checks"], "| CLI ok")

    # 2) clone de inferencia (float32, sem Dropout/augmentation)
    infer = tei.rebuild_inference_cnn(cnn_model)
    probe_imgs = img_x_test[:512].astype("float32")
    clone_diff = float(np.abs(infer.predict(probe_imgs, verbose=0) - cnn_model.predict(probe_imgs, verbose=0)).max())
    assert clone_diff < 1e-5, f"clone de inferencia difere do modelo treinado: {clone_diff}"

    # 3) ONNX (tf2onnx via Model.export) + inspecao com onnxruntime
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        tex.export_onnx(infer, onnx_path, tei.INPUT_NAME, opset=tex.ONNX_OPSET)
        onnx_info = tex.inspect_onnx(onnx_path)
        onnx_bytes = onnx_path.read_bytes()
    assert onnx_info["input"]["shape"][1:] == [64, 64, 3], onnx_info
    assert onnx_info["output"]["shape"][1:] == [3], onnx_info
    print(f"ONNX: opset {onnx_info['opset']}, {onnx_info['bytes'] / 1e6:.2f} MB, ops {onnx_info['ops']}")

    # 4) serving skew no teste completo (caminho Keras x caminho de producao da API)
    t0 = time.time()
    keras_probs = cnn_model.predict(img_x_test, batch_size=512, verbose=0)
    served_probs = tei.serving_path_probs(onnx_bytes, img_x_test, png_bytes)
    skew = tei.skew_report(keras_probs, served_probs, img_y_test)
    print(f"serving skew ({skew['samples']:,d} imagens, {time.time() - t0:.0f}s): {skew}")
    if skew["top1_agreement"] < CNN_SKEW_MIN_TOP1 or skew["accuracy_delta_pp"] > CNN_SKEW_MAX_DACC_PP:
        raise RuntimeError(f"SERVING SKEW acima do limite ({skew}); artefato NAO escrito")

    # 5) quality gate: acuracia de validacao pelo caminho de servico vs. validacao do Keras menos a margem
    served_val_acc = float((tei.serving_path_probs(onnx_bytes, img_x_val, png_bytes).argmax(axis=1) == img_y_val).mean())
    threshold = float(np.floor((cnn_val_acc - CNN_GATE_MARGIN) * 100) / 100)
    assert threshold > 1 / 3, "limiar do gate nao supera o acaso"
    gate = tex.check_gate("accuracy", "validation", served_val_acc, threshold, True)

    # 6) golden samples sinteticos (PNG tiny; alguns em tamanhos diferentes para exercitar o resize da API)
    windows, sigma = tei.synthetic_ohlcv(20, seed=2026, edges=cnn_edges)
    charts = render_windows(windows)
    odd_sizes = {16: (128, 128), 17: (96, 80), 18: (200, 150), 19: (48, 64)}  # (largura, altura)
    golden_pngs: dict[str, bytes] = {}
    names = []
    truth = np.digitize(sigma, cnn_edges)  # regime "verdadeiro" pela volatilidade do gerador
    for i, chart in enumerate(charts):
        data = tei.encode_resized_png(chart, odd_sizes[i]) if i in odd_sizes else png_bytes(chart)
        name = f"synthetic_{i:02d}_{REGIME_NAMES[truth[i]]}.png"
        assert len(data) < 20_000, (name, len(data))
        golden_pngs[name] = data
        names.append(name)
    golden_batch = np.stack([tex.pillow_preprocess(golden_pngs[n], 64, 64) for n in names]).astype(np.float32)
    golden_probs = cnn_model.predict(golden_batch, verbose=0)
    synthetic_acc = float((golden_probs.argmax(axis=1) == truth).mean())
    golden = {
        "schema_version": "1.0",
        "kind": "image",
        "model": tei.IMAGE_MODEL_NAME,
        "version": CNN_VERSION,
        "directory": "golden",
        "cases": [
            {"file": n, "output": [float(v) for v in p], "label": REGIME_NAMES[int(t)]}
            for n, p, t in zip(names, golden_probs, truth, strict=True)
        ],
    }

    # 7) manifesto, results.json e escrita atomica validada com o contrato
    csv_file = next(Path(kagglehub.dataset_download("camnugent/sandp500")).rglob("all_stocks_5yr.csv"))
    pred_test = keras_probs.argmax(axis=1)
    versions = {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "tf2onnx": __import__("tf2onnx").__version__,
    }
    hardware = globals().get("HW") or tei.hardware_string()
    manifest_fields = {
        "onnx_opset": onnx_info["opset"],
        "name": tei.IMAGE_MODEL_NAME,
        "version": CNN_VERSION,
        "status": "candidate",
        "task": "image_classification",
        "description": (
            "Regime de volatilidade dos proximos 5 pregoes (baixa/media/alta, tercis do treino) a partir de um grafico "
            "OHLC + media movel + volume de 20 pregoes de acoes do S&P 500, renderizado em 64x64x3. Uso educacional; "
            "nao e recomendacao de investimento."
        ),
        "created_at": tex.utc_now(),
        "git_sha": tei.git_sha(REPO_ROOT),
        "input": {
            "kind": "image",
            "height": 64,
            "width": 64,
            "channels": 3,
            "resample": "bilinear",
            "value_range": [0.0, 255.0],
            "max_bytes": 262_144,
            "formats": ["PNG"],
        },
        "output": {"kind": "classification", "labels": list(REGIME_NAMES)},
        "inputs": [{"name": onnx_info["input"]["name"], "dtype": "float32", "shape": onnx_info["input"]["shape"]}],
        "outputs": [{"name": onnx_info["output"]["name"], "dtype": "float32", "shape": onnx_info["output"]["shape"]}],
        "metrics": {
            "validation": {"accuracy": float(cnn_val_acc), "loss": float(cnn_val_loss)},
            "test": {"accuracy": float(cnn_test_acc), "macro_f1": float(f1_score(img_y_test, pred_test, average="macro"))},
        },
        "quality_gate": gate,
        "serving_skew": {k: skew[k] for k in ("samples", "max_abs_diff", "top1_agreement", "accuracy_delta_pp")},
        "training": {
            "dataset": "S&P 500 5 anos (Kaggle camnugent/sandp500), janelas OHLCV de 20 pregoes renderizadas em 64x64x3",
            "data_sha256": tei.file_sha256(csv_file),
            "seed": int(globals().get("CNN_SEED", 42)),
            "python": versions["python"],
            "tensorflow": versions["tensorflow"],
            "keras": versions["keras"],
            "hardware": hardware,
        },
    }
    baselines = {
        "chance": 1 / 3,
        "classe_majoritaria_treino_no_teste": float((img_y_test == np.bincount(img_y_train).argmax()).mean()),
        "har_logistico_mesmas_imagens": float(cnn_har_acc_test),
    }
    if cnn_rede2_available:  # numeros da Rede 2 (parte 2); na execucao isolada da parte 3 nao existem
        for key, var in (("rede2_har", "vol_baseline_acc_har"), ("rede2_mlp", "vol_baseline_acc_mlp")):
            value = globals().get(var)
            if value is not None and np.isfinite(value):
                baselines[key] = float(value)
    results = {
        "model": tei.IMAGE_MODEL_NAME,
        "version": CNN_VERSION,
        "status": "candidate",
        "metrics": manifest_fields["metrics"],
        "quality_gate": gate,
        "serving_skew": manifest_fields["serving_skew"],
        "baselines_acuracia_teste": baselines,
        "direction_sanity": cnn_direction_acc,
        "ablation": records(cnn_ablation),
        "lr_sweep": records(cnn_lr_table),
        "pilot_escala_vertical": records(cnn_pilot_table) if len(cnn_pilot_table) else [],
        "confusion_matrix_test": cnn_confusion.tolist(),
        "dataset": {
            "stride": int(CNN_STRIDE),
            "imagens_treino": len(img_x_train),
            "imagens_validacao": len(img_x_val),
            "imagens_teste": len(img_x_test),
            "tercis_y_treino": [float(v) for v in cnn_edges],
            "classes": list(REGIME_NAMES),
        },
        "architecture": {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cnn_cfg.items()}, "parametros": int(cnn_model.count_params())},
        "timings_s": {
            "epoca_media": float(cnn_epoch_seconds),
            "primeira_epoca": float(cnn_first_epoch_seconds),
            "renderizacao_imagens": float(cnn_render_seconds),
            "melhor_epoca": int(cnn_best_epoch),
        },
        "export": {
            "rota": "tf2onnx via Model.export(format='onnx', opset_version=17)",
            "onnx": {k: onnx_info[k] for k in ("opset", "ir_version", "ops", "bytes")},
            "clone_vs_treinado_max_abs_diff": clone_diff,
            "golden_casos": len(names),
            "golden_png_maior_bytes": max(len(b) for b in golden_pngs.values()),
            "golden_acuracia_sintetica_vs_regime_do_gerador": synthetic_acc,
            "rasterizacao_igual_a_tools_render_chart": renderer_check,
        },
        "versions": versions,
        "hardware": hardware,
        "export_seconds": time.time() - _t_export,
    }

    def validate(staging: Path) -> None:
        """Relê o artefato como a API o faria: ONNX + PNGs do golden pelo caminho de serving, dentro da tolerância."""
        golden_file = load_golden(staging / "golden.json")
        sess = ort.InferenceSession(str(staging / "model.onnx"), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        for case in golden_file.cases:
            x = tex.pillow_preprocess(staging / golden_file.directory / case.file, 64, 64)[None]
            got = sess.run(None, {name: x})[0][0]
            assert np.allclose(got, case.output, atol=golden_file.tolerance.output_atol), case.file

    bundle = tex.ArtifactBundle(
        name=tei.IMAGE_MODEL_NAME,
        version=CNN_VERSION,
        onnx_bytes=onnx_bytes,
        manifest_fields=manifest_fields,
        golden=golden,
        results=results,
        golden_pngs=golden_pngs,
    )
    target = tex.write_artifact(ARTIFACT_ROOT, bundle, extra_validate=validate)
    reread = load_manifest(target / "manifest.json")
    print(f"artefato escrito em {target} | status {reread.status.value} | gate {reread.quality_gate.value:.4f} >= {reread.quality_gate.threshold:.2f} | {len(names)} golden")
    return target


# %%
if EXPORT_OK:
    cnn_artifact_dir = export_candlestick_cnn()
    print(json.dumps(json.loads((cnn_artifact_dir / "manifest.json").read_text())["serving_skew"], indent=2))
    print(sorted(p.name for p in cnn_artifact_dir.iterdir()))
else:
    print("exportação pulada")
if "part_times" in globals():
    part_times["96_export_cnn"] = time.time() - _t_export
print(f"Exportação da CNN concluída em {(time.time() - _t_export) / 60:.1f} min.", file=sys.stderr)
