"""Parte 95 - apendice: exporta ``credit-default-mlp`` e ``volatility-mlp`` para ``artifacts/<nome>/1.0.0/`` (PLANO 4.5).

Fora da area avaliada. Depende das partes 10 e 20 e dos modulos ``trinet_export`` (ao lado do notebook) e ``trinet-contract``,
importados dentro de ``try/except ImportError`` para o notebook continuar autocontido.
"""

# %% [markdown]
# # Apêndice · Exportação das redes tabulares para a API (`credit-default-mlp`, `volatility-mlp`)
#
# > **Fora da área avaliada.** Transforma os modelos treinados em **artefatos** consumidos por uma API de inferência
# > (`artifacts/<modelo>/1.0.0/`: `model.onnx`, `weights.npz`, `manifest.json`, `golden.json`, `results.json`).
# > O restante do notebook funciona sem esta parte.
#
# Para cada modelo, na ordem (qualquer falha **interrompe antes de escrever** o artefato):
#
# 1. **Clone de inferência em float32**: sem `Dropout`, pesos copiados; cada `Normalization` vira um `Rescaling` com constantes (o `Normalization` do Keras 3 guarda média/variância como
#    tensores soltos, e o conversor os exportaria como *entradas* extras do grafo).
# 2. **ONNX** com `Model.export(format="onnx", opset_version=17)` (por baixo, `tf2onnx`). Se a conversão falhar, o plano B documentado monta o ONNX com `onnx.helper` a partir dos pesos extraídos
#    (as duas rotas existem no código; a usada fica registrada em `results.json`). Nomes e formas dos tensores são **lidos do ONNX Runtime**.
# 3. **Runtime NumPy de reserva** (`weights.npz` + lista de camadas do contrato; o da volatilidade termina em *desnormalizar → exp*), verificado contra o Keras.
# 4. **Registros crus de referência (golden)**: ≥ 200 registros com `features` (o vetor que entra no grafo), `output` (saída do Keras em CPU) e rótulo real. A API reproduz o pré-processamento e o resultado de cada um.
# 5. **Paridade**: ONNX Runtime × Keras em **todos** os casos golden e em validação + teste inteiros; runtime NumPy também.
# 6. **Quality gate**: métrica de validação **servida pelo ONNX** contra o limiar (validação do Keras ± margem); se reprovar, **nada é escrito**.
# 7. `manifest.json` + `golden.json` + `results.json`, validados com `trinet-contract`, relidos e conferidos por SHA-256 (escrita atômica).

# %%
import hashlib
import json
import subprocess
import tempfile

from sklearn.metrics import precision_score, recall_score

_t_export = time.time()
try:  # so o apendice depende destes modulos
    import onnx
    import onnxruntime as ort
    from trinet_contract import (
        CategoricalField,
        ClassificationOutput,
        NumericField,
        OneHotFeature,
        PassthroughFeature,
        RatioFeature,
        RegressionOutput,
        TabularInput,
        load_golden,
        load_manifest,
    )

    import trinet_export as tex

    EXPORT_OK = True
except ImportError as _exc:
    EXPORT_OK = False
    print(f"Apêndice de exportação desativado (dependência ausente: {_exc}). O restante do notebook não é afetado.")


def _find_repo_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "contract").is_dir() and (candidate / "tools").is_dir():
            return candidate
    return None


REPO_ROOT = _find_repo_root(Path.cwd().resolve())
ARTIFACT_ROOT = Path(os.environ.get("TRINET_ARTIFACTS_DIR") or (REPO_ROOT / "artifacts" if REPO_ROOT else "artifacts")).resolve()
EXPORT_VERSION = "1.0.0"
CREDIT_ACC_MARGIN = 0.03  # quality gate: acuracia de validacao servida >= acuracia de validacao do Keras - 3 pp
CREDIT_PRAUC_MARGIN = 0.01  # gate interno adicional (PR-AUC)
VOL_RMSE_MARGIN = 0.05  # quality gate: RMSE de validacao servido <= RMSE de validacao do Keras x 1,05
VOL_QLIKE_MARGIN = 0.02
N_GOLDEN = 200
API_MAX_BATCH = 256  # limite de instancias por requisicao da API: os casos golden cabem numa unica chamada
print(f"exportação ativa: {EXPORT_OK} | artefatos em: {ARTIFACT_ROOT}")

# %% [markdown]
# ### Funções auxiliares (clone, ONNX com plano B, verificações)

# %%
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _kaggle_csv() -> Path:
    return next((DATA_DIR / "kagglehub").rglob("all_stocks_5yr.csv"))


def build_runtimes(model, input_name: str, n_in: int, n_out: int, probe: np.ndarray) -> dict:
    """Clone de inferencia, ONNX (Keras/tf2onnx ou plano B), pesos NumPy; verifica cada passo contra o Keras em ``probe``."""
    infer = tex.rebuild_inference(model, input_name)
    keras_out = model.predict(probe, batch_size=8192, verbose=0)
    clone_diff = float(np.abs(infer.predict(probe, batch_size=8192, verbose=0) - keras_out).max())
    arrays, layer_specs = tex.extract_mlp(model)
    numpy_diff = float(np.abs(tex.numpy_mlp_forward(arrays, layer_specs, probe) - keras_out).max())
    method, error = "keras.export(format='onnx') [tf2onnx]", None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        try:
            tex.export_onnx(infer, path, input_name, opset=tex.ONNX_OPSET)
            onnx_bytes = path.read_bytes()
            info = tex.inspect_onnx(path)
        except Exception as exc:  # noqa: BLE001 - plano B documentado
            error = f"{type(exc).__name__}: {exc}"[:300]
            method = "onnx.helper a partir dos pesos (plano B)"
            onnx_bytes = tex.onnx_from_weights(layer_specs, arrays, n_in, n_out, input_name=input_name)
            path.write_bytes(onnx_bytes)
            info = tex.inspect_onnx(path)
    onnx_diff = float(np.abs(tex.ort_predict(onnx_bytes, probe).reshape(keras_out.shape) - keras_out).max())
    return {"infer": infer, "arrays": arrays, "layers": layer_specs, "npz": tex.npz_bytes(arrays), "onnx": onnx_bytes, "info": info,
            "method": method, "error": error, "diffs": {"clone": clone_diff, "numpy": numpy_diff, "onnx": onnx_diff}}  # fmt: skip


def check_spec_matches_notebook(spec, records: list[dict], notebook_features: np.ndarray, names: list[str]) -> float:
    """O pre-processamento do MANIFESTO (3 primitivas) reproduz o vetor de features do notebook? Devolve o maior |diff|."""
    assert spec.feature_names() == list(names), "ordem/nomes das colunas do manifesto diferem do notebook"
    got = tex.records_to_features(spec, records)
    diff = float(np.abs(got.astype("float64") - notebook_features.astype("float64")).max())
    rel = float((np.abs(got - notebook_features) / np.maximum(np.abs(notebook_features), 1.0)).max())
    assert rel <= 1e-6, f"features do manifesto divergem do notebook (rel {rel:.2e})"
    return diff


def training_info(dataset: str, data_sha: str) -> dict:
    return {"dataset": dataset, "data_sha256": data_sha, "seed": SEED, "python": sys.version.split()[0], "tensorflow": tf.__version__, "keras": keras.__version__, "hardware": HW[:200]}  # fmt: skip


def validate_staging(staging: Path, features: np.ndarray, expected: np.ndarray, atol: float, top1: bool) -> None:
    """Rele o artefato escrito e repete as verificacoes de paridade com o ONNX e o NumPy gravados em disco."""
    manifest = load_manifest(staging / "manifest.json")
    out = tex.ort_predict(staging / "model.onnx", features).reshape(expected.shape)
    assert np.allclose(out, expected, atol=atol, rtol=1e-5), f"ONNX gravado diverge do golden (max {np.abs(out - expected).max():.2e})"
    npz = np.load(staging / "weights.npz", allow_pickle=False)
    layer_specs = [dict(layer.model_dump(mode="json")) for layer in manifest.runtimes[1].layers]
    np_out = tex.numpy_mlp_forward({k: npz[k] for k in npz.files}, layer_specs, features).reshape(expected.shape)
    assert np.allclose(np_out, expected, atol=atol, rtol=1e-5), f"NumPy gravado diverge do golden (max {np.abs(np_out - expected).max():.2e})"
    if top1:
        assert (out.argmax(axis=1) == expected.argmax(axis=1)).all(), "top-1 do ONNX difere do Keras"


# %% [markdown]
# ## Modelo 1 · `credit-default-mlp`
#
# Campos crus da API: `limit_bal` (≥ 1), `education` (`graduate`/`university`/`high_school`/`other`), `age`, `pay_1..6` (−2..8), `bill_amt1..6`, `pay_amt1..6` (≥ 0). **`SEX` e `MARRIAGE` não existem na API.**
# Pré-processamento fora do grafo: *passthrough* das 20 numéricas, *one-hot* de `education` e a razão `bill_amt1 / limit_bal` (`bill_utilisation`; o denominador tem mínimo 1, então a API nunca divide por zero).
# `Normalization` fica **dentro** do grafo. A decisão da API é o `argmax` das probabilidades; o limiar por custo (seção 1.8) é uma recomendação extra, guardada em `results.json`.
# **Gate:** acurácia de validação servida ≥ acurácia do Keras − 3 pp (é a métrica que o CI da API consegue re-pontuar) e, internamente, PR-AUC ≥ PR-AUC do Keras − 0,01.

# %%
CREDIT_LABELS = ("no_default", "default")


def credit_spec():
    fields = [
        NumericField(name="limit_bal", description="limite de credito (NT$); >= 1 (denominador da razao)", minimum=1.0, maximum=1e7),
        NumericField(name="age", description="idade em anos", minimum=18.0, maximum=120.0),
        *[NumericField(name=f"pay_{m}", description=f"situacao do pagamento {m} meses atras (set->abr/2005): -2/-1/0 = sem atraso, 1..8 = meses de atraso", minimum=-2.0, maximum=8.0) for m in range(1, 7)],
        *[NumericField(name=f"bill_amt{m}", description=f"valor da fatura {m} (NT$); pode ser negativo (credito do cliente)", minimum=-1e7, maximum=1e7) for m in range(1, 7)],
        *[NumericField(name=f"pay_amt{m}", description=f"valor pago {m} (NT$)", minimum=0.0, maximum=1e7) for m in range(1, 7)],
        CategoricalField(name="education", description="escolaridade (codigos 0, 4, 5 e 6 do dataset = other)", vocabulary=EDU_VOCAB),
    ]  # fmt: skip
    features = [*[PassthroughFeature(source=n) for n in CREDIT_NUMERIC], OneHotFeature(source="education"), RatioFeature(name="bill_utilisation", numerator="bill_amt1", denominator="limit_bal")]
    return TabularInput(fields=tuple(fields), features=tuple(features))


def export_credit() -> Path:
    t0 = time.time()
    spec = credit_spec()
    # limites de entrada da API devem cobrir todo o treino/val/teste/novas instancias
    allx = np.concatenate([X_train, X_val, X_test, X_new])
    for f in spec.fields:
        if isinstance(f, NumericField):
            col = allx[:, CREDIT_FEATURES.index(f.name)]
            assert col.min() >= f.minimum and col.max() <= f.maximum, f"{f.name} fora dos limites da API"

    # 1-3) clone, ONNX, NumPy
    probe = np.concatenate([X_val, X_test])
    rt = build_runtimes(credit_model, "features", N_CREDIT, 2, probe)
    print(f"ONNX por: {rt['method']}" + (f" | erro do Keras/tf2onnx: {rt['error']}" if rt["error"] else ""))
    print(f"ONNX: opset {rt['info']['opset']}, {rt['info']['bytes'] / 1e3:.1f} kB, ops {rt['info']['ops']} | entrada {rt['info']['input']} | saída {rt['info']['output']}")
    print(f"diferença máxima vs Keras (val+teste = {len(probe):,} linhas): clone {rt['diffs']['clone']:.2e} | ONNX {rt['diffs']['onnx']:.2e} | NumPy {rt['diffs']['numpy']:.2e}")
    assert max(rt["diffs"].values()) < 1e-4

    # 4) golden: 200 linhas do teste + 20 novas instancias (rotulos reais)
    pick = np.random.default_rng(SEED).choice(len(idx_test), size=N_GOLDEN, replace=False)
    g_frame = pd.concat([credit_df.loc[np.asarray(idx_test)[pick]], credit_new])
    g_records = credit_to_api(g_frame)
    assert len(g_records) <= API_MAX_BATCH, f"{len(g_records)} casos golden excedem o lote máximo da API ({API_MAX_BATCH})"
    g_features = credit_feature_matrix(g_frame)
    manifest_vs_notebook = check_spec_matches_notebook(spec, g_records, g_features, CREDIT_FEATURES)
    g_out = credit_model.predict(g_features, batch_size=8192, verbose=0).astype("float64")
    g_labels = [CREDIT_LABELS[int(v)] for v in g_frame["default"]]
    golden = tex.tabular_golden(model="credit-default-mlp", version=EXPORT_VERSION, records=g_records, features=g_features, outputs=g_out, labels=g_labels, output_atol=1e-4)
    onnx_golden_diff = float(np.abs(tex.ort_predict(rt["onnx"], g_features) - g_out).max())
    assert onnx_golden_diff < 1e-4, onnx_golden_diff

    # 5-6) metricas servidas pelo ONNX e quality gate
    served = {"validation": tex.ort_predict(rt["onnx"], X_val)[:, 1], "test": tex.ort_predict(rt["onnx"], X_test)[:, 1]}
    keras_p = {"validation": credit_model.predict(X_val, batch_size=8192, verbose=0)[:, 1], "test": credit_model.predict(X_test, batch_size=8192, verbose=0)[:, 1]}
    ys = {"validation": y_val, "test": y_test}
    metrics = {}
    for split_name in ("validation", "test"):
        p, y = served[split_name], ys[split_name]
        pred = (p >= 0.5).astype(int)
        metrics[split_name] = {
            "accuracy": float((pred == y).mean()), "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)), "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "ks": ks_statistic(y, p), "brier": float(brier_score_loss(y, p)),
        }  # fmt: skip
    keras_val_acc = float(((keras_p["validation"] >= 0.5).astype(int) == y_val).mean())
    keras_val_pr = float(average_precision_score(y_val, keras_p["validation"]))
    gate = tex.check_gate("accuracy", "validation", metrics["validation"]["accuracy"], float(np.floor((keras_val_acc - CREDIT_ACC_MARGIN) * 100) / 100), True)
    extra_gate = tex.check_gate("pr_auc", "validation", metrics["validation"]["pr_auc"], float(np.floor((keras_val_pr - CREDIT_PRAUC_MARGIN) * 1000) / 1000), True)
    print(f"gate (acurácia val. servida {gate['value']:.4f} >= {gate['threshold']:.2f}) e gate interno (PR-AUC val. {extra_gate['value']:.4f} >= {extra_gate['threshold']:.3f}): OK")

    all_served = np.concatenate([served["validation"], served["test"]])
    all_keras = np.concatenate([keras_p["validation"], keras_p["test"]])
    skew = {"samples": len(all_served), "max_abs_diff": float(np.abs(all_served - all_keras).max()), "top1_agreement": float(((all_served >= 0.5) == (all_keras >= 0.5)).mean())}
    print(f"serving skew (ONNX × Keras): {skew}")

    in_info, out_info = rt["info"]["input"], rt["info"]["output"]
    manifest_fields = {
        "name": "credit-default-mlp", "version": EXPORT_VERSION, "status": "candidate", "task": "classification",
        "description": ("Classificador de inadimplencia de cartao de credito (UCI #350, Taiwan 2005): MLP Normalization -> Dense(relu) -> Dropout -> Dense(2, softmax), "
                        "treinado com class_weight. SEX e MARRIAGE nao sao features. Projeto educacional: nao e um modelo de decisao de credito real."),
        "created_at": tex.utc_now(), "git_sha": None, "input": spec, "output": ClassificationOutput(labels=CREDIT_LABELS),
        "inputs": [{"name": in_info["name"], "shape": in_info["shape"]}], "outputs": [{"name": out_info["name"], "shape": out_info["shape"]}],
        "metrics": metrics, "quality_gate": gate, "serving_skew": skew,
        "training": training_info("Default of Credit Card Clients (UCI #350, CC BY 4.0, DOI 10.24432/C55S3H)", credit_data_sha256), "onnx_opset": rt["info"]["opset"],
    }  # fmt: skip
    ab = (
        credit_abl.groupby(["class_weight", "depth", "width"])[["val_pr_auc", "val_roc_auc", "val_f1_05", "val_f1_best", "params", "epochs"]].mean().reset_index().round(5).to_dict(orient="records")
    )
    results = {
        "model": "credit-default-mlp", "version": EXPORT_VERSION, "status": "candidate",
        "disclaimer": "Projeto educacional (T2 CIC407). Nao e um modelo de decisao de credito real; custos do limiar sao ilustrativos.",
        "architecture": {"width": CREDIT_WIDTH, "depth": CREDIT_DEPTH, "dropout": 0.2, "class_weight": use_class_weight, "features": CREDIT_FEATURES},
        "decision": {"argmax_threshold": 0.5, "cost_threshold_recommended": credit_threshold, "cost_fn_over_fp": COST_FN / COST_FP, "note": "limiar por custo escolhido na validacao (premissa ilustrativa)"},
        "metrics": metrics, "notebook_test_table": credit_results["test"], "extra_gate": extra_gate, "fairness_by_sex_test": fair.reset_index().to_dict(orient="records"),
        "ablation_mean_over_3_seeds": ab, "onnx": {"method": rt["method"], "keras_export_error": rt["error"], **rt["info"]}, "parity_max_abs_diff": rt["diffs"],
        "manifest_vs_notebook_features_max_abs_diff": manifest_vs_notebook, "golden_cases": len(g_records),
        "timings_seconds": {"train": credit_train_seconds, "export": time.time() - t0, "notebook_part_10": part_times.get("10_credit")},
        "versions": {"python": sys.version.split()[0], "tensorflow": tf.__version__, "keras": keras.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
        "data": {"csv_sha256": credit_data_sha256, "train": len(y_train), "val": len(y_val), "test": len(y_test)},
    }  # fmt: skip
    bundle = tex.ArtifactBundle(
        name="credit-default-mlp", version=EXPORT_VERSION, onnx_bytes=rt["onnx"], manifest_fields=manifest_fields, golden=golden, results=results,
        npz=rt["npz"], numpy_layers=rt["layers"],
    )  # fmt: skip
    return tex.write_artifact(ARTIFACT_ROOT, bundle, extra_validate=lambda staging: validate_staging(staging, g_features, g_out, 1e-4, top1=True))


# %%
if EXPORT_OK:
    credit_artifact = export_credit()
    print("artefato escrito em:", credit_artifact)
    print(sorted(p.name for p in credit_artifact.iterdir()))

# %% [markdown]
# ## Modelo 2 · `volatility-mlp`
#
# A API recebe **features já calculadas** (a engenharia de séries temporais é de quem chama; a escolha está documentada num ADR). Campos crus: `ret_1d..5d`, `rv_5/10/20/60`, `pk_5/20`, `range_5`, `volume_ratio`, `log_dollar_volume`,
# `gap`, `mom_20/60`, `drawdown_60`, `mkt_rv_20` e `dow` (categórica). *Passthrough* das 19 numéricas, *one-hot* de `dow` e razão `rv_5 / rv_20` (`rv_20 ≥ 1e-6`).
# O grafo é `Normalization → Dense… → Dense(1) → Normalization(invert=True) → exp` e **devolve a volatilidade anualizada em fração**; o runtime NumPy de reserva termina em *desnormalizar → exp* (`ExpLayer`).
#
# **Tolerância relativa e `output_atol`.** O contrato só tem tolerância absoluta (`output_atol`), então escolhemos um valor **justificado em unidades naturais**: as volatilidades servidas ficam em ≈ 0,05 a 1,0 (5% a 100% a.a.);
# o erro de float32 do `exp` é da ordem de 1e-7 em termos relativos. Usamos `max(1e-4, 20 × erro observado)`, o que dá erro relativo ≲ 0,05% na mediana (≈ 0,16) e a API ainda soma `rtol = 1e-5` por cima.
# **Gate:** RMSE de validação (natural) servido pelo ONNX ≤ RMSE do Keras × 1,05 (e QLIKE ≤ QLIKE do Keras + 0,02, internamente).

# %%
def volatility_spec():
    descr = {"ret_1d": "log-retorno de hoje", "rv_5": "vol. realizada anualizada, 5 pregoes", "rv_20": "vol. realizada anualizada, 20 pregoes (>= 1e-6: denominador da razao)",
             "pk_5": "vol. de Parkinson anualizada, 5 pregoes", "range_5": "media de (maxima - minima) / fechamento, 5 pregoes", "volume_ratio": "volume / media de 20 dias",
             "log_dollar_volume": "ln(fechamento x volume)", "gap": "abertura / fechamento anterior - 1", "mom_20": "retorno acumulado em 20 pregoes",
             "drawdown_60": "fechamento / maximo de 60 pregoes - 1 (<= 0)", "mkt_rv_20": "media de rv_20 de todos os tickers no dia (proxy de VIX)"}  # fmt: skip
    fields = [NumericField(name=n, description=descr.get(n, ""), minimum=VOL_BOUNDS[n][0], maximum=VOL_BOUNDS[n][1]) for n in VOL_NUMERIC]
    fields.append(CategoricalField(name="dow", description="dia da semana do fechamento", vocabulary=DOW_VOCAB))
    features = [*[PassthroughFeature(source=n) for n in VOL_NUMERIC], OneHotFeature(source="dow"), RatioFeature(name="rv_ratio", numerator="rv_5", denominator="rv_20")]
    return TabularInput(fields=tuple(fields), features=tuple(features))


def vol_records(frame: pd.DataFrame) -> list[dict]:
    out = frame[VOL_NUMERIC].astype(float).copy()
    out["dow"] = frame["dow"].astype(str)
    return out.to_dict(orient="records")


def export_volatility() -> Path:
    t0 = time.time()
    spec = volatility_spec()

    probe = np.concatenate([vX_val, vX_test])
    rt = build_runtimes(vol_infer, "features", N_VOL, 1, probe)
    print(f"ONNX por: {rt['method']}" + (f" | erro do Keras/tf2onnx: {rt['error']}" if rt["error"] else ""))
    print(f"ONNX: opset {rt['info']['opset']}, {rt['info']['bytes'] / 1e3:.1f} kB, ops {rt['info']['ops']} | entrada {rt['info']['input']} | saída {rt['info']['output']}")
    print(f"diferença máxima vs Keras (val+teste = {len(probe):,} linhas), em fração a.a.: clone {rt['diffs']['clone']:.2e} | ONNX {rt['diffs']['onnx']:.2e} | NumPy {rt['diffs']['numpy']:.2e}")
    assert max(rt["diffs"].values()) < 1e-4

    # golden: 200 amostras do TESTE (sorteio uniforme no tempo) + 10 tickers x 5 dias de novas instancias (sem rotulo).
    # O total (250) fica abaixo do limite de 256 instancias por requisicao da API (o teste de replay HTTP envia todos os casos de uma vez).
    pick = np.random.default_rng(SEED).choice(len(vX_test), size=N_GOLDEN, replace=False)
    t_frame = parts["test"].iloc[np.sort(pick)]
    new_golden = vol_new[vol_new["ticker"].isin(sorted(vol_new["ticker"].unique())[:10])]
    g_frame = pd.concat([t_frame, new_golden[new_golden.columns.intersection(t_frame.columns)]])
    g_labels = [float(v) for v in t_frame["y"]] + [None] * len(new_golden)
    g_records = vol_records(g_frame)
    assert len(g_records) <= API_MAX_BATCH, f"{len(g_records)} casos golden excedem o lote máximo da API ({API_MAX_BATCH})"
    g_features = vol_feature_matrix(g_frame)
    manifest_vs_notebook = check_spec_matches_notebook(spec, g_records, g_features, VOL_FEATURES)
    g_out = vol_infer.predict(g_features, batch_size=8192, verbose=0).astype("float64")
    onnx_golden_diff = float(np.abs(tex.ort_predict(rt["onnx"], g_features) - g_out).max())
    numpy_golden_diff = float(np.abs(tex.numpy_mlp_forward(rt["arrays"], rt["layers"], g_features) - g_out).max())
    atol = float(max(1e-4, tex.round_up(20 * max(onnx_golden_diff, numpy_golden_diff), 1e-5)))
    print(f"golden: {len(g_records)} casos | diff máx. ONNX {onnx_golden_diff:.2e}, NumPy {numpy_golden_diff:.2e} | output_atol = {atol:g} (≈ {atol / float(np.median(g_out)):.3%} da mediana {float(np.median(g_out)):.3f})")
    golden = tex.tabular_golden(model="volatility-mlp", version=EXPORT_VERSION, records=g_records, features=g_features, outputs=g_out, labels=g_labels, output_atol=atol)

    served = {"validation": tex.ort_predict(rt["onnx"], vX_val).reshape(-1).astype("float64"), "test": tex.ort_predict(rt["onnx"], vX_test).reshape(-1).astype("float64")}
    ys = {"validation": vy_val, "test": vy_test}
    metrics = {}
    for split_name in ("validation", "test"):
        sc = vol_scores(ys[split_name], served[split_name])
        metrics[split_name] = {"rmse": sc["RMSE"], "mae": sc["MAE"], "mae_over_mean": sc["MAE/média"], "rmse_over_mean": sc["RMSE/média"], "qlike": sc["QLIKE"], "r2_log": sc["R² (log)"], "bias": sc["viés"]}  # fmt: skip
    keras_val = vol_scores(vy_val, vol_infer.predict(vX_val, batch_size=8192, verbose=0).reshape(-1))
    gate = tex.check_gate("rmse", "validation", metrics["validation"]["rmse"], tex.round_up(keras_val["RMSE"] * (1 + VOL_RMSE_MARGIN), 1e-4), False)
    extra_gate = tex.check_gate("qlike", "validation", metrics["validation"]["qlike"], tex.round_up(keras_val["QLIKE"] + VOL_QLIKE_MARGIN, 1e-3), False)
    print(f"gate (RMSE val. servido {gate['value']:.4f} <= {gate['threshold']:.4f}) e gate interno (QLIKE val. {extra_gate['value']:.4f} <= {extra_gate['threshold']:.3f}): OK")

    keras_all = vol_infer.predict(probe, batch_size=8192, verbose=0).reshape(-1)
    skew = {"samples": len(keras_all), "max_abs_diff": float(np.abs(np.concatenate([served["validation"], served["test"]]) - keras_all).max())}
    print(f"serving skew (ONNX × Keras), fração a.a.: {skew}")

    in_info, out_info = rt["info"]["input"], rt["info"]["output"]
    manifest_fields = {
        "name": "volatility-mlp", "version": EXPORT_VERSION, "status": "candidate", "task": "regression",
        "description": ("Regressor da volatilidade realizada dos proximos 5 pregoes de acoes do S&P 500 (anualizada, em fracao): MLP sobre ln(y) padronizado, "
                        "com clone de inferencia Normalization(invert) -> exp. Entrada: features ja calculadas no fechamento de t. Projeto educacional: nao e recomendacao de investimento."),
        "created_at": tex.utc_now(), "git_sha": None, "input": spec, "output": RegressionOutput(target="realized_volatility_5d", unit="volatilidade anualizada (fração)"),
        "inputs": [{"name": in_info["name"], "shape": in_info["shape"]}], "outputs": [{"name": out_info["name"], "shape": out_info["shape"]}],
        "metrics": metrics, "quality_gate": gate, "serving_skew": skew,
        "training": training_info("S&P 500 5 anos (Kaggle camnugent/sandp500, all_stocks_5yr.csv; licenca nao verificada, dados nao redistribuidos)", sha256_of(_kaggle_csv())), "onnx_opset": rt["info"]["opset"],
    }  # fmt: skip
    ab = vol_abl.groupby(["loss", "depth", "width"])[["val_QLIKE", "val_RMSE", "val_MAE", "val_R2log", "params", "epochs"]].mean().reset_index().round(5).to_dict(orient="records")
    results = {
        "model": "volatility-mlp", "version": EXPORT_VERSION, "status": "candidate",
        "disclaimer": "Projeto educacional (T2 CIC407). Nao e recomendacao de investimento.",
        "architecture": {"width": VOL_WIDTH, "depth": VOL_DEPTH, "loss": VOL_LOSS, "huber_delta_z": HUBER_DELTA, "target_mu_ln": v_mu, "target_sigma_ln": v_sigma, "features": VOL_FEATURES},
        "metrics": metrics, "notebook_test_table": vol_results["test"], "baselines_test": {k: v for k, v in vol_results["test"].items() if k != "MLP"}, "extra_gate": extra_gate,
        "error_by_month_test": vol_results["by_month"], "error_by_regime_test": vol_results["by_regime"], "tercile_edges": vol_results["tercile_edges"], "tercile_accuracy_test": vol_results["tercile_acc"],
        "split": vol_results["split"], "sizes": vol_results["sizes"], "ablation_mean_over_3_seeds": ab,
        "onnx": {"method": rt["method"], "keras_export_error": rt["error"], **rt["info"]}, "parity_max_abs_diff": rt["diffs"],
        "golden_tolerance": {"output_atol": atol, "note": "unidades naturais (fracao a.a.); a API soma rtol=1e-5"}, "manifest_vs_notebook_features_max_abs_diff": manifest_vs_notebook,
        "golden_cases": len(g_records), "timings_seconds": {"train": vol_train_seconds, "export": time.time() - t0, "notebook_part_20": part_times.get("20_volatility")},
        "versions": {"python": sys.version.split()[0], "tensorflow": tf.__version__, "keras": keras.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
        "data": {"sp500_rows": int(sp500_report["linhas_limpas"]), "cleaning": sp500_report, "samples": vol_results["sizes"]},
    }  # fmt: skip
    bundle = tex.ArtifactBundle(
        name="volatility-mlp", version=EXPORT_VERSION, onnx_bytes=rt["onnx"], manifest_fields=manifest_fields, golden=golden, results=results,
        npz=rt["npz"], numpy_layers=rt["layers"],
    )  # fmt: skip
    return tex.write_artifact(ARTIFACT_ROOT, bundle, extra_validate=lambda staging: validate_staging(staging, g_features, g_out, atol, top1=False))


# %%
if EXPORT_OK:
    volatility_artifact = export_volatility()
    print("artefato escrito em:", volatility_artifact)
    print(sorted(p.name for p in volatility_artifact.iterdir()))

# %% [markdown]
# ## Verificação final dos artefatos
#
# Relê os dois diretórios com o contrato (`load_manifest`, `load_golden`), confere SHA-256, e, se o repositório estiver disponível, roda `tools/check_artifacts.py` (o mesmo gate do CI).

# %%
if EXPORT_OK:
    rows = []
    for name in ("credit-default-mlp", "volatility-mlp"):
        d = ARTIFACT_ROOT / name / EXPORT_VERSION
        m, g = load_manifest(d / "manifest.json"), load_golden(d / "golden.json")
        rows.append({"modelo": name, "status": m.status.value, "runtimes": [r.kind for r in m.runtimes], "casos golden": len(g.cases), "gate": f"{m.quality_gate.metric} {m.quality_gate.value:.4f} vs {m.quality_gate.threshold:.4f}",
                     "gate ok": m.quality_gate.passed, "tamanho (kB)": round(sum(p.stat().st_size for p in d.iterdir() if p.is_file()) / 1e3, 1)})  # fmt: skip
    show(pd.DataFrame(rows))
    if REPO_ROOT is not None and (REPO_ROOT / "tools" / "check_artifacts.py").is_file():
        proc = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "check_artifacts.py"), "--root", str(ARTIFACT_ROOT)], capture_output=True, text=True, check=False)  # noqa: S603
        print((proc.stdout + proc.stderr).strip()[-1500:])
        print("tools/check_artifacts.py: código de saída", proc.returncode, "(0 = tudo certo; a CNN pode ainda não ter sido exportada)")
    part_times["95_export"] = time.time() - _t_export
    print(f"Exportação concluída em {part_times['95_export'] / 60:.1f} min.")
