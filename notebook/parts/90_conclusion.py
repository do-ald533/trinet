"""Parte 90 - conclusoes: tabelas lado a lado, o que funcionou, limitacoes, etica e fechamento (versoes, hardware, tempo).

Usa credit_test_table, vol_test_table, credit_results/vol_results e, se a parte 30 (CNN) tiver rodado, cnn_test_acc,
cnn_comparison, cnn_vs_har_pvalue, vol_baseline_acc_*.
"""

# %% [markdown]
# # 4 · Conclusões
#
# Três redes, construídas camada a camada em `tf.keras.layers`, para três riscos financeiros. As tabelas abaixo resumem o desempenho **no teste** (tocado uma única vez em cada rede) **contra os baselines**,
# que são o que dá sentido às métricas: um número isolado não diz se o modelo vale a pena.

# %% [markdown]
# ## 4.1 Rede 1 · Risco de crédito (classificação)

# %%
credit_cols = ["acurácia", "precisão (1)", "revocação (1)", "F1 (1)", "ROC-AUC", "PR-AUC", "KS", "Brier", "custo/cliente"]
show(credit_test_table[credit_cols])
_mlp, _lr, _maj = credit_test_table.loc["MLP (0,5)"], credit_test_table.loc["regressão logística (0,5)"], credit_test_table.loc["maioria (sempre adimplente)"]
_mlp_c, _lr_c = credit_test_table.loc["MLP (limiar por custo)"], credit_test_table.loc["regressão logística (custo)"]
print(f"MLP × regressão logística (teste): ROC-AUC {_mlp['ROC-AUC']:.3f} × {_lr['ROC-AUC']:.3f} | PR-AUC {_mlp['PR-AUC']:.3f} × {_lr['PR-AUC']:.3f} | KS {_mlp['KS']:.3f} × {_lr['KS']:.3f} | Brier {_mlp['Brier']:.3f} × {_lr['Brier']:.3f}")
print(f"Custo por cliente (FN = {COST_FN:g} × FP, premissa ilustrativa): aprovar todos {COST_FN * float(y_test.mean()):.3f} | regressão logística {_lr_c['custo/cliente']:.3f} | MLP {_mlp_c['custo/cliente']:.3f} (limiar {credit_threshold:.2f}, escolhido na validação)")
print(f"Acurácia da classe majoritária: {_maj['acurácia']:.3f} (piso enganoso: revocação 0) | MLP: {_mlp['acurácia']:.3f}")

# %% [markdown]
# ## 4.2 Rede 2 · Risco de mercado, via números (regressão da volatilidade)

# %%
vol_cols = ["RMSE", "MAE", "MAE/média", "RMSE/média", "QLIKE", "R² (log)", "viés"]
show(vol_test_table[vol_cols])
_v, _h, _p, _m0 = (vol_test_table.loc[k] for k in ("MLP", "HAR linear", "persistência (rv_5)", "média do treino"))
print(f"MLP × HAR (teste, unidades naturais): RMSE {_v['RMSE']:.4f} × {_h['RMSE']:.4f} ({_v['RMSE'] / _h['RMSE'] - 1:+.1%}) | MAE {_v['MAE']:.4f} × {_h['MAE']:.4f} ({_v['MAE'] / _h['MAE'] - 1:+.1%}) | QLIKE {_v['QLIKE']:.3f} × {_h['QLIKE']:.3f}")
print(f"MLP × persistência: RMSE {_v['RMSE'] / _p['RMSE'] - 1:+.1%}, MAE {_v['MAE'] / _p['MAE'] - 1:+.1%} | MLP × média do treino: RMSE {_v['RMSE'] / _m0['RMSE'] - 1:+.1%}, MAE {_v['MAE'] / _m0['MAE'] - 1:+.1%}")
print(f"Margem de erro relativa do MLP: MAE/média = {_v['MAE/média']:.1%}, RMSE/média = {_v['RMSE/média']:.1%}")

# %% [markdown]
# ## 4.3 Rede 3 · Risco de mercado, via imagem (CNN) e a pergunta "números × imagem"
#
# Mesmo alvo (regime de volatilidade dos próximos 5 pregões: `baixa`, `media`, `alta`, tercis do treino), mesmo período de teste. A CNN vê **só o desenho** do gráfico;
# a Rede 2 vê as **25 features numéricas**. A comparação justa usa **as mesmas linhas** de teste (a CNN treina e testa numa subamostra com passo, ver parte 3).
# A célula só roda se a parte 3 tiver sido executada nesta sessão (variáveis `cnn_*`); caso contrário, apenas avisa.

# %%
if "cnn_test_acc" in globals():
    from scipy.stats import binomtest

    _tidx = parts["test"].index.to_numpy()
    _same = np.isin(_tidx, cnn_idx_test)  # linhas de teste que a CNN também avaliou (ordem de `sp` preservada nas duas listas)
    _true = regime_labels(vy_test[_same], vol_tercile_edges)
    assert _same.sum() == len(img_y_test) and np.array_equal(_true, img_y_test), "rótulos da Rede 2 e da CNN divergem nas mesmas linhas"
    _preds = {
        "CNN (gráficos)": cnn_pred_test,
        "Rede 2: MLP (números)": regime_labels(yhat["MLP"][_same], vol_tercile_edges),
        "HAR linear (números)": regime_labels(yhat["HAR linear"][_same], vol_tercile_edges),
        "persistência rv_5": regime_labels(yhat["persistência (rv_5)"][_same], vol_tercile_edges),
    }

    def _rec(pred, k):
        return float(((pred == k) & (_true == k)).sum() / max(1, (_true == k).sum()))

    vol_vs_cnn = pd.DataFrame(
        {
            n: {
                "acurácia": float((pr == _true).mean()),
                "acurácia balanceada": float(np.mean([_rec(pr, k) for k in range(3)])),
                **{f"revocação {REGIME_NAMES[k]}": _rec(pr, k) for k in range(3)},
            }
            for n, pr in _preds.items()
        }
    ).T
    vol_vs_cnn.loc["classe majoritária"] = [
        float(np.bincount(_true).max() / len(_true)),
        1 / 3,
        *[float(np.bincount(_true).argmax() == k) for k in range(3)],
    ]
    show(vol_vs_cnn)
    print(f"{len(_true):,} linhas de teste em comum | acaso = 33,3% | CNN: validação {cnn_val_acc:.1%}, teste {cnn_test_acc:.1%} | {cnn_epoch_seconds:.0f} s/época")
    print(f"Custo de treino: CNN {cnn_epoch_seconds * len(cnn_history['loss']) / 60:.1f} min ({len(cnn_history['loss'])} épocas) × MLP da Rede 2 {vol_train_seconds / 60:.1f} min" if "vol_train_seconds" in globals() else "")
    _b = int(((_preds["CNN (gráficos)"] == _true) & (_preds["Rede 2: MLP (números)"] != _true)).sum())
    _c = int(((_preds["CNN (gráficos)"] != _true) & (_preds["Rede 2: MLP (números)"] == _true)).sum())
    cnn_vs_mlp_pvalue = float(binomtest(_b, _b + _c, 0.5).pvalue) if _b + _c else 1.0
    print(f"CNN × MLP nas mesmas linhas: só a CNN acerta {_b}, só o MLP acerta {_c} (McNemar exato, p = {cnn_vs_mlp_pvalue:.3g})")
    if "cnn_vs_har_pvalue" in globals():
        print(f"CNN × HAR nas mesmas imagens: p = {cnn_vs_har_pvalue:.3g} (célula 3.9)")
    if "cnn_direction_acc" in globals():
        print(f"Sanidade, direção (sobe/desce) pela CNN: {cnn_direction_acc.get('test', float('nan')):.1%} no teste | sempre a classe majoritária: {cnn_direction_acc.get('test_always_prior_class', float('nan')):.1%} | acaso 50%")
    _acc = {n: float((pr == _true).mean()) for n, pr in _preds.items()}
    _bal = {n: float(vol_vs_cnn.loc[n, "acurácia balanceada"]) for n in _preds}
    _best_num = max(_acc["Rede 2: MLP (números)"], _acc["HAR linear (números)"])
    _gap = _acc["CNN (gráficos)"] - _best_num
    _bal_gap = _bal["CNN (gráficos)"] - max(_bal["Rede 2: MLP (números)"], _bal["HAR linear (números)"])
    if _gap > 0.01 and cnn_vs_mlp_pvalue < 0.05 and _bal_gap > 0.01:
        print("Leitura: o gráfico supera as features numéricas neste teste, em acurácia e em acurácia balanceada (diferença distinguível do ruído).")
    elif _gap > 0.01 and cnn_vs_mlp_pvalue < 0.05:
        print(
            f"Leitura: a CNN ganha em acurácia bruta ({_gap:+.1%}), mas não em acurácia balanceada ({_bal_gap:+.1%}): a vantagem vem de favorecer `baixa`, "
            "a classe mais frequente no teste (revocação alta nela, baixa em `media`), e não de separar melhor os três regimes. Sem vantagem clara do gráfico, a um custo de treino maior."
        )
    elif _gap < -0.01 and cnn_vs_mlp_pvalue < 0.05:
        print("Leitura: as features numéricas superam o gráfico neste teste: a imagem NÃO acrescenta nada mensurável e custa mais caro de treinar (a CNN fica abaixo do MLP/HAR).")
    else:
        print("Leitura: empate técnico entre o gráfico e as features numéricas neste teste: a imagem não acrescenta nada mensurável e custa mais caro de treinar.")
else:
    print("Parte 3 (CNN) não executada nesta sessão: sem comparação números × imagem.")

# %% [markdown]
# ## 4.4 O que funcionou
#
# - **Ganhos modestos e medidos, não presumidos:** o MLP supera a regressão logística no crédito (ROC-AUC ≈ 0,77 × 0,72, PR-AUC ≈ 0,54 × 0,49) e supera o HAR na volatilidade por **margem pequena**
#   (≈ 2% no RMSE e 3% no MAE), diferença que o teste de Diebold-Mariano por pregão (seção 2.9) separa do ruído. Em volatilidade, a **simplicidade venceu**: a rede de uma camada oculta com 32 neurônios foi a melhor na ablação.
# - **Normalização visível e dentro do modelo** (`Normalization.adapt` só no treino): as três redes receberam entradas em escala comparável, e a API de inferência recebe valores **brutos**, sem *skew* de pré-processamento.
# - **Métricas escolhidas pelo problema**, não pelo costume: PR-AUC, KS e custo para o crédito desbalanceado (a acurácia da classe majoritária engana); RMSE/MAE **em unidades naturais** e QLIKE para a volatilidade.
# - **Comparar com baselines simples** evitou conclusões falsas: a regressão logística no crédito e o HAR na volatilidade são adversários sérios, e o ganho do MLP sobre eles é medido, não presumido.
# - **Ablação curta e rastreável** (largura × profundidade × 3 seeds, com e sem `class_weight`, Huber × MSE) com a **regra do 1 erro-padrão**: escolher a rede mais simples indistinguível da melhor.
# - **Validação cronológica com embargo** na volatilidade (70/15/15 por data, 5 pregões de purge) e um **teste unitário de ausência de vazamento** nas features.
# - **Limiar por custo** no crédito: trocar o corte arbitrário 0,5 por um custo explícito (ilustrativo) reduz o custo esperado por cliente em relação a "aprovar todos" e ao corte 0,5.
#
# ## 4.5 Limitações (leia antes de confiar em qualquer número)
#
# - **Viés de sobrevivência:** o S&P 500 do dataset contém apenas as empresas que **estavam no índice em fev/2018**; quem saiu (falência, aquisição) não aparece. Isso tende a **subestimar** a volatilidade e o risco de cauda.
# - **Período curto e regime único:** cinco anos de uma bolsa em alta quase contínua; o teste (mai/2017 a jan/2018) é **muito calmo**, exceto por um **choque no fim** (fevereiro de 2018) que **nenhuma feature de janeiro conhece**. O desempenho em outro regime (2008, 2020) é desconhecido.
# - **Deslocamento de regime treino → teste:** o treino é mais agitado que a validação e o teste (as médias do alvo caem), então os baselines e o modelo têm viés de nível; o erro por mês mostra isso.
# - **Preços ajustados de fonte não auditada** (Kaggle, licença não verificada): splits e spin-offs sem ajuste geram retornos absurdos; descartamos janelas com `|r| > 0,4`, o que também remove alguns eventos reais (AMD, CHK, WMB).
# - **Alvo ruidoso:** a volatilidade "realizada" de **5 retornos** é uma medida barulhenta da volatilidade verdadeira; por isso o R² no log é baixo por construção e o teto de qualquer modelo é modesto.
# - **Crédito:** dados de **Taiwan, 2005** (um único banco, antes da crise de 2008); os **custos do limiar (FN = 5 × FP) são ilustrativos**, não de um banco real; a equidade foi medida só por sexo e só em taxas.
# - **Amostras pequenas nas novas instâncias:** 20 clientes e 12 tickers × 5 dias servem para **ilustrar o uso**, não para estimar desempenho.
#
# ## 4.6 Considerações éticas e de uso
#
# - **Atributos protegidos:** `SEX` e `MARRIAGE` **não são features** (a Regulation B / ECOA, nos EUA, proíbe sexo e estado civil; a idade só em sistemas empiricamente derivados e sem penalizar idosos). Mesmo assim o modelo
#   pode reproduzi-los por *proxies*; por isso medimos taxa de aprovação, TPR e FPR por grupo. Medir não é corrigir: uso real exigiria análise jurídica e testes de mitigação.
# - **Decisões automatizadas:** a LGPD (art. 20) garante ao titular o direito de **solicitar a revisão** de decisões tomadas unicamente com base em tratamento automatizado (⚠️ confirmar o texto no Planalto). Um modelo de crédito
#   real precisaria de explicabilidade por cliente e de um caminho de contestação.
# - **Mercado:** os modelos de volatilidade **não são recomendação de investimento**; descrevem risco (agitação), não retorno. Previsões de volatilidade erradas no choque de 2018 mostram como um modelo "calmo" falha justamente quando importa.
# - **Dados:** o dataset de crédito é CC BY 4.0 (Yeh, 2009); o S&P 500 do Kaggle tem licença não verificada, então **não é redistribuído** e só imagens **sintéticas** vão para fora do notebook.
#
# ## 4.7 Próximos passos
#
# Corrigir o viés de prever a **mediana** na volatilidade (a saída `exp(E[ln y])` fica abaixo da média por desigualdade de Jensen; um fator de suavização `exp(σ²/2)` aproximaria a média), recalibrar as probabilidades do crédito (Platt/isotônica na validação), repetir o diagnóstico de equidade com outros atributos e limiares, validar a volatilidade com **janela deslizante** (*walk-forward*) em mais regimes e
# comparar com modelos de volatilidade condicional (GARCH) e com features de opções (VIX por ação) que o dataset não tem.

# %% [markdown]
# ## 4.8 Fechamento: versões, hardware e tempo de execução

# %%
_total = time.time() - NOTEBOOK_T0
print(f"Python {sys.version.split()[0]} | TensorFlow {tf.__version__} | Keras {keras.__version__} | NumPy {np.__version__} | pandas {pd.__version__} | scikit-learn {sklearn.__version__}")
print(f"Hardware: {HW}")
print(f"Seed base: {SEED} | threads do TensorFlow: {'todas' if THREADS == 0 else THREADS}")
times = {k: v / 60 for k, v in part_times.items()}
show(pd.DataFrame({"minutos": times}).round(1))
print(f"Tempo total até aqui (sem o apêndice de exportação): {_total / 60:.1f} min")
