"""Parte 20 - Rede 2: volatilidade realizada futura (regressao), modelo ``volatility-mlp``.

Usa ``sp`` e ``split`` da parte 15. Variaveis para as partes seguintes: vol_model, vol_infer, vol_results,
vol_baseline_acc_mlp, vol_baseline_acc_har, vol_tercile_edges, VOL_NUMERIC/VOL_FEATURES/DOW_VOCAB/VOL_BOUNDS,
v_mu/v_sigma, vol_abl, build_features().
"""

# %% [markdown]
# # Parte 2 · Risco de mercado: quão agitada vai ser a próxima semana? (regressão)
#
# **Problema.** Prever a volatilidade dos próximos 5 pregões de uma ação do S&P 500 usando só informação até o fechamento de hoje. A volatilidade é a medida mais básica de risco de mercado e é previsível, porque vem em ondas (dias agitados costumam ser seguidos por dias agitados). A direção do preço, ao contrário, é quase aleatória (checamos isso numa célula no fim desta parte).
#
# - **Alvo:** `y_t = sqrt(252 · média(r²_{t+1..t+5}))`, a volatilidade anualizada em fração (0,20 = 20% ao ano).
# - **Por que prever `ln(y)` e não `y`?** A volatilidade é sempre positiva e assimétrica à direita (muitos dias calmos e poucos muito agitados). Em log ela fica quase simétrica, os erros passam a ser proporcionais (errar 0,10 num alvo de 0,10 é grave, mas num de 0,60 não é) e a saída `exp(·)` garante previsão positiva.
# - **Como avaliamos:** em unidades naturais (RMSE, MAE e as margens relativas à média do alvo, como o enunciado pede) e também pelo QLIKE, uma perda muito usada em previsão de volatilidade.
#
# > Conteúdo educacional. Não é recomendação de investimento.

# %% [markdown]
# ## 2.1 Features (todas calculadas só com dados até o fechamento de `t`)
#
# | Grupo | Features |
# |---|---|
# | retornos recentes | `ret_1d..ret_5d`: log-retornos defasados (hoje, ontem, …) |
# | volatilidade realizada | `rv_5`, `rv_10`, `rv_20`, `rv_60` (anualizada) e a razão `rv_ratio = rv_5 / rv_20` (a vol de curto prazo está acima ou abaixo da de médio prazo?) |
# | amplitude intradiária | `pk_5`, `pk_20` (estimador de **Parkinson**, usa máxima e mínima), `range_5` (média de (máx − mín) / fechamento) |
# | liquidez | `volume_ratio` (volume ÷ média de 20 dias), `log_dollar_volume` |
# | gaps e tendência | `gap` (abertura ÷ fechamento anterior − 1), `mom_20`, `mom_60`, `drawdown_60` |
# | mercado | `mkt_rv_20`: volatilidade média do mercado (média de `rv_20` dos 505 tickers no dia, um proxy tipo VIX) |
# | calendário | `dow` (dia da semana, *one-hot*) |
#
# **Sem vazamento.** Cada feature em `t` usa só linhas com data menor ou igual a `t` do mesmo ticker (e, em `mkt_rv_20`, só o fechamento de `t` dos outros tickers). A construção é vetorizada por ticker (`groupby`), e a célula seguinte tem um teste: alterar as linhas futuras não pode mudar nenhuma feature do passado. Amostras sem 60 pregões de histórico (ou com buraco na janela) são descartadas, então não há valores ausentes e não precisamos de imputação.

# %%
_t_part = time.time()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

layers = keras.layers
RAW_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]
VOL_NUMERIC = [f"ret_{k}d" for k in range(1, 6)] + ["rv_5", "rv_10", "rv_20", "rv_60", "pk_5", "pk_20", "range_5", "volume_ratio", "log_dollar_volume", "gap", "mom_20", "mom_60", "drawdown_60", "mkt_rv_20"]  # fmt: skip
DOW_VOCAB = ("mon", "tue", "wed", "thu", "fri")
VOL_FEATURES = [*VOL_NUMERIC, *[f"dow={d}" for d in DOW_VOCAB], "rv_ratio"]
N_VOL = len(VOL_FEATURES)
# limites aceitos pela API (validacao de entrada); conferidos abaixo contra todas as amostras de treino/val/teste
VOL_BOUNDS = {**{f"ret_{k}d": (-0.4, 0.4) for k in range(1, 6)}, "rv_5": (0.0, 10.0), "rv_10": (0.0, 10.0), "rv_20": (1e-6, 10.0), "rv_60": (0.0, 10.0),
              "pk_5": (0.0, 10.0), "pk_20": (0.0, 10.0), "range_5": (0.0, 1.0), "volume_ratio": (1e-6, 1000.0), "log_dollar_volume": (0.0, 40.0),
              "gap": (-1.0, 10.0), "mom_20": (-1.0, 10.0), "mom_60": (-1.0, 10.0), "drawdown_60": (-1.0, 0.0), "mkt_rv_20": (0.0, 5.0)}  # fmt: skip


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features cruas por linha (ticker, date), vetorizadas por ticker. ``df`` precisa estar ordenado por (ticker, date).

    Calcula o log-retorno a partir do fechamento (sem usar colunas externas). Nenhuma feature usa linhas posteriores a ``t``.
    """
    tick = df["ticker"]
    grp = lambda s: s.groupby(tick, sort=False)  # noqa: E731
    roll_mean = lambda s, k: grp(s).transform(lambda x: x.rolling(k).mean())  # noqa: E731
    lr = np.log(df["close"]).groupby(tick, sort=False).diff()
    out: dict[str, pd.Series] = {}
    for k in range(1, 6):
        out[f"ret_{k}d"] = grp(lr).shift(k - 1)
    r2 = lr**2
    for k in (5, 10, 20, 60):
        out[f"rv_{k}"] = np.sqrt(ANN * roll_mean(r2, k))
    hl2 = np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2))  # Parkinson (1980)
    for k in (5, 20):
        out[f"pk_{k}"] = np.sqrt(ANN * roll_mean(hl2, k))
    out["range_5"] = roll_mean((df["high"] - df["low"]) / df["close"], 5)
    out["volume_ratio"] = (df["volume"] / roll_mean(df["volume"].astype("float64"), 20)).clip(lower=1e-6)
    out["log_dollar_volume"] = np.log(df["close"] * df["volume"])
    out["gap"] = df["open"] / grp(df["close"]).shift(1) - 1
    out["mom_20"] = df["close"] / grp(df["close"]).shift(20) - 1
    out["mom_60"] = df["close"] / grp(df["close"]).shift(60) - 1
    out["drawdown_60"] = df["close"] / grp(df["close"]).transform(lambda x: x.rolling(60).max()) - 1
    # mercado: media de rv_20 entre os tickers do mesmo dia; ignora tickers com artefato (|r| > 0.4) nos ultimos 20 pregoes
    artefact = grp((lr.abs() > MAX_ABS_RET).astype("float64")).transform(lambda x: x.rolling(20).max()) > 0
    out["mkt_rv_20"] = out["rv_20"].where(~artefact).groupby(df["date"]).transform("mean")
    feat = pd.DataFrame(out, index=df.index)[VOL_NUMERIC]
    feat["dow"] = df["date"].dt.day_of_week.map(dict(enumerate(DOW_VOCAB)))
    return feat


def vol_feature_matrix(feat: pd.DataFrame) -> np.ndarray:
    """Features cruas -> matriz float32 de 25 colunas (ordem do manifesto: numericas, one-hot de dow, razao rv_5/rv_20)."""
    onehot = np.stack([(feat["dow"] == d).to_numpy(dtype="float64") for d in DOW_VOCAB], axis=1)
    ratio = (feat["rv_5"] / feat["rv_20"]).to_numpy(dtype="float64")[:, None]
    return np.concatenate([feat[VOL_NUMERIC].to_numpy(dtype="float64"), onehot, ratio], axis=1).astype("float32")


_t0 = time.time()
feat_all = build_features(sp[RAW_COLS])
print(f"features de {len(feat_all):,} linhas em {time.time() - _t0:.1f}s")

# %% [markdown]
# ### Teste: nenhuma feature em `t` muda se as linhas futuras mudarem
#
# Pegamos 40 tickers, escolhemos uma data de corte e alteramos os preços e volumes de todas as linhas depois dela (multiplicando por fatores aleatórios entre 0,5 e 2). As features das linhas até o corte têm que sair idênticas. Como controle, o alvo (que olha o futuro de propósito) tem que mudar.

# %%
_tickers = sorted(sp["ticker"].unique())[:: max(1, sp["ticker"].nunique() // 40)][:40]
_sub = sp.loc[sp["ticker"].isin(_tickers), [*RAW_COLS, "logret"]].reset_index(drop=True)
_cut = pd.Timestamp("2016-03-01")
_alt = _sub.copy()
_later = _alt["date"] > _cut
_noise = np.random.default_rng(0)
for _c in ("open", "high", "low", "close", "volume"):
    _alt[_c] = _alt[_c].astype("float64")
    _alt.loc[_later, _c] = _alt.loc[_later, _c] * _noise.uniform(0.5, 2.0, int(_later.sum()))
_f0, _f1 = build_features(_sub[RAW_COLS]), build_features(_alt[RAW_COLS])
_past_same = np.array_equal(_f0.loc[~_later, VOL_NUMERIC].to_numpy(), _f1.loc[~_later, VOL_NUMERIC].to_numpy(), equal_nan=True)
_future_changed = not np.array_equal(_f0.loc[_later, VOL_NUMERIC].to_numpy(), _f1.loc[_later, VOL_NUMERIC].to_numpy(), equal_nan=True)
_alt["logret"] = np.log(_alt["close"]).groupby(_alt["ticker"]).diff()
_y0, _y1 = forward_vol(_sub), forward_vol(_alt)
_target_changed = not np.array_equal(_y0[~_later].to_numpy(), _y1[~_later].to_numpy(), equal_nan=True)
print(f"linhas até o corte ({_cut.date()}): {int((~_later).sum()):,} | linhas futuras alteradas: {int(_later.sum()):,}")
print(f"features do passado idênticas após alterar o futuro: {_past_same}")
print(f"features do futuro mudaram (o teste é sensível): {_future_changed}")
print(f"controle positivo: o ALVO do passado mudou (olha o futuro de propósito): {_target_changed}")
assert _past_same and _future_changed and _target_changed, "VAZAMENTO DE FUTURO nas features!"
del _sub, _alt, _f0, _f1, _y0, _y1

# %% [markdown]
# ## 2.2 Montagem do conjunto, divisão por data e EDA
#
# Usamos as mesmas amostras válidas e a mesma divisão por data da parte 2A (70/15/15 dos pregões, com 5 pregões de intervalo entre as partes e teste terminando 5 pregões antes do fim). O conjunto é o mesmo que a rede 3 (CNN) vai usar. Não amostramos o treino: são ≈ 400 mil linhas × 25 colunas (≈ 40 MB), e o MLP treina em poucos minutos na CPU.

# %%
vol_df = pd.concat([sp[["date", "ticker", "y", "part", "valid", "valid_feat"]], feat_all], axis=1)
sel = vol_df[vol_df["valid"]].copy()
assert not sel[VOL_NUMERIC].isna().any().any(), "amostras válidas não podem ter NaN"
viol = {n: int(((sel[n] < lo) | (sel[n] > hi)).sum()) for n, (lo, hi) in VOL_BOUNDS.items()}
print("violações dos limites de entrada da API:", {k: v for k, v in viol.items() if v} or "nenhuma")
assert not any(viol.values())

parts = {p: sel[sel["part"] == p] for p in ("train", "val", "test")}
overview = pd.DataFrame(
    {p: {"linhas": len(d), "pregões": d["date"].nunique(), "início": d["date"].min().date(), "fim": d["date"].max().date(),
         "y médio": d["y"].mean(), "y mediano": d["y"].median(), "desvio de ln y": np.log(d["y"]).std()} for p, d in parts.items()}
).T
show(overview)
print(split.summary())

vX_train, vX_val, vX_test = (vol_feature_matrix(parts[p]) for p in ("train", "val", "test"))
vy_train, vy_val, vy_test = (parts[p]["y"].to_numpy() for p in ("train", "val", "test"))
v_dates_test = parts["test"]["date"].to_numpy()
print(f"vX_train {vX_train.shape} | vX_val {vX_val.shape} | vX_test {vX_test.shape} | memória do treino: {vX_train.nbytes / 1e6:.0f} MB")

# %%
daily = sel.groupby("date")["y"].mean()
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1.6, 1, 1]})
axes[0].plot(daily.index, daily.to_numpy(), lw=0.9, color="#4c78a8")
for a, b, c, lab in (("train_start", "train_end", "#54a24b", "treino"), ("val_start", "val_end", "#f58518", "validação"), ("test_start", "test_end", "#e45756", "teste")):
    axes[0].axvspan(split.bounds[a], split.bounds[b], color=c, alpha=0.15, label=lab)
axes[0].set_title("Volatilidade média do mercado (alvo médio por dia) e a divisão cronológica")
axes[0].set_ylabel("y médio (a.a.)")
axes[0].legend(loc="upper left")
axes[1].hist(parts["train"]["y"], bins=80, color="#4c78a8")
axes[1].set_title("y (treino): assimétrico à direita")
axes[1].set_xlim(0, 1.2)
axes[2].hist(np.log(parts["train"]["y"]), bins=80, color="#54a24b")
axes[2].set_title("ln y (treino): quase simétrico")
plt.show()

# %% [markdown]
# **O que vemos.** A volatilidade vem em ondas: picos em ago/2015, jan–fev/2016 e jun/2016 (treino), um mercado bem calmo em 2017 (validação e começo do teste) e um salto em jan–fev/2018, no fim do teste (o alvo das últimas amostras do teste já cobre os primeiros pregões de fevereiro). Isso tem duas consequências. Primeiro, o treino é mais agitado que a validação e o teste (as médias da tabela acima: 0,215 contra 0,193 e 0,191), então um modelo que só decora a média do treino erra por nível. Segundo, o salto de 2018 é um choque que nenhuma feature de janeiro conhece, então o erro no fim do teste deve ser grande para qualquer modelo. Por isso também olhamos o erro por período. O histograma confirma que `y` é assimétrico e que `ln y` é quase simétrico.

# %%
corr_ln = sel.assign(ln_y=np.log(sel["y"]))[[*VOL_NUMERIC, "ln_y"]].corr()["ln_y"].drop("ln_y").sort_values()
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(corr_ln.index, corr_ln.to_numpy(), color=np.where(corr_ln.to_numpy() > 0, "#e45756", "#4c78a8"))
ax.set_title("Correlação das features com ln y (todas as amostras válidas, só exploratório)")
plt.show()
print("Maiores |correlação| com ln y:", ", ".join(f"{k} ({v:+.2f})" for k, v in corr_ln.reindex(corr_ln.abs().sort_values(ascending=False).index).head(6).items()))

# %% [markdown]
# **O que vemos.** As medidas de volatilidade passada (`rv_*`, `pk_*`, `range_5`) e a do mercado (`mkt_rv_20`) dominam, como esperado do agrupamento de volatilidade. As janelas longas (`rv_60`, `pk_20`) e a amplitude `range_5` correlacionam mais que a `rv_5`, porque a volatilidade de só 5 retornos é uma estimativa ruidosa. `drawdown_60` (≈ −0,33) e `mom_*` (≈ −0,13) têm correlação negativa: ações em queda ficam mais voláteis. Os retornos defasados `ret_*d`, com sinal, quase não se correlacionam linearmente com o log da volatilidade (o tamanho do movimento importa mais que o sinal), mas mantivemos essas colunas para a rede poder usá-las de forma não linear. A correlação é só exploratória (calculada com todas as amostras válidas) e não entra na escolha das features.

# %% [markdown]
# ## 2.3 Preparação: normalização das features e padronização do alvo (`ln y`)
#
# - **Features:** `Normalization.adapt()` só no treino, e a camada fica dentro do modelo. A tabela mostra a média e o desvio antes e depois.
# - **Alvo:** `ln y` é padronizado com a média e o desvio do treino (`z = (ln y − μ) / σ`). A rede aprende `z`, e para voltar à volatilidade em fração ao ano usamos `z·σ + μ` e depois `exp`, no clone de inferência.

# %%
v_norm = layers.Normalization(axis=-1, name="norm")
v_norm.adapt(vX_train)
v_mean, v_var = np.asarray(v_norm.mean).reshape(-1), np.asarray(v_norm.variance).reshape(-1)
vZ_train, vZ_val = v_norm(vX_train).numpy(), v_norm(vX_val).numpy()
show(
    pd.DataFrame(
        {"média (antes)": vX_train.mean(axis=0), "desvio (antes)": vX_train.std(axis=0), "máx (antes)": vX_train.max(axis=0),
         "média (depois)": vZ_train.mean(axis=0), "desvio (depois)": vZ_train.std(axis=0), "média val. (depois)": vZ_val.mean(axis=0)},
        index=VOL_FEATURES,
    )
)
ln_train = np.log(vy_train)
v_mu, v_sigma = float(ln_train.mean()), float(ln_train.std())
vz_train, vz_val = (np.log(vy_train) - v_mu) / v_sigma, (np.log(vy_val) - v_mu) / v_sigma
vz_test = (np.log(vy_test) - v_mu) / v_sigma
print(f"alvo: ln y no treino tem média μ = {v_mu:.4f} e desvio σ = {v_sigma:.4f} | z no treino: média {vz_train.mean():+.3f}, desvio {vz_train.std():.3f} | z na validação: média {vz_val.mean():+.3f}")

# %% [markdown]
# **O que vemos.** As features ficam com média 0 e desvio 1 no treino. A média da validação (última coluna) não é 0 em várias colunas (`rv_*`, `mkt_rv_20`): é o deslocamento de regime, já que a validação é mais calma que o treino. É isso que a regra de calcular as estatísticas só com o treino preserva. Padronizar o alvo ajuda o otimizador (a saída fica na ordem de 1) e dá uma escala natural para a perda Huber.

# %% [markdown]
# ## 2.4 Baselines e métricas
#
# Um MLP só vale a pena se vencer alternativas simples:
# - **Média do treino:** prevê sempre a volatilidade média do treino;
# - **Persistência:** prevê que a próxima semana terá a volatilidade da última (`rv_5`);
# - **HAR linear** (Corsi, 2009): regressão linear de `ln y` em `ln rv_5`, `ln rv_20` e `ln rv_60` (semana, mês e trimestre). É o baseline clássico de previsão de volatilidade e é difícil de bater.
#
# **Métricas** (em unidades naturais, fração ao ano, menos a última): RMSE, MAE, MAE / média do alvo e RMSE / média do alvo (a margem de erro relativa que o enunciado pede), QLIKE `média(y²/ŷ² − ln(y²/ŷ²) − 1)` (subestimar a volatilidade custa mais que superestimar, o que faz sentido em risco) e R² no log.

# %%
def qlike(y, yhat) -> float:
    ratio = (np.asarray(y, dtype="float64") / np.asarray(yhat, dtype="float64")) ** 2
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def vol_scores(y, yhat) -> dict[str, float]:
    y, yhat = np.asarray(y, dtype="float64"), np.asarray(yhat, dtype="float64")
    err = yhat - y
    rmse, mae = float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))
    return {"RMSE": rmse, "MAE": mae, "MAE/média": mae / float(y.mean()), "RMSE/média": rmse / float(y.mean()), "QLIKE": qlike(y, yhat),
            "R² (log)": float(r2_score(np.log(y), np.log(yhat))), "viés": float(err.mean())}  # fmt: skip


def har_design(X) -> np.ndarray:
    cols = [VOL_FEATURES.index(c) for c in ("rv_5", "rv_20", "rv_60")]
    return np.column_stack([np.ones(len(X)), np.log(np.clip(X[:, cols], VOL_FLOOR, None))])


har_beta, *_ = np.linalg.lstsq(har_design(vX_train), vz_train, rcond=None)  # em unidades padronizadas de ln y


def har_predict(X) -> np.ndarray:
    return np.exp(v_mu + v_sigma * (har_design(X) @ har_beta))


def persistence_predict(X) -> np.ndarray:
    return np.clip(X[:, VOL_FEATURES.index("rv_5")], VOL_FLOOR, None)


def mean_predict(X) -> np.ndarray:
    return np.full(len(X), vy_train.mean())


baselines = {"média do treino": mean_predict, "persistência (rv_5)": persistence_predict, "HAR linear": har_predict}
val_base = pd.DataFrame({k: vol_scores(vy_val, f(vX_val)) for k, f in baselines.items()}).T
print("Baselines na VALIDAÇÃO (o teste só é usado uma vez, no fim):")
show(val_base)
print(f"HAR: coeficientes (padronizados) = {np.round(har_beta, 3)} (intercepto, ln rv_5, ln rv_20, ln rv_60)")

# %% [markdown]
# ## 2.5 Perda Huber: `delta` em unidades de `ln y` padronizado
#
# A perda Huber é quadrática para erros pequenos e linear para os grandes (`|e| > δ`), então limita a influência de eventos extremos. O `δ` precisa estar na escala do ruído. Calibramos assim: pegamos os resíduos do HAR no treino (em unidades de `z`), estimamos o desvio robusto `σ_res = 1,4826 · MAD` (que não é afetado pelas caudas) e usamos `δ = 1,345 · σ_res`, a constante clássica de Huber, que mantém ≈ 95% da eficiência do MSE quando o ruído é normal.
#
# A Huber é uma candidata, não uma decisão: na seção 2.6 comparamos com o MSE (3 seeds) e a escolha final sai da validação.

# %%
_res = vz_train - har_design(vX_train) @ har_beta
sigma_res = 1.4826 * float(np.median(np.abs(_res - np.median(_res))))
HUBER_DELTA = round(1.345 * sigma_res, 3)
print(f"resíduo do HAR no treino (em z): desvio = {_res.std():.3f}, desvio robusto (MAD) = {sigma_res:.3f} | fração de resíduos com |e| > δ: {(np.abs(_res) > HUBER_DELTA).mean():.1%}")
print(f"δ de Huber = 1,345 × {sigma_res:.3f} = {HUBER_DELTA}  (em unidades de ln y padronizado; em ln y cru: {HUBER_DELTA * v_sigma:.3f}, ou seja, ~{(np.exp(HUBER_DELTA * v_sigma) - 1):.0%} de erro multiplicativo)")

# %% [markdown]
# ## 2.6 Ablação: largura × profundidade × perda (3 seeds cada)
#
# A grade tem largura da 1ª camada oculta em {32, 64, 128} × profundidade em {1, 2, 3} (cada camada tem metade da anterior), com Huber em todas e 3 seeds. Depois, na configuração escolhida, comparamos Huber × MSE (3 seeds). O treino usa Adam 1e-3, `ReduceLROnPlateau` e `EarlyStopping` na loss de validação. Na decisão usamos a validação completa, com o QLIKE como critério principal (padrão em previsão de volatilidade) e a regra do 1 erro-padrão: ficamos com a configuração mais simples dentro do ruído da melhor.
#
# **Amostragem só na ablação (para caber na CPU):** cada treino da grade usa 1 a cada 3 pregões do treino (≈ 133 mil linhas). Dias seguidos são quase redundantes (os alvos de 5 dias se sobrepõem e as features mudam pouco de um dia para o outro), então sortear por data perde pouca informação. O modelo final treina com todas as linhas.

# %%
def build_vol_mlp(width: int, depth: int, name: str = "volatility_mlp") -> keras.Model:
    """Camada a camada: Input -> Normalization (treino) -> [Dense(relu)] x depth -> Dense(1) (ln y padronizado, saida linear)."""
    inputs = keras.Input(shape=(N_VOL,), name="features")
    x = layers.Normalization(axis=-1, mean=v_mean, variance=v_var, name="norm")(inputs)
    for i in range(depth):
        x = layers.Dense(max(8, width // 2**i), activation="relu", name=f"dense_{i}")(x)
    outputs = layers.Dense(1, name="out")(x)
    return keras.Model(inputs, outputs, name=name)


def to_natural(z) -> np.ndarray:
    return np.exp(v_mu + v_sigma * np.asarray(z, dtype="float64").reshape(-1))


_train_day = np.searchsorted(split.days, parts["train"]["date"].to_numpy())
abl_mask = _train_day % 3 == 0  # regra de amostragem da ablação: 1 a cada 3 pregoes do treino
vX_abl, vz_abl = vX_train[abl_mask], vz_train[abl_mask]
print(f"ablação: {abl_mask.sum():,} de {len(abl_mask):,} linhas de treino ({abl_mask.mean():.0%}); validação completa ({len(vX_val):,})")


def fit_vol(width: int, depth: int, seed: int, loss: str, max_epochs: int = 30, verbose: int = 0, full: bool = False):
    set_seed(seed)
    model = build_vol_mlp(width, depth)
    loss_obj = keras.losses.Huber(delta=HUBER_DELTA) if loss == "huber" else "mse"
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss_obj, metrics=["mae"], jit_compile=False)
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    x, z = (vX_train, vz_train) if full else (vX_abl, vz_abl)
    history = model.fit(x, z, validation_data=(vX_val, vz_val), epochs=max_epochs, batch_size=1024, callbacks=callbacks, verbose=verbose)
    return model, history


_rows = []
_t0 = time.time()
_runs = [(w, d, "huber") for d in (1, 2, 3) for w in (32, 64, 128)]
for width, depth, loss in _runs:
    for seed in (1, 2, 3):
        t1 = time.time()
        m, h = fit_vol(width, depth, seed, loss)
        sc = vol_scores(vy_val, to_natural(m.predict(vX_val, batch_size=8192, verbose=0)))
        _rows.append({"loss": loss, "depth": depth, "width": width, "seed": seed, "params": m.count_params(), "val_QLIKE": sc["QLIKE"], "val_RMSE": sc["RMSE"],
                      "val_MAE": sc["MAE"], "val_R2log": sc["R² (log)"], "epochs": len(h.history["loss"]), "seconds": time.time() - t1})  # fmt: skip
    print(f"  {loss} depth={depth} width={width} pronto ({time.time() - _t0:.0f}s)", flush=True)
vol_abl = pd.DataFrame(_rows)
keras.backend.clear_session()

# %%
agg_v = (
    vol_abl.groupby(["loss", "depth", "width"])
    .agg(params=("params", "first"), qlike=("val_QLIKE", "mean"), qlike_std=("val_QLIKE", "std"), rmse=("val_RMSE", "mean"), mae=("val_MAE", "mean"), r2log=("val_R2log", "mean"), epochs=("epochs", "mean"))
    .reset_index()
)  # fmt: skip
agg_h = agg_v[agg_v["loss"] == "huber"].copy()
agg_h["se"] = agg_h["qlike_std"] / np.sqrt(3)
top = agg_h.loc[agg_h["qlike"].idxmin()]
near = agg_h[agg_h["qlike"] <= top["qlike"] + top["se"]]
pick = near.sort_values(["params", "qlike"]).iloc[0]
VOL_WIDTH, VOL_DEPTH = int(pick["width"]), int(pick["depth"])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, col, ttl in zip(axes, ("qlike", "rmse"), ("QLIKE na validação (menor é melhor)", "RMSE na validação (fração a.a.)"), strict=True):
    mean = agg_h.pivot(index="depth", columns="width", values=col)
    std = vol_abl[vol_abl["loss"] == "huber"].groupby(["depth", "width"])["val_QLIKE" if col == "qlike" else "val_RMSE"].std().unstack()
    labels = mean.map("{:.4f}".format) + "\n±" + std.map("{:.4f}".format)
    sns.heatmap(mean, annot=labels, fmt="", cmap="viridis_r", ax=ax, cbar=False)
    ax.set_title(f"{ttl}\nmédia ± desvio de 3 seeds (Huber)")
    ax.set_xlabel("largura da 1ª camada oculta")
    ax.set_ylabel("nº de camadas ocultas")
plt.show()
show(agg_h.sort_values("qlike").head(6))
print(f"melhor QLIKE: {top['qlike']:.4f} (largura {int(top['width'])}, profundidade {int(top['depth'])}, EP {top['se']:.4f}) -> pela regra do 1 EP: largura {VOL_WIDTH}, profundidade {VOL_DEPTH} ({int(pick['params']):,} parâmetros)")

# %%
_rows = []
for loss in ("mse", "huber"):
    for seed in (1, 2, 3):
        m, h = fit_vol(VOL_WIDTH, VOL_DEPTH, seed, loss)
        sc = vol_scores(vy_val, to_natural(m.predict(vX_val, batch_size=8192, verbose=0)))
        _rows.append({"loss": loss, "seed": seed, "val_QLIKE": sc["QLIKE"], "val_RMSE": sc["RMSE"], "val_MAE": sc["MAE"], "val_R2log": sc["R² (log)"], "epochs": len(h.history["loss"])})
keras.backend.clear_session()
loss_cmp = pd.DataFrame(_rows)
loss_tab = loss_cmp.groupby("loss")[["val_QLIKE", "val_RMSE", "val_MAE", "val_R2log"]].agg(["mean", "std"])
show(loss_tab)
_q = loss_cmp.groupby("loss")["val_QLIKE"].agg(["mean", "std"])
VOL_LOSS = "huber" if _q.loc["huber", "mean"] <= _q.loc["mse", "mean"] + _q.loc["mse", "std"] / np.sqrt(3) else "mse"  # empate (1 EP) -> Huber, mais robusta
print(f"DECISÃO: perda = {VOL_LOSS} (QLIKE de validação: Huber {_q.loc['huber', 'mean']:.4f} × MSE {_q.loc['mse', 'mean']:.4f}) | largura {VOL_WIDTH}, profundidade {VOL_DEPTH}")

# %% [markdown]
# **O que vemos na ablação.** Em RMSE a grade inteira é praticamente igual (≈ 0,129 em todas as células). No QLIKE, as redes de 1 camada são as melhores e mais estáveis entre seeds, e as de 2 e 3 camadas ficam um pouco piores (e mais instáveis, principalmente com 64 e 128 neurônios): mais capacidade não ajudou, e aumentou a variância. Isso faz sentido, porque a previsão de volatilidade é dominada por poucas features lineares em log (as `rv_*`), o HAR linear é um baseline forte e há pouco dado independente (≈ 820 pregões de treino). A regra do 1 erro-padrão escolhe a rede mais simples dentro do ruído da melhor, que aqui também é a de menor QLIKE.
#
# Na comparação Huber × MSE (3 seeds) deu praticamente empate (RMSE e MAE iguais até a 4ª casa, QLIKE diferindo na 3ª). A regra que definimos antes de ver o resultado (ficar com a Huber, que é mais robusta, a menos que ela seja pior que a MSE por mais de 1 erro-padrão) escolheu a MSE (linha `DECISÃO` acima): a Huber ficou ≈ 0,003 de QLIKE atrás, um pouco acima de 1 erro-padrão da MSE (≈ 0,0025). A robustez da Huber não deu ganho mensurável nesta amostra, talvez porque o alvo já está em log, o que suaviza os extremos. (Se uma nova execução inverter a decisão, vale reler a justificativa de compilação da seção 2.7.)

# %% [markdown]
# ## 2.7 Modelo final, camada a camada, e justificativas
#
# ```
# Input(25) → Normalization → Dense(w, relu) → … → Dense(1)        [treino: ln y padronizado]
# clone de inferência: … → Normalization(invert=True) → Activation("exponential")   [saída: volatilidade, fração a.a.]
# ```
#
# A ablação escolheu 1 camada oculta com 32 neurônios.
#
# ### Justificativa: Arquitetura (Rede 2)
# - **Ativação da camada oculta: ReLU.** A entrada é padronizada, as camadas são rasas e a relação entre as features e a volatilidade futura é aproximadamente linear por partes no log (a volatilidade futura cresce com a passada, com quebras em regimes extremos). A ReLU aprende isso sem saturar, é barata e converge rápido. `tanh` e `sigmoid` saturariam nas caudas (dias de pânico) e atrasariam o treino.
# - **Profundidade e largura:** a grade da seção 2.6 (3 seeds) mostrou que mais capacidade não ajuda (as redes de 1 camada foram as melhores no QLIKE). Ficamos com a menor rede dentro do erro-padrão da melhor, que é 1 camada oculta de 32 neurônios. Poucas camadas bastam porque o sinal vem de poucas features e há pouco dado independente (≈ 820 pregões de treino, e as ações de um mesmo dia andam juntas por causa do mercado).
# - **Saída: `Dense(1)` linear**, sobre o alvo padronizado em log. É uma regressão sem limite natural, e a positividade vem do `exp(·)` no clone de inferência (`Normalization(invert=True)` desfaz a padronização e `Activation("exponential")` devolve a volatilidade em fração ao ano). Uma ReLU ou softplus na saída cortaria previsões e atrapalharia o gradiente.
#
# ### Justificativa: Compilação (Rede 2)
# - **Loss: `mse`, sobre `ln y` padronizado.** Na ablação (seção 2.6, 3 seeds) comparamos duas candidatas: a Huber, com `δ = 1,345·σ_res` (seção 2.5), que é quadrática no miolo e linear nas caudas, e a MSE. As duas empataram em RMSE e MAE. No QLIKE de validação (nosso critério principal) a MSE ficou um pouco à frente (0,8036 contra 0,8064), e pela regra que fixamos antes (Huber, a menos que seja pior por mais de 1 erro-padrão) a escolha foi a MSE. O log já deixa o erro proporcional e achata as caudas, então a robustez extra da Huber não compensou. (A MSE em `ln y` estima a média de `ln y`, então a saída `exp(·)` é uma média geométrica, próxima da mediana, e não a média da volatilidade. Voltamos a esse limite na seção 4.7.)
# - **Taxa de aprendizado: Adam 1e-3 com `ReduceLROnPlateau` (fator 0,5, paciência 2).** 1e-3 é estável para MLPs com entrada padronizada, e reduzir a taxa quando a validação estagna refina o mínimo sem precisar escolher o ponto de parada na mão. O `EarlyStopping` (paciência 5, `restore_best_weights`) evita overfitting.
# - **Métricas:** RMSE e MAE em unidades naturais (exigidos) e as razões com a média do alvo (margem de erro relativa); QLIKE (padrão da área, penaliza mais subestimar o risco) e R² no log como complementos. Durante o treino acompanhamos a `mae` em unidades de `z` e a loss.

# %%
set_seed(SEED)
vol_model = build_vol_mlp(VOL_WIDTH, VOL_DEPTH)
loss_obj = keras.losses.Huber(delta=HUBER_DELTA) if VOL_LOSS == "huber" else "mse"
vol_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss_obj, metrics=["mae"], jit_compile=False)
vol_model.summary()

# %%
_t1 = time.time()
vol_hist = vol_model.fit(
    vX_train, vz_train, validation_data=(vX_val, vz_val), epochs=60, batch_size=1024,
    callbacks=[keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
               keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)],
    verbose=0,
)  # fmt: skip
vol_train_seconds = time.time() - _t1
vh = pd.DataFrame(vol_hist.history)
best_ep = int(vh["val_loss"].idxmin()) + 1
print(f"{len(vh)} épocas em {vol_train_seconds:.0f}s | melhor época: {best_ep} | val_loss = {vh['val_loss'].min():.4f}")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
axes[0].plot(vh["loss"], label="treino")
axes[0].plot(vh["val_loss"], label="validação")
axes[0].set_title(f"Loss ({VOL_LOSS}, em z)")
axes[1].plot(vh["mae"], label="treino")
axes[1].plot(vh["val_mae"], label="validação")
axes[1].set_title("MAE em ln y padronizado")
axes[2].plot(vh["learning_rate"], color="C2")
axes[2].set_yscale("log")
axes[2].set_title("Taxa de aprendizado (ReduceLROnPlateau)")
for ax in axes:
    ax.axvline(best_ep - 1, color="k", ls=":", lw=1)
    ax.set_xlabel("época")
axes[0].legend()
axes[1].legend()
plt.show()

# %% [markdown]
# **O que vemos.** A loss de validação é maior que a de treino (≈ 0,87 contra ≈ 0,68, em unidades de `z`), mas não sobe ao longo das épocas: a distância é grande e constante. Isso não parece overfitting clássico (que mostraria a validação piorando enquanto o treino melhora). Parece o deslocamento de regime: a validação (2016–2017) é mais calma e tem outro nível de volatilidade que o treino (a média de `z` na validação é negativa, seção 2.3), e uma rede de 865 parâmetros mal consegue decorar 400 mil linhas. A validação para de melhorar em poucas épocas, o `EarlyStopping` restaura os melhores pesos (linha pontilhada) e o `ReduceLROnPlateau` corta a taxa quando estagna (painel da direita).

# %% [markdown]
# ## 2.8 Clone de inferência: de volta às unidades naturais
#
# O clone de inferência usa os mesmos pesos do modelo treinado e acrescenta `Normalization(invert=True)` (desfaz a padronização do alvo) e `Activation("exponential")`. A saída passa a ser a volatilidade anualizada, em fração.

# %%
v_denorm = layers.Normalization(axis=-1, mean=np.array([v_mu], dtype="float32"), variance=np.array([v_sigma**2], dtype="float32"), invert=True, name="target")
vol_infer = keras.Model(vol_model.input, layers.Activation("exponential", name="exp")(v_denorm(vol_model.output)), name="volatility_inference")
_chk = vol_infer.predict(vX_val[:1000], verbose=0).reshape(-1)
assert np.allclose(_chk, to_natural(vol_model.predict(vX_val[:1000], verbose=0)), rtol=1e-4)
print("clone de inferência == exp(z·σ + μ):", "OK")

# %% [markdown]
# ## 2.9 Avaliação no teste (usado uma única vez)
#
# Avaliamos o modelo e os baselines uma vez no teste (2017-05 a 2018-01, e o alvo das últimas amostras já olha para o salto de fevereiro/2018). Mostramos as métricas exigidas (RMSE, MAE, MAE/média e RMSE/média, em unidades naturais) e algumas extras (QLIKE, R² no log e viés).

# %%
yhat_mlp = vol_infer.predict(vX_test, batch_size=8192, verbose=0).reshape(-1).astype("float64")
yhat = {"MLP": yhat_mlp, **{k: f(vX_test) for k, f in baselines.items()}}
vol_test_table = pd.DataFrame({k: vol_scores(vy_test, v) for k, v in yhat.items()}).T.loc[["MLP", "HAR linear", "persistência (rv_5)", "média do treino"]]
show(vol_test_table)
print(f"média do alvo no teste: {vy_test.mean():.4f} | amostras: {len(vy_test):,}")
mlp = vol_test_table.loc["MLP"]
print(f"MLP: RMSE = {mlp['RMSE']:.4f} ({mlp['RMSE/média']:.1%} da média), MAE = {mlp['MAE']:.4f} ({mlp['MAE/média']:.1%} da média), QLIKE = {mlp['QLIKE']:.4f}, R² (log) = {mlp['R² (log)']:.3f}")

# %% [markdown]
# **O que vemos.** `MAE/média` e `RMSE/média` são a margem de erro relativa pedida: comparam o erro típico (MAE) e o erro quadrático (RMSE) com a volatilidade média do teste. O erro típico é da ordem de 40% da média, o que mostra que prever a volatilidade de 5 pregões de uma ação é difícil, porque o alvo tem muito ruído. A persistência (`rv_5`) é a pior em RMSE, MAE e QLIKE, já que a volatilidade de 5 dias é uma estimativa ruidosa. O HAR é o rival mais sério, e o MLP o supera por uma margem pequena (≈ 2% no RMSE e 3% no MAE, e R² no log maior). Mais abaixo, um teste estatístico por pregão verifica se essa diferença é maior que o ruído.
#
# Um detalhe no QLIKE: a média do treino tem o menor QLIKE do teste, apesar de ter R² no log negativo (ela não ordena nada). Isso vem do deslocamento de regime: o treino foi mais agitado, então a média do treino superestima o teste, e o QLIKE perdoa superestimar e pune subestimar. Já o MLP e o HAR subestimam (viés negativo, em parte porque preveem a mediana `exp(E[ln y])`, que fica abaixo da média, e em parte porque não conhecem o choque de 2018). Por isso o QLIKE do teste favorece aqui um modelo enviesado para cima, e não o tomamos como prova de que a média é melhor: lemos QLIKE, RMSE, MAE e R² no log em conjunto.

# %%
test_df = parts["test"][["date", "ticker", "y"]].copy()
for k, v in yhat.items():
    test_df[k] = v
test_df["mês"] = test_df["date"].dt.to_period("M").astype(str)
months = []
for mth, d in test_df.groupby("mês"):
    row = {"mês": mth, "amostras": len(d), "y médio": d["y"].mean()}
    for k in ("MLP", "HAR linear", "persistência (rv_5)", "média do treino"):
        row[f"RMSE {k}"] = float(np.sqrt(np.mean((d[k] - d["y"]) ** 2)))
    row["MAE MLP"] = float(np.mean(np.abs(d["MLP"] - d["y"])))
    row["viés MLP"] = float(np.mean(d["MLP"] - d["y"]))
    months.append(row)
by_month = pd.DataFrame(months).set_index("mês")
show(by_month)
daily_t = test_df.groupby("date")[["y", "MLP", "HAR linear", "persistência (rv_5)"]].mean()
fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
axes[0].plot(daily_t.index, daily_t["y"], color="k", lw=1.6, label="real (média dos tickers)")
axes[0].plot(daily_t.index, daily_t["MLP"], color="C0", lw=1.2, label="MLP")
axes[0].plot(daily_t.index, daily_t["HAR linear"], color="C1", lw=1.0, ls="--", label="HAR")
axes[0].set_title("Teste: volatilidade futura média por dia, real × prevista (salto de jan–fev/2018 no fim)")
axes[0].legend()
axes[0].tick_params(axis="x", rotation=25)
width_m = 0.27
xs = np.arange(len(by_month))
for i, (k, c) in enumerate((("MLP", "C0"), ("HAR linear", "C1"), ("persistência (rv_5)", "C2"))):
    axes[1].bar(xs + (i - 1) * width_m, by_month[f"RMSE {k}"], width_m, label=k, color=c)
axes[1].set_xticks(xs, by_month.index, rotation=25)
axes[1].set_title("RMSE por mês do teste")
axes[1].legend()
plt.show()

# %% [markdown]
# **O que vemos.** O RMSE não piora de forma clara em jan/2018 (varia de ≈ 0,09 a ≈ 0,17 entre os meses, com o pior em set/2017), mas o viés do MLP em jan/2018 é cerca de 3 vezes maior que nos outros meses: o alvo das últimas amostras de janeiro já olha para os pregões do salto de fevereiro, enquanto as features ainda mostram um mercado calmo. Nenhum previsor que use só o passado consegue evitar isso, e isso também mostra os limites do trabalho (período curto, um único regime e um choque no teste). A curva real × prevista mostra que o MLP e o HAR ficam quase planos (≈ 0,16–0,19): acompanham o nível do mercado em 2017, mas não as ondas de curto prazo (as altas de jul, set e nov), e só reagem ao choque de fevereiro nos últimos dias, e muito menos que o real. A diferença entre os dois modelos é pequena em todos os meses.

# %%
# A vantagem do MLP sobre o HAR é distinguível do ruído? Teste de Diebold-Mariano com a perda agregada por PREGÃO (ações no mesmo dia são fortemente
# correlacionadas pelo mercado: a unidade quase independente é o dia, não a linha) e variância de longo prazo de Newey-West (defasagem 5).
from scipy.stats import norm


def diebold_mariano(df: pd.DataFrame, a: str, b: str, loss, lag: int = 5) -> tuple[float, float, float]:
    """Média de loss(b) - loss(a) por dia (> 0: `a` é melhor), estatística t e p-valor bilateral (normal)."""
    d = (loss(df[b] - df["y"]) - loss(df[a] - df["y"])).groupby(df["date"]).mean().to_numpy()
    n = len(d)
    c = d - d.mean()
    lrv = c @ c / n + 2 * sum((1 - k / (lag + 1)) * (c[k:] @ c[:-k]) / n for k in range(1, lag + 1))
    t = d.mean() / np.sqrt(lrv / n)
    return float(d.mean()), float(t), float(2 * (1 - norm.cdf(abs(t))))


vol_dm = {}
for _name, _loss in (("erro quadrático", np.square), ("erro absoluto", np.abs)):
    for _rival in ("HAR linear", "persistência (rv_5)", "média do treino"):
        vol_dm[(_name, _rival)] = diebold_mariano(test_df, "MLP", _rival, _loss)
vol_dm_table = pd.DataFrame(vol_dm, index=["ganho médio do MLP por dia", "t (Newey-West)", "p-valor"]).T
vol_dm_table.index.names = ["perda", "rival"]
show(vol_dm_table)
_har_sq, _har_abs = vol_dm[("erro quadrático", "HAR linear")], vol_dm[("erro absoluto", "HAR linear")]
print(f"MLP × HAR: erro quadrático p = {_har_sq[2]:.3g} | erro absoluto p = {_har_abs[2]:.3g} ({len(test_df['date'].unique())} pregões de teste)")
print("Leitura: " + ("a vantagem do MLP sobre o HAR é estatisticamente distinguível do ruído (p < 0,05) em ambas as perdas, embora pequena em tamanho." if max(_har_sq[2], _har_abs[2]) < 0.05
      else "a vantagem do MLP sobre o HAR é pequena e NÃO é distinguível do ruído em pelo menos uma das perdas: tratamos os dois como equivalentes em poder preditivo."))

# %%
edges = regime_edges(vy_train)  # tercis do alvo calculados SO no treino (V3.8)
vol_tercile_edges = edges
true_cls = regime_labels(vy_test, edges)
reg_rows = []
for cls, name in enumerate(REGIME_NAMES):
    m = true_cls == cls
    row = {"regime real": name, "amostras": int(m.sum()), "y médio": vy_test[m].mean()}
    for k in ("MLP", "HAR linear"):
        row[f"RMSE {k}"] = float(np.sqrt(np.mean((yhat[k][m] - vy_test[m]) ** 2)))
        row[f"MAE {k}"] = float(np.mean(np.abs(yhat[k][m] - vy_test[m])))
        row[f"viés {k}"] = float(np.mean(yhat[k][m] - vy_test[m]))
    reg_rows.append(row)
by_regime = pd.DataFrame(reg_rows).set_index("regime real")
print(f"cortes dos tercis (treino): baixa < {edges[0]:.4f} <= média < {edges[1]:.4f} <= alta")
show(by_regime)
print("(Nota: agrupar pelo regime REAL mistura um efeito de regressão à média: nos regimes extremos o erro é assimétrico mesmo para um modelo sem viés condicional.)")

vol_baseline_acc_mlp = float((regime_labels(yhat["MLP"], edges) == true_cls).mean())
vol_baseline_acc_har = float((regime_labels(yhat["HAR linear"], edges) == true_cls).mean())
vol_baseline_acc_persist = float((regime_labels(yhat["persistência (rv_5)"], edges) == true_cls).mean())
vol_majority_acc = float(np.bincount(true_cls, minlength=3).max() / len(true_cls))
print(f"Acurácia em 3 classes de regime (tercis do treino), teste: MLP = {vol_baseline_acc_mlp:.3%} | HAR = {vol_baseline_acc_har:.3%} | persistência = {vol_baseline_acc_persist:.3%} | classe majoritária = {vol_majority_acc:.3%} | acaso = 33,3%")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
sub = np.random.default_rng(SEED).choice(len(vy_test), size=min(40000, len(vy_test)), replace=False)
hb = axes[0].hexbin(np.log(yhat_mlp[sub]), np.log(vy_test[sub]), gridsize=60, bins="log", cmap="viridis", mincnt=1)
lims = [min(np.log(yhat_mlp[sub]).min(), np.log(vy_test[sub]).min()), max(np.log(yhat_mlp[sub]).max(), np.log(vy_test[sub]).max())]
axes[0].plot(lims, lims, "r--", lw=1)
axes[0].set(xlabel="ln previsto", ylabel="ln real", title="Previsto × real (teste, escala log; vermelho: y = x)")
resid = np.log(vy_test) - np.log(yhat_mlp)
axes[1].hist(resid, bins=100, color="#4c78a8")
axes[1].axvline(0, color="r")
axes[1].set(xlabel="resíduo = ln real − ln previsto", title=f"Resíduos (média {resid.mean():+.3f}, desvio {resid.std():.3f})")
axes[2].hexbin(np.log(yhat_mlp[sub]), resid[sub], gridsize=60, bins="log", cmap="viridis", mincnt=1)
axes[2].axhline(0, color="r", lw=1)
axes[2].set(xlabel="ln previsto", ylabel="resíduo", title="Resíduo × previsto (sem estrutura = bom)")
plt.show()

# %% [markdown]
# **O que vemos.** A nuvem previsto × real em log tem inclinação menor que a diagonal: as previsões ficam num intervalo estreito (ln previsto entre ≈ −2,6 e −1) enquanto o real vai de ≈ −4 a 1. É o que se espera de um R² baixo (≈ 0,23): o modelo acerta o nível esperado, mas não os extremos, e a dispersão vertical (o ruído que não dá para reduzir, já que o alvo é a volatilidade realizada de só 5 retornos) domina. Os resíduos em log ficam centrados perto de zero, com uma leve cauda à direita (dias em que a volatilidade real foi bem maior que a prevista, por eventos). O gráfico resíduo × previsto não mostra estrutura forte, então o modelo não erra de forma sistemática em algum nível de volatilidade. O viés negativo em unidades naturais (≈ −0,02) vem da cauda à direita, de prever a mediana e do choque de 2018.

# %% [markdown]
# ## 2.10 Checagem: e a direção (sobe/desce)? Quase aleatória
#
# Para justificar o alvo, fizemos uma checagem curta: com as mesmas 25 features, uma regressão logística tenta prever se o retorno dos próximos 5 pregões é positivo.

# %%
_fwd = np.log(sp["close"]).groupby(sp["ticker"], sort=False).shift(-HORIZON) - np.log(sp["close"])
dir_df = pd.DataFrame({"up": (_fwd > 0).astype(int)}, index=sp.index).loc[sel.index]
d_tr, d_te = dir_df.loc[parts["train"].index, "up"].to_numpy(), dir_df.loc[parts["test"].index, "up"].to_numpy()
lr_dir = LogisticRegression(max_iter=300).fit(vZ_train, d_tr)
acc_dir = float((lr_dir.predict(v_norm(vX_test).numpy()) == d_te).mean())
print(f"direção em 5 pregões, teste: regressão logística = {acc_dir:.3%} | sempre 'sobe' = {d_te.mean():.3%} | acaso = 50%")
print("-> a direção não é prevista de forma útil com essas features (o ganho é nulo ou ínfimo); a volatilidade, sim, é previsível.")

# %% [markdown]
# ## 2.11 Previsões em novas instâncias: os 5 últimos pregões (ainda sem alvo)
#
# Os 5 últimos pregões do dataset (2018-02-01 a 2018-02-07) ainda não têm alvo (faltam os 5 dias seguintes), então não fazem parte do teste, que termina 5 pregões antes. São previsões de verdade, fora da amostra. Sorteamos 12 tickers com seed fixa, entre os que têm janela válida nos 5 dias.

# %%
new_dates = new_instance_dates(sp["date"])
new_rows = vol_df[vol_df["date"].isin(new_dates) & vol_df["valid_feat"]]
_complete = new_rows.groupby("ticker").size()
candidates = _complete[_complete == len(new_dates)].index
vol_new_tickers = pick_new_tickers(candidates, k=12, seed=7)
vol_new = new_rows[new_rows["ticker"].isin(vol_new_tickers)].sort_values(["ticker", "date"]).copy()
vX_new = vol_feature_matrix(vol_new)
vol_new["previsão (vol a.a.)"] = vol_infer.predict(vX_new, verbose=0).reshape(-1)
vol_new["rv_5 (últimos 5 dias)"] = vol_new["rv_5"]
print(f"{len(candidates)} tickers com janela válida nos 5 últimos pregões | sorteados (seed 7): {vol_new_tickers}")
vol_new["dia"] = vol_new["date"].dt.strftime("%m-%d")
pivot = vol_new.pivot(index="ticker", columns="dia", values="previsão (vol a.a.)")
pivot["rv_5 em 02-07"] = vol_new[vol_new["date"] == new_dates[-1]].set_index("ticker")["rv_5"]
show(pivot)
print("Previsão = volatilidade anualizada esperada para os 5 pregões SEGUINTES a cada data (ex.: 0,25 = 25% a.a.). Sem rótulo: o futuro ainda não aconteceu.")

# %%
vol_results = {
    "width": VOL_WIDTH, "depth": VOL_DEPTH, "loss": VOL_LOSS, "huber_delta": HUBER_DELTA, "mu": v_mu, "sigma": v_sigma,
    "val_baselines": val_base.to_dict(orient="index"),
    "test": vol_test_table.to_dict(orient="index"),
    "by_month": by_month.to_dict(orient="index"),
    "by_regime": by_regime.to_dict(orient="index"),
    "tercile_edges": [float(e) for e in vol_tercile_edges],
    "tercile_acc": {"mlp": vol_baseline_acc_mlp, "har": vol_baseline_acc_har, "persistence": vol_baseline_acc_persist, "majority": vol_majority_acc},
    "epochs": len(vh), "train_seconds": vol_train_seconds, "direction_acc": acc_dir,
    "split": {k: str(v.date()) for k, v in split.bounds.items()},
    "sizes": {p: len(d) for p, d in parts.items()},
}
vol_val_scores = vol_scores(vy_val, vol_infer.predict(vX_val, batch_size=8192, verbose=0).reshape(-1))
vol_results["val"] = vol_val_scores
part_times["20_volatility"] = time.time() - _t_part
print(f"Parte 2 concluída em {part_times['20_volatility'] / 60:.1f} min.")
