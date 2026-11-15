"""Parte 10 - Rede 1: risco de credito (classificacao binaria desbalanceada), modelo ``credit-default-mlp``.

Variaveis que as partes seguintes (conclusao e exportacao) usam: credit_model, credit_results, credit_abl, credit_threshold,
CREDIT_FIELDS/CREDIT_FEATURES/EDU_VOCAB, credit_*_raw (registros crus) e credit_data_sha256.
"""

# %% [markdown]
# # Parte 1 · Risco de crédito: quem vai deixar de pagar? (classificação)
#
# **Problema.** Um banco quer estimar, para cada cliente de cartão de crédito, a probabilidade de ele ficar inadimplente no mês seguinte. Usamos o dataset Default of Credit Card Clients (UCI #350, Yeh 2009), com 30.000 clientes de Taiwan entre abril e setembro de 2005.
#
# O alvo é qualitativo (inadimplente ou não) e desbalanceado: só cerca de 22% dos clientes são inadimplentes. Por isso a acurácia sozinha engana (um modelo que diz sempre "paga" já acerta ~78%), e precisamos olhar também PR-AUC, F1 e KS, testar `class_weight` e escolher o limiar de decisão com um critério de custo.
#
# > Conteúdo educacional. Não é um modelo de decisão de crédito real.

# %% [markdown]
# ## 1.1 Dados, colunas e novas instâncias
#
# O arquivo tem 23 colunas preditoras (`X1..X23`) e o alvo `Y`. Renomeamos as colunas para nomes mais legíveis:
#
# | Coluna | Significado |
# |---|---|
# | `limit_bal` (X1) | limite de crédito (NT$) |
# | `sex` (X2), `marriage` (X4) | sexo (1 = M, 2 = F) e estado civil: **não entram como features** (seção 1.3) |
# | `education` (X3) | 1 pós-graduação, 2 universidade, 3 ensino médio, 4 outros (**0, 5 e 6 não são documentados**) |
# | `age` (X5) | idade |
# | `pay_1..pay_6` (X6–X11) | situação do pagamento de **set → abr/2005**: −1 = em dia, 1..9 = meses de atraso (**−2 e 0 não são documentados**) |
# | `bill_amt1..6` (X12–X17) | valor da fatura de set → abr |
# | `pay_amt1..6` (X18–X23) | valor pago de set → abr |
# | `default` (Y) | **alvo:** 1 = inadimplente no mês seguinte |
#
# Antes de qualquer análise, separamos 20 linhas reais (10 de cada classe) para usar como novas instâncias. Elas ficam fora da EDA, da remoção de duplicatas, da normalização e do treino, e só entram no final, como se fossem clientes novos.

# %%
import hashlib
import urllib.request

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

layers = keras.layers
_t_part = time.time()

CREDIT_URL = "https://archive.ics.uci.edu/static/public/350/data.csv"
credit_csv = DATA_DIR / "raw" / "uci350_default_credit.csv"
if not credit_csv.exists():
    request = urllib.request.Request(CREDIT_URL, headers={"User-Agent": "trinet-notebook (curso)"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        credit_csv.write_bytes(response.read())
credit_data_sha256 = hashlib.sha256(credit_csv.read_bytes()).hexdigest()

RENAME = {"X1": "limit_bal", "X2": "sex", "X3": "education_code", "X4": "marriage", "X5": "age", "Y": "default"}
RENAME |= {f"X{5 + m}": f"pay_{m}" for m in range(1, 7)}
RENAME |= {f"X{11 + m}": f"bill_amt{m}" for m in range(1, 7)}
RENAME |= {f"X{17 + m}": f"pay_amt{m}" for m in range(1, 7)}
credit_raw = pd.read_csv(credit_csv).drop(columns=["ID"]).rename(columns=RENAME)
print(f"{credit_raw.shape[0]:,} linhas x {credit_raw.shape[1]} colunas | NaN: {int(credit_raw.isna().sum().sum())} | SHA-256 do CSV: {credit_data_sha256[:16]}...")

# 20 novas instancias reais (10 por classe), sorteadas entre linhas que NAO tem duplicata no arquivo
_unique_rows = credit_raw[~credit_raw.duplicated(keep=False)]
_rng = np.random.default_rng(SEED)
_new_idx = np.concatenate([_rng.choice(_unique_rows.index[_unique_rows["default"] == c], size=10, replace=False) for c in (0, 1)])
credit_new = credit_raw.loc[_new_idx].copy()
credit_pool = credit_raw.drop(index=_new_idx)
print(f"novas instâncias reservadas: {len(credit_new)} (inadimplentes: {int(credit_new['default'].sum())}) | restam {len(credit_pool):,}")

# %% [markdown]
# ## 1.2 Remoção de duplicatas e análise exploratória (EDA)
#
# Sem a coluna `ID`, o dataset tem 35 linhas idênticas. Removemos essas duplicatas antes de dividir os dados, porque uma linha repetida no treino e no teste inflaria a métrica. Depois olhamos o balanceamento das classes, os códigos sem documentação, as correlações e a relação entre o atraso de pagamento e a inadimplência.

# %%
n_dup = int(credit_pool.duplicated().sum())
credit_df = credit_pool.drop_duplicates().reset_index(drop=True)
print(f"duplicatas removidas: {n_dup} | linhas após dedup: {len(credit_df):,}")
assert n_dup == 35

prevalence = credit_df["default"].mean()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
counts = credit_df["default"].value_counts().sort_index()
axes[0].bar(["adimplente (0)", "inadimplente (1)"], counts.to_numpy(), color=["#4c78a8", "#e45756"])
for i, v in enumerate(counts.to_numpy()):
    axes[0].text(i, v, f"{v:,}\n({v / len(credit_df):.1%})", ha="center", va="bottom")
axes[0].set_title("Balanceamento de classes")
axes[0].set_ylim(0, counts.max() * 1.18)
rate = credit_df.groupby("pay_1")["default"].agg(["mean", "size"])
axes[1].bar(rate.index.astype(str), rate["mean"], color="#e45756")
axes[1].axhline(prevalence, color="k", ls="--", lw=1, label=f"média geral ({prevalence:.1%})")
axes[1].set_xlabel("pay_1 (situação em set/2005; -2 e 0 não documentados)")
axes[1].set_ylabel("taxa de inadimplência")
axes[1].set_title("Atraso mais recente × inadimplência")
axes[1].legend()
plt.show()

# %% [markdown]
# **O que vemos.** O desbalanceamento (≈ 22% contra 78%) é o motivo para testar `class_weight` (na ablação da seção 1.5), usar PR-AUC e F1 e escolher o limiar pelo custo. O atraso mais recente (`pay_1`) é muito informativo: a taxa de inadimplência fica entre 13% e 17% para quem está em dia (−2, −1 e 0), sobe para ≈ 34% com 1 mês de atraso e para ≈ 69% com 2 meses. Por isso mantivemos `pay_*` como números, já que a ordem do atraso importa, inclusive os códigos −2 e 0, que não estão no dicionário de dados mas se comportam como "sem atraso".

# %%
EDU_LABEL = {0: "(não doc.)", 1: "pós-graduação", 2: "universidade", 3: "ensino médio", 4: "outros", 5: "(não doc.)", 6: "(não doc.)"}
EDU_MAP = {1: "graduate", 2: "university", 3: "high_school", 4: "other", 0: "other", 5: "other", 6: "other"}
EDU_VOCAB = ("graduate", "university", "high_school", "other")

edu_tab = credit_df.groupby("education_code")["default"].agg(linhas="size", inadimplencia="mean")
edu_tab.insert(0, "significado", [EDU_LABEL[c] for c in edu_tab.index])
edu_tab["vira"] = [EDU_MAP[c] for c in edu_tab.index]
print("EDUCATION: códigos 0, 5 e 6 não são documentados; 0, 4, 5 e 6 viram `other` (4 categorias):")
show(edu_tab)
mar = credit_df.groupby("marriage")["default"].agg(linhas="size", inadimplencia="mean")
print("MARRIAGE (não vira feature; 0 não é documentado):")
show(mar)
print(f"BILL_AMT1 negativo (crédito a favor do cliente): {int((credit_df['bill_amt1'] < 0).sum())} linhas, mantidas (é informação real)")
print("PAY_1 (valores):", credit_df["pay_1"].value_counts().sort_index().to_dict())

# %%
num_cols = ["limit_bal", "age"] + [f"pay_{m}" for m in range(1, 7)] + [f"bill_amt{m}" for m in range(1, 7)] + [f"pay_amt{m}" for m in range(1, 7)]
corr = credit_df[[*num_cols, "default"]].corr()
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, ax=axes[0], cbar_kws={"shrink": 0.7}, xticklabels=True, yticklabels=True)
axes[0].set_title("Correlação de Pearson (features numéricas e alvo)")
axes[0].tick_params(labelsize=7)
target_corr = corr["default"].drop("default").sort_values()
axes[1].barh(target_corr.index, target_corr.to_numpy(), color=np.where(target_corr.to_numpy() > 0, "#e45756", "#4c78a8"))
axes[1].set_title("Correlação com a inadimplência")
axes[1].tick_params(labelsize=7)
plt.show()
print("Maiores |correlação| com o alvo:", ", ".join(f"{k} ({v:+.2f})" for k, v in target_corr.reindex(target_corr.abs().sort_values(ascending=False).index).head(5).items()))

# %% [markdown]
# **O que vemos.** As variáveis `pay_1..pay_6` têm as maiores correlações com o alvo, e a mais recente (`pay_1`) é a mais forte. As seis faturas (`bill_amt*`) são muito correlacionadas entre si, porque a fatura de um mês é quase igual à do mês anterior. O limite de crédito tem correlação negativa moderada: quanto maior o limite, menor a inadimplência. Como as faturas são redundantes, criamos a razão de utilização `bill_amt1 / limit_bal`, uma medida que um banco realmente usaria.

# %% [markdown]
# ## 1.3 Preparação dos dados
#
# **Variáveis preditoras e alvo.** O alvo é `default`. As 25 features são:
# - 20 colunas numéricas usadas como estão: `limit_bal`, `age`, `pay_1..6`, `bill_amt1..6` e `pay_amt1..6`;
# - one-hot de `education` (4 categorias, com os códigos 0, 4, 5 e 6 juntos em `other`);
# - a razão de utilização `bill_utilisation = bill_amt1 / limit_bal` (o menor limite do dataset é 10.000, então não há divisão por zero).
#
# **`SEX` e `MARRIAGE` ficam fora das features.** Em crédito, usar sexo ou estado civil na decisão é discriminação por atributo protegido (a Regulation B / ECOA, nos EUA, proíbe os dois, e a LGPD, art. 20, dá ao titular o direito de pedir a revisão de decisões automatizadas; vale conferir o texto da lei antes de citar). Mesmo assim o modelo pode reproduzir esses grupos por outras variáveis, como `age` e `limit_bal`. Por isso guardamos `sex` apenas para o diagnóstico de equidade no teste (seção 1.9).
#
# **Divisão 70/15/15** (treino, validação e teste), estratificada, para manter a mesma proporção de inadimplentes nas três partes. O teste só é usado uma vez, no final.

# %%
CREDIT_NUMERIC = num_cols
CREDIT_FEATURES = [*CREDIT_NUMERIC, *[f"education={v}" for v in EDU_VOCAB], "bill_utilisation"]
N_CREDIT = len(CREDIT_FEATURES)


def credit_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Linhas cruas -> matriz float32 de 25 colunas (mesma ordem do manifesto: passthrough, one-hot, razão)."""
    edu = frame["education_code"].map(EDU_MAP)
    onehot = np.stack([(edu == v).to_numpy(dtype="float64") for v in EDU_VOCAB], axis=1)
    util = (frame["bill_amt1"] / frame["limit_bal"]).to_numpy(dtype="float64")[:, None]
    return np.concatenate([frame[CREDIT_NUMERIC].to_numpy(dtype="float64"), onehot, util], axis=1).astype("float32")


idx_train, idx_rest = train_test_split(credit_df.index, test_size=0.30, stratify=credit_df["default"], random_state=SEED)
idx_val, idx_test = train_test_split(idx_rest, test_size=0.50, stratify=credit_df.loc[idx_rest, "default"], random_state=SEED)
credit_part = {"treino": idx_train, "validação": idx_val, "teste": idx_test}
print(pd.DataFrame({k: {"linhas": len(v), "inadimplentes": int(credit_df.loc[v, "default"].sum()), "taxa": credit_df.loc[v, "default"].mean()} for k, v in credit_part.items()}).T.to_string())

X_train, X_val, X_test = (credit_feature_matrix(credit_df.loc[i]) for i in (idx_train, idx_val, idx_test))
y_train, y_val, y_test = (credit_df.loc[i, "default"].to_numpy() for i in (idx_train, idx_val, idx_test))
X_new = credit_feature_matrix(credit_new)
y_new = credit_new["default"].to_numpy()
print(f"X_train {X_train.shape} | X_val {X_val.shape} | X_test {X_test.shape} | novas {X_new.shape}")

# %% [markdown]
# ## 1.4 Normalização (obrigatória)
#
# As colunas têm escalas muito diferentes: `limit_bal` está na casa de 10⁵, `pay_*` vai de −2 a 8 e a razão de utilização fica entre 0 e ~1. Redes treinadas por gradiente funcionam melhor com entradas de média 0 e desvio 1. Usamos a camada `tf.keras.layers.Normalization`: o `adapt()` calcula a média e a variância só com o treino (para não vazar informação da validação e do teste), e a camada fica dentro do modelo.

# %%
norm_layer = layers.Normalization(axis=-1, name="norm")
norm_layer.adapt(X_train)
norm_mean = np.asarray(norm_layer.mean).reshape(-1)
norm_var = np.asarray(norm_layer.variance).reshape(-1)
Z_train = norm_layer(X_train).numpy()
Z_val = norm_layer(X_val).numpy()
norm_table = pd.DataFrame(
    {
        "média (antes)": X_train.mean(axis=0),
        "desvio (antes)": X_train.std(axis=0),
        "média (depois)": Z_train.mean(axis=0),
        "desvio (depois)": Z_train.std(axis=0),
        "média val. (depois)": Z_val.mean(axis=0),
    },
    index=CREDIT_FEATURES,
)
show(norm_table)
fig, axes = plt.subplots(2, 2, figsize=(9, 5))
for j, name in enumerate(["limit_bal", "bill_amt1"]):
    k = CREDIT_FEATURES.index(name)
    axes[0, j].hist(X_train[:, k], bins=60, color="#4c78a8")
    axes[0, j].set_title(f"{name}: antes")
    axes[1, j].hist(Z_train[:, k], bins=60, color="#54a24b")
    axes[1, j].set_title(f"{name}: depois (média 0, desvio 1)")
plt.show()

# %% [markdown]
# **O que vemos.** Depois do `adapt`, todas as colunas do treino têm média ≈ 0 e desvio ≈ 1. Na validação a média fica perto de 0, mas não exatamente, e isso é esperado, porque os parâmetros vêm só do treino. A forma das distribuições não muda (continuam assimétricas, com cauda longa): a normalização só muda a escala. As outras preparações foram a remoção de duplicatas, a junção das categorias raras de `education`, o one-hot e a razão de utilização. Não há valores ausentes, então não precisamos de imputação.

# %% [markdown]
# ## 1.5 Ablação: largura × profundidade × `class_weight` (3 seeds cada)
#
# Em vez de usar um tuner, fizemos uma grade pequena, para ficar claro como escolhemos:
# 1. **Largura × profundidade:** largura da 1ª camada oculta em {16, 32, 64, 128} × profundidade em {1, 2, 3} camadas ocultas (cada camada tem metade dos neurônios da anterior, com `Dropout(0,2)` depois de cada uma), usando `class_weight` balanceado e 3 seeds por configuração.
# 2. **Com × sem `class_weight`** na arquitetura escolhida no item 1, também com 3 seeds.
#
# O treino usa Adam (1e-3), lotes de 512 e early stopping na PR-AUC de validação (a métrica que mais importa numa classe rara), restaurando os melhores pesos. A decisão sai da validação: PR-AUC como critério principal e F1 como desempate. Entre configurações com PR-AUC parecida (diferença menor que 1 erro-padrão da melhor), ficamos com a mais simples.

# %%
Y_train_oh = keras.utils.to_categorical(y_train, 2)
Y_val_oh = keras.utils.to_categorical(y_val, 2)
_n = np.bincount(y_train)
class_weight = {0: len(y_train) / (2 * _n[0]), 1: len(y_train) / (2 * _n[1])}  # "balanceado": n / (2 n_c)
print(f"class_weight balanceado: {{0: {class_weight[0]:.3f}, 1: {class_weight[1]:.3f}}}")


def build_credit_mlp(width: int, depth: int, dropout: float = 0.2, name: str = "credit_mlp") -> keras.Model:
    """Camada a camada: Input -> Normalization (estatística do treino) -> [Dense(relu) -> Dropout] x depth -> Dense(2, softmax)."""
    inputs = keras.Input(shape=(N_CREDIT,), name="features")
    x = layers.Normalization(axis=-1, mean=norm_mean, variance=norm_var, name="norm")(inputs)
    for i in range(depth):
        x = layers.Dense(max(8, width // 2**i), activation="relu", name=f"dense_{i}")(x)
        x = layers.Dropout(dropout, name=f"dropout_{i}")(x)
    outputs = layers.Dense(2, activation="softmax", name="out")(x)
    return keras.Model(inputs, outputs, name=name)


class ValidationScores(keras.callbacks.Callback):
    """Calcula PR-AUC e ROC-AUC na validação a cada época (entram em ``logs`` para o EarlyStopping)."""

    def __init__(self, x, y):
        super().__init__()
        self.x, self.y = x, y

    def on_epoch_end(self, epoch, logs=None):
        p = self.model(self.x, training=False).numpy()[:, 1]
        logs["val_pr_auc"] = average_precision_score(self.y, p)
        logs["val_roc_auc"] = roc_auc_score(self.y, p)


def best_f1(y_true, p) -> tuple[float, float]:
    """Maior F1 sobre todos os limiares (e o limiar), para comparar modelos sem depender do corte 0,5."""
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    k = int(np.argmax(f1[:-1]))
    return float(f1[k]), float(thr[k])


def fit_credit(width: int, depth: int, seed: int, use_cw: bool, max_epochs: int = 40, patience: int = 6):
    set_seed(seed)
    model = build_credit_mlp(width, depth)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
    callbacks = [
        ValidationScores(X_val, y_val),
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=patience, restore_best_weights=True),
    ]
    history = model.fit(
        X_train, Y_train_oh, validation_data=(X_val, Y_val_oh), epochs=max_epochs, batch_size=512,
        class_weight=class_weight if use_cw else None, callbacks=callbacks, verbose=0,
    )  # fmt: skip
    return model, history


def credit_run(width: int, depth: int, seed: int, use_cw: bool) -> dict:
    t1 = time.time()
    m, h = fit_credit(width, depth, seed, use_cw)
    p = m.predict(X_val, batch_size=4096, verbose=0)[:, 1]
    f1_best, _ = best_f1(y_val, p)
    return {"class_weight": use_cw, "depth": depth, "width": width, "seed": seed, "params": m.count_params(),
            "val_pr_auc": average_precision_score(y_val, p), "val_roc_auc": roc_auc_score(y_val, p),
            "val_f1_05": f1_score(y_val, p >= 0.5), "val_f1_best": f1_best, "epochs": len(h.history["loss"]), "seconds": time.time() - t1}  # fmt: skip


_rows = []
_t0 = time.time()
for depth in (1, 2, 3):
    for width in (16, 32, 64, 128):
        _rows += [credit_run(width, depth, seed, True) for seed in (1, 2, 3)]
    print(f"  profundidade {depth} pronta ({time.time() - _t0:.0f}s)", flush=True)
credit_abl_grid = pd.DataFrame(_rows)
keras.backend.clear_session()

# %%
agg = (
    credit_abl_grid.groupby(["depth", "width"])
    .agg(params=("params", "first"), pr_auc=("val_pr_auc", "mean"), pr_auc_std=("val_pr_auc", "std"), roc_auc=("val_roc_auc", "mean"),
         f1_05=("val_f1_05", "mean"), f1_best=("val_f1_best", "mean"), epochs=("epochs", "mean"))
    .reset_index()
)  # fmt: skip
agg["se"] = agg["pr_auc_std"] / np.sqrt(3)
top = agg.loc[agg["pr_auc"].idxmax()]
near = agg[agg["pr_auc"] >= top["pr_auc"] - top["se"]]  # regra do 1 erro-padrão: a configuração mais simples dentro do ruído da melhor
chosen = near.sort_values(["params", "pr_auc"], ascending=[True, False]).iloc[0]
CREDIT_WIDTH, CREDIT_DEPTH = int(chosen["width"]), int(chosen["depth"])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, col, scol, title in zip(axes, ("pr_auc", "f1_05"), ("pr_auc_std", None), ("PR-AUC na validação", "F1 (limiar 0,5) na validação"), strict=True):
    mean = agg.pivot(index="depth", columns="width", values=col)
    if scol:
        std = agg.pivot(index="depth", columns="width", values=scol)
        labels = mean.map("{:.4f}".format) + "\n±" + std.map("{:.4f}".format)
    else:
        labels = mean.map("{:.3f}".format)
    sns.heatmap(mean, annot=labels, fmt="", cmap="viridis", ax=ax, cbar=False)
    ax.set_title(f"{title}\n(média de 3 seeds, class_weight balanceado)")
    ax.set_xlabel("largura da 1ª camada oculta")
    ax.set_ylabel("nº de camadas ocultas")
plt.show()
show(agg.sort_values("pr_auc", ascending=False).head(6))
print(f"melhor PR-AUC: {top['pr_auc']:.4f} (largura {int(top['width'])}, profundidade {int(top['depth'])}, EP {top['se']:.4f}) -> regra do 1 EP: largura {CREDIT_WIDTH}, profundidade {CREDIT_DEPTH} ({int(chosen['params']):,} parâmetros)")

# %%
# class_weight: com x sem, na arquitetura escolhida (3 seeds cada). A configuração "com" já foi treinada na grade.
_rows = [credit_run(CREDIT_WIDTH, CREDIT_DEPTH, seed, False) for seed in (1, 2, 3)]
keras.backend.clear_session()
credit_abl = pd.concat([credit_abl_grid, pd.DataFrame(_rows)], ignore_index=True)
cw_cmp = credit_abl[(credit_abl["width"] == CREDIT_WIDTH) & (credit_abl["depth"] == CREDIT_DEPTH)].groupby("class_weight")[["val_pr_auc", "val_roc_auc", "val_f1_05", "val_f1_best"]].agg(["mean", "std"])
show(cw_cmp)
_pr = credit_abl[(credit_abl["width"] == CREDIT_WIDTH) & (credit_abl["depth"] == CREDIT_DEPTH)].groupby("class_weight")
_m, _s = _pr["val_pr_auc"].mean(), _pr["val_pr_auc"].std() / np.sqrt(3)
_tie = abs(_m[True] - _m[False]) <= max(_s[True], _s[False])  # empate em PR-AUC (dentro de 1 EP) -> desempata pelo F1 em 0,5
_f1 = _pr["val_f1_05"].mean()
use_class_weight = bool(_f1[True] >= _f1[False]) if _tie else bool(_m[True] > _m[False])
print(f"PR-AUC: com {_m[True]:.4f} × sem {_m[False]:.4f} | empate dentro de 1 EP: {_tie} | F1(0,5): com {_f1[True]:.4f} × sem {_f1[False]:.4f}")
print(f"DECISÃO: class_weight = {use_class_weight} | 1ª camada = {CREDIT_WIDTH} neurônios | {CREDIT_DEPTH} camada(s) oculta(s)")
print(f"{len(credit_abl)} treinos na ablação, {credit_abl['seconds'].sum() / 60:.1f} min no total")

# %% [markdown]
# **O que vemos na ablação.** A PR-AUC de validação varia pouco em toda a grade (de ≈ 0,536 a ≈ 0,549, cerca de 1,3 ponto percentual). O problema é tabular, tem poucas colunas e o sinal está concentrado nos `pay_*`, então redes pequenas já aproveitam quase tudo. A largura da primeira camada ajuda um pouco (de 16 para 128 neurônios) e a profundidade quase nada: o ganho de 2 para 3 camadas está dentro da variação entre seeds. Por isso ficamos com a configuração mais simples dentro de 1 erro-padrão da melhor, que também tem menos risco de overfitting. Sobre o `class_weight`, a PR-AUC de validação ficou mais alta sem ele (0,556 contra 0,548), enquanto o F1 em 0,5 favorece o `class_weight` só porque ele empurra as probabilidades para cima (o mesmo efeito se consegue mexendo no limiar, seção 1.8).

# %% [markdown]
# ## 1.6 Modelo final, camada a camada, e justificativas
#
# ```
# Input(25) → Normalization → [Dense(w, relu) → Dropout(0,2)] × profundidade → Dense(2, softmax)
# ```
#
# A ablação escolheu 2 camadas ocultas, com 128 e 64 neurônios.
#
# ### Justificativa: Arquitetura (Rede 1)
# - **Ativação das camadas ocultas: ReLU.** Escolhemos a ReLU porque ela é simples, não satura para valores positivos (evita o gradiente que some na sigmoid e na tanh), converge rápido e é o padrão em MLPs tabulares. Como a entrada já está normalizada, ela funciona bem, e o `Dropout(0,2)` ajuda a regularizar. A softmax só faz sentido na saída.
# - **Profundidade e largura:** usamos a grade da seção 1.5 (heatmap, 3 seeds) e a regra do 1 erro-padrão, que escolheram 2 camadas ocultas, com 128 e 64 neurônios. Redes maiores ganharam pouco (≈ 1 ponto percentual de PR-AUC entre a menor e a maior), então ficamos com a menor rede que não se distingue da melhor.
# - **Saída: `Dense(2, softmax)`.** As duas classes são exclusivas, e a softmax devolve probabilidades que somam 1 (`p(adimplente)` e `p(inadimplente)`), combinando com a `categorical_crossentropy`. Uma saída única com sigmoid seria equivalente, mas usamos dois neurônios para combinar com a matriz de confusão e com os rótulos one-hot.
#
# ### Justificativa: Compilação (Rede 1)
# - **Loss: `categorical_crossentropy`** (rótulos one-hot). É a log-verossimilhança das probabilidades previstas, pune bastante as previsões erradas feitas com muita confiança e dá probabilidades que servem para escolher o limiar e checar a calibração. Para o desbalanceamento (≈ 22%) testamos duas coisas: `class_weight` balanceado (ablação da seção 1.5) e o limiar de decisão por custo (seção 1.8). Na validação o `class_weight` não melhorou a PR-AUC e deslocou as probabilidades para cima. Como o limiar por custo já resolve o equilíbrio entre precisão e revocação, o modelo final é treinado sem `class_weight`.
# - **Taxa de aprendizado: Adam com 1e-3.** É o valor padrão do Adam e funciona bem em MLPs com entrada normalizada. As curvas (seção 1.7) descem de forma suave, sem oscilar, e o early stopping na PR-AUC de validação interrompe o treino antes do overfitting, restaurando os melhores pesos.
# - **Métricas:** acompanhamos a acurácia, só como referência (≈ 78% é o que se acerta dizendo sempre "adimplente"), e principalmente a PR-AUC (área sob a curva precisão × revocação). Ela não depende do limiar, não é inflada pela classe majoritária e mede o que importa numa classe rara: achar os inadimplentes sem muitos alarmes falsos. F1 e ROC-AUC complementam, e KS e Brier entram na avaliação final.

# %%
set_seed(SEED)
credit_model = build_credit_mlp(CREDIT_WIDTH, CREDIT_DEPTH, name="credit_default_mlp")
credit_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
credit_model.summary()

# %% [markdown]
# ## 1.7 Treino final e curvas
#
# Treinamos com os dados de treino e acompanhamos a validação. O `EarlyStopping` (PR-AUC de validação, paciência 10) restaura os melhores pesos.

# %%
_t1 = time.time()
credit_hist = credit_model.fit(
    X_train, Y_train_oh, validation_data=(X_val, Y_val_oh), epochs=100, batch_size=512,
    class_weight=class_weight if use_class_weight else None,
    callbacks=[ValidationScores(X_val, y_val), keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=10, restore_best_weights=True)],
    verbose=0,
)  # fmt: skip
credit_train_seconds = time.time() - _t1
hist = pd.DataFrame(credit_hist.history)
best_epoch = int(hist["val_pr_auc"].idxmax()) + 1
print(f"{len(hist)} épocas em {credit_train_seconds:.0f}s | melhor época (PR-AUC de validação): {best_epoch} | PR-AUC val = {hist['val_pr_auc'].max():.4f}")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
axes[0].plot(hist["loss"], label="treino")
axes[0].plot(hist["val_loss"], label="validação")
axes[0].set_title("Loss (categorical_crossentropy)")
axes[1].plot(hist["accuracy"], label="treino")
axes[1].plot(hist["val_accuracy"], label="validação")
axes[1].set_title("Acurácia")
axes[2].plot(hist["val_pr_auc"], color="C2", label="PR-AUC validação")
axes[2].plot(hist["val_roc_auc"], color="C3", label="ROC-AUC validação")
axes[2].set_title("Métricas de validação")
for ax in axes:
    ax.axvline(best_epoch - 1, color="k", ls=":", lw=1)
    ax.set_xlabel("época")
    ax.legend()
plt.show()

# %% [markdown]
# **O que vemos.** A loss cai rápido no começo e de forma suave, sem oscilar com o Adam 1e-3. Depois, a loss de treino continua caindo devagar enquanto a de validação estaciona em torno de 0,43, e a distância entre as duas cresce um pouco: é um sinal de overfitting leve (a rede é pequena e tem `Dropout`). A acurácia de validação fica em ≈ 0,82, acima do piso de 78% da classe majoritária, e a ROC-AUC e a PR-AUC de validação ficam quase planas depois de ≈ 20 épocas. A linha pontilhada marca a melhor época, cujos pesos o `EarlyStopping` restaura. Como a diferença entre as últimas épocas é muito pequena, o ponto exato de parada pouco importa.

# %% [markdown]
# ## 1.8 Limiar de decisão pelo custo (escolhido na validação)
#
# O corte 0,5 não tem nada de especial quando as classes são desbalanceadas. Em crédito os dois erros custam diferente: aprovar quem não paga (falso negativo, FN) custa bem mais do que recusar um bom cliente (falso positivo, FP). Como suposição de exemplo (não é um custo real de banco), usamos custo(FN) = 5 × custo(FP) e escolhemos, na validação, o limiar que minimiza o custo total. O teste só entra depois, com o limiar já fixado.

# %%
COST_FN, COST_FP = 5.0, 1.0  # premissa ilustrativa: um inadimplente aprovado custa 5x um bom cliente recusado


def expected_cost(y_true, p, threshold) -> float:
    pred = p >= threshold
    fn = int(((~pred) & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    return (COST_FN * fn + COST_FP * fp) / len(y_true)


p_val = credit_model.predict(X_val, batch_size=4096, verbose=0)[:, 1]
grid = np.round(np.arange(0.02, 0.98, 0.005), 3)
val_costs = np.array([expected_cost(y_val, p_val, t) for t in grid])
credit_threshold = float(np.median(grid[val_costs == val_costs.min()]))
print(f"limiar por custo (validação): {credit_threshold:.3f} | custo/cliente: {expected_cost(y_val, p_val, credit_threshold):.4f}")
print(f"limiar 0,5 (validação):         0.500 | custo/cliente: {expected_cost(y_val, p_val, 0.5):.4f}")
print(f"aprovar todos (validação):             custo/cliente: {COST_FN * y_val.mean():.4f}")
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(grid, val_costs, color="C0")
ax.axvline(credit_threshold, color="C3", ls="--", label=f"mínimo = {credit_threshold:.2f}")
ax.axvline(0.5, color="k", ls=":", label="0,5")
ax.set_xlabel("limiar de decisão (recusa se p(inadimplente) ≥ limiar)")
ax.set_ylabel("custo por cliente (FN = 5, FP = 1)")
ax.set_title("Custo esperado na validação")
ax.legend()
plt.show()

# %% [markdown]
# ## 1.9 Avaliação no teste (usado uma única vez)
#
# Calculamos tudo de uma vez: a matriz de confusão (com a diagonal destacada) e o relatório de classificação nos dois limiares (0,5 e o escolhido por custo), além de ROC-AUC, PR-AUC, KS, Brier e a curva de calibração. Comparamos com dois baselines: a classe majoritária e uma regressão logística treinada no mesmo treino.

# %%
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET_NAMES = ["adimplente (0)", "inadimplente (1)"]
p_test = credit_model.predict(X_test, batch_size=4096, verbose=0)[:, 1]

# baselines treinados no MESMO treino; limiar do baseline logístico escolhido na validação pela mesma regra de custo
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced" if use_class_weight else None))
lr.fit(X_train, y_train)
p_lr_val, p_lr_test = lr.predict_proba(X_val)[:, 1], lr.predict_proba(X_test)[:, 1]
lr_costs = np.array([expected_cost(y_val, p_lr_val, t) for t in grid])
lr_threshold = float(np.median(grid[lr_costs == lr_costs.min()]))


def ks_statistic(y_true, p) -> float:
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(tpr - fpr))


def score_row(name, y_true, p, threshold) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "modelo": name, "limiar": threshold, "acurácia": (tp + tn) / len(y_true),
        "precisão (1)": tp / max(tp + fp, 1), "revocação (1)": tp / max(tp + fn, 1), "F1 (1)": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, p) if np.ptp(p) > 0 else 0.5, "PR-AUC": average_precision_score(y_true, p),
        "KS": ks_statistic(y_true, p) if np.ptp(p) > 0 else 0.0, "Brier": brier_score_loss(y_true, p),
        "custo/cliente": expected_cost(y_true, p, threshold),
    }  # fmt: skip


p_major = np.zeros(len(y_test))  # classe majoritária: sempre p(inadimplente) = 0
rows = [
    score_row("maioria (sempre adimplente)", y_test, p_major, 0.5),
    score_row("regressão logística (0,5)", y_test, p_lr_test, 0.5),
    score_row("regressão logística (custo)", y_test, p_lr_test, lr_threshold),
    score_row("MLP (0,5)", y_test, p_test, 0.5),
    score_row("MLP (limiar por custo)", y_test, p_test, credit_threshold),
]
credit_test_table = pd.DataFrame(rows).set_index("modelo")
show(credit_test_table)
print(f"custo/cliente se aprovar todos: {COST_FN * y_test.mean():.4f} (FN = {COST_FN:g} × FP)")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, thr, title in zip(axes, (0.5, credit_threshold), ("limiar 0,5", f"limiar por custo ({credit_threshold:.2f})"), strict=True):
    cm = confusion_matrix(y_test, (p_test >= thr).astype(int), labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i, j]:,}\n({pct[i, j]:.1%} da classe real)" for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax, xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, linewidths=1, linecolor="white")
    for i in range(2):  # destaca a diagonal (acertos)
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="#e45756", lw=3))
    ax.set_xlabel("classe prevista")
    ax.set_ylabel("classe real")
    ax.set_title(f"MLP no teste, {title}\n(diagonal destacada = acertos)")
plt.show()
for thr, title in ((0.5, "limiar 0,5"), (credit_threshold, f"limiar por custo ({credit_threshold:.2f})")):
    print(f"--- Relatório de classificação (teste, {title}) ---")
    print(classification_report(y_test, (p_test >= thr).astype(int), target_names=TARGET_NAMES, digits=3))

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for name, p, color in (("MLP", p_test, "C0"), ("regressão logística", p_lr_test, "C1")):
    fpr, tpr, _ = roc_curve(y_test, p)
    axes[0].plot(fpr, tpr, color=color, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
    prec, rec, _ = precision_recall_curve(y_test, p)
    axes[1].plot(rec, prec, color=color, label=f"{name} (AP {average_precision_score(y_test, p):.3f})")
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
    axes[2].plot(mean_pred, frac_pos, "o-", color=color, label=f"{name} (Brier {brier_score_loss(y_test, p):.3f})")
axes[0].plot([0, 1], [0, 1], "k:")
axes[0].set(xlabel="taxa de falsos positivos", ylabel="revocação", title="Curva ROC")
axes[1].axhline(y_test.mean(), color="k", ls=":", label=f"acaso ({y_test.mean():.2f})")
axes[1].set(xlabel="revocação", ylabel="precisão", title="Curva precisão × revocação")
axes[2].plot([0, 1], [0, 1], "k:", label="calibração perfeita")
axes[2].set(xlabel="probabilidade prevista (média do grupo)", ylabel="fração real de inadimplentes", title="Calibração (10 grupos por quantil)")
for ax in axes:
    ax.legend()
plt.show()

# %% [markdown]
# **O que vemos.**
# - A classe majoritária acerta ~78%, mas nunca detecta um inadimplente (revocação 0). Isso mostra por que a acurácia engana.
# - A regressão logística é um baseline forte em dados tabulares pequenos. No teste, o MLP fica à frente nas métricas que não dependem do limiar (ROC-AUC ≈ 0,77 contra 0,72, PR-AUC ≈ 0,54 contra 0,49 e KS ≈ 0,42 contra 0,37). É um ganho moderado, mas consistente: a rede ajuda, sem milagre (a PR-AUC de 0,54 ainda está longe de 1).
# - O limiar por custo troca precisão por revocação. Como o FN custa 5 vezes o FP, vale aceitar mais alarmes falsos para deixar escapar menos inadimplentes (a revocação vai de ≈ 0,38 para ≈ 0,76 e a precisão cai). A coluna `custo/cliente` mostra o ganho: o MLP no limiar por custo (≈ 0,57) é melhor que o MLP em 0,5 (≈ 0,73), que a regressão logística no limiar dela (≈ 0,64) e que aprovar todos (≈ 1,11).
# - Na calibração (painel da direita), as probabilidades do MLP ficam perto da diagonal, e o Brier do MLP (≈ 0,137) é menor que o da regressão logística (≈ 0,146). Se tivéssemos usado `class_weight`, as probabilidades ficariam infladas e não poderiam ser lidas como risco em %. Nos grupos de maior risco há poucos clientes, então a curva se afasta um pouco da diagonal e essa parte é ruidosa.

# %% [markdown]
# ### Diagnóstico de equidade por sexo (no teste)
#
# `sex` não é feature, mas o modelo ainda pode tratar os grupos de forma diferente por causa de outras variáveis. Comparamos homens e mulheres no teste, com o limiar por custo, usando a taxa de aprovação (fração com p abaixo do limiar), a TPR (inadimplentes que o modelo consegue sinalizar) e a FPR (bons clientes sinalizados por engano). Ressalva: os dados são de Taiwan, de 2005, e o diagnóstico só compara taxas. Ele não prova nem descarta discriminação.

# %%
sex_test = credit_df.loc[idx_test, "sex"].map({1: "masculino", 2: "feminino"}).to_numpy()
pred_cost = (p_test >= credit_threshold).astype(int)
fair_rows = []
for grp in ("masculino", "feminino"):
    m = sex_test == grp
    tn, fp, fn, tp = confusion_matrix(y_test[m], pred_cost[m], labels=[0, 1]).ravel()
    fair_rows.append({"grupo": grp, "clientes": int(m.sum()), "inadimplência real": y_test[m].mean(), "taxa de aprovação": float((pred_cost[m] == 0).mean()),
                      "TPR (revocação)": tp / max(tp + fn, 1), "FPR": fp / max(fp + tn, 1), "PR-AUC": average_precision_score(y_test[m], p_test[m])})  # fmt: skip
fair = pd.DataFrame(fair_rows).set_index("grupo")
show(fair)
di = fair.loc["feminino", "taxa de aprovação"] / fair.loc["masculino", "taxa de aprovação"]
print(f"razão de aprovação (feminino/masculino): {di:.3f} | diferença de TPR: {fair.loc['feminino', 'TPR (revocação)'] - fair.loc['masculino', 'TPR (revocação)']:+.3f} | diferença de FPR: {fair.loc['feminino', 'FPR'] - fair.loc['masculino', 'FPR']:+.3f}")
print("Regra prática dos 4/5 (razão de aprovação >= 0,80):", "atendida" if di >= 0.8 else "NÃO atendida")

# %% [markdown]
# **O que vemos.** No teste, a taxa de aprovação das mulheres é ≈ 8% maior que a dos homens (razão ≈ 1,09, acima do piso de 0,80 da regra dos 4/5), e a TPR e a FPR delas são um pouco menores (diferenças de poucos pontos percentuais). Ou seja, o modelo recusa um pouco mais os homens, o que acompanha a diferença real de inadimplência nos dados (≈ 23,6% contra 21,1%). A PR-AUC é igual nos dois grupos (≈ 0,54): o modelo ranqueia os dois igualmente bem. A FPR (≈ 0,39 a 0,42) é alta nos dois por causa do limiar baixo escolhido por custo. Num uso real seria preciso medir também a calibração por grupo e testar formas de mitigar diferenças. Aqui o objetivo é medir e documentar.

# %% [markdown]
# ## 1.10 Previsões em novas instâncias (20 clientes reais separados no início)
#
# Essas 20 linhas nunca participaram da EDA, da normalização, do treino, da validação nem do teste. Aplicamos o modelo com o limiar por custo e comparamos com o resultado real.

# %%
p_new = credit_model.predict(X_new, verbose=0)[:, 1]
new_view = credit_new[["limit_bal", "age", "pay_1", "pay_2", "bill_amt1", "pay_amt1"]].copy()
new_view["p(inadimplente)"] = p_new
new_view["decisão"] = np.where(p_new >= credit_threshold, "recusar/limitar", "aprovar")
new_view["real"] = np.where(y_new == 1, "inadimplente", "adimplente")
new_view["acertou"] = (p_new >= credit_threshold).astype(int) == y_new
show(new_view.reset_index(drop=True))
print(f"acertos nas 20 novas instâncias: {int(new_view['acertou'].sum())}/20 (amostra pequena e balanceada 10/10: serve para ilustrar, não para estimar desempenho)")

# %%
credit_results = {
    "use_class_weight": use_class_weight,
    "width": CREDIT_WIDTH,
    "depth": CREDIT_DEPTH,
    "threshold": credit_threshold,
    "val_pr_auc": float(average_precision_score(y_val, p_val)),
    "val_roc_auc": float(roc_auc_score(y_val, p_val)),
    "val_f1_05": float(f1_score(y_val, p_val >= 0.5)),
    "test": credit_test_table.to_dict(orient="index"),
    "epochs": len(hist),
    "train_seconds": credit_train_seconds,
}
# registros crus (campos da API) para o apendice de exportacao
credit_api_fields = [*CREDIT_NUMERIC, "education"]


def credit_to_api(frame: pd.DataFrame) -> list[dict]:
    out = frame[CREDIT_NUMERIC].astype(float).copy()
    out["education"] = frame["education_code"].map(EDU_MAP)
    return out.to_dict(orient="records")


part_times["10_credit"] = time.time() - _t_part
print(f"Parte 1 concluída em {part_times['10_credit'] / 60:.1f} min.")
