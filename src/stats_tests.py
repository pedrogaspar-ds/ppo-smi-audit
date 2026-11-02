"""Formal Sharpe-difference tests for the defence (Ledoit & Wolf 2008 style).
HAC-studentized statistic, circular block bootstrap of the studentized statistic, Holm within families.
Light: numpy/pandas only (no torch import)."""
import json, math, numpy as np, pandas as pd
RES = "results"; TEST_N = 500
df = pd.read_pickle(f"{RES}/data.pkl"); test = df.iloc[-TEST_N:]
e1 = json.load(open(f"{RES}/E1_thesis_30seeds.json")); e2 = json.load(open(f"{RES}/E2_fixed_10seeds.json"))
summ = json.load(open(f"{RES}/summary.json")); rep = json.load(open(f"{RES}/representatives.json")); win = summ["test_window"]
n = 499
r = test["r_next"].to_numpy()[:n]; rf = test["rf_next"].to_numpy()[:n]
tr = test["r_cssmi_next"].to_numpy()[:n]; r6040 = 0.6 * tr + 0.4 * test["r_bond_next"].to_numpy()[:n]
COST = 0.001
def net(A, cash=True, cost=True):
    A = np.asarray(A, float); dA = np.abs(np.diff(np.concatenate([[0.0], A])))
    return A * r + ((1 - A) * rf if cash else 0.0) - (COST * dA if cost else 0.0)
rsi = test["RSI"].to_numpy()[:n]; Ar = np.zeros(n); inpos = False
for i in range(n):
    if not inpos and rsi[i] < 30: inpos = True
    elif inpos and rsi[i] > 70: inpos = False
    Ar[i] = 1.0 if inpos else 0.0
rsi_net = net(Ar)

def moments(a, b):
    ma, mb, ga, gb = a.mean(), b.mean(), (a * a).mean(), (b * b).mean()
    va, vb = ga - ma * ma, gb - mb * mb
    d = ma / math.sqrt(va) - mb / math.sqrt(vb)
    grad = np.array([ga / va ** 1.5, -gb / vb ** 1.5, -ma / (2 * va ** 1.5), mb / (2 * vb ** 1.5)])
    return d, grad, (ma, mb, ga, gb)
def hac(y, lag):
    T = len(y); S = y.T @ y / T
    for L in range(1, lag + 1):
        G = y[L:].T @ y[:-L] / T; S = S + (1 - L / (lag + 1)) * (G + G.T)
    return S
def lw_test(a, b, block=10, M=4999, seed=0):
    a = np.asarray(a, float); b = np.asarray(b, float); T = len(a)
    if a.std() == 0 or b.std() == 0: return None
    d, g, m = moments(a, b)
    y = np.column_stack([a - m[0], b - m[1], a * a - m[2], b * b - m[3]])
    lag = int(math.floor(4 * (T / 100) ** (2 / 9)))
    se = math.sqrt(g @ hac(y, lag) @ g / T); stat = abs(d) / se
    rng = np.random.default_rng(seed); l = int(math.ceil(T / block)); Tb = l * block; hits = 0
    for _ in range(M):
        idx = ((rng.integers(0, T, size=l)[:, None] + np.arange(block)[None, :]) % T).ravel()
        aa, bb = a[idx], b[idx]
        if aa.std() == 0 or bb.std() == 0: continue
        ds, gs, ms = moments(aa, bb)
        ys = np.column_stack([aa - ms[0], bb - ms[1], aa * aa - ms[2], bb * bb - ms[3]])
        z = ys.reshape(l, block, 4).sum(axis=1) / math.sqrt(block)
        ses = math.sqrt(gs @ (z.T @ z / l) @ gs / Tb)
        if abs(ds - d) / ses >= stat: hits += 1
    return dict(sharpe_diff_ann=d * math.sqrt(252), se_ann=se * math.sqrt(252), t=d / se, p=(hits + 1) / (M + 1), block=block)
def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj.tolist()

out = {"window": win, "method": "Ledoit-Wolf (2008)-style test of equal Sharpe ratios: HAC (Bartlett, Newey-West lag) studentized difference, circular block bootstrap of the studentized statistic, 4,999 resamples; Holm adjustment within each family"}
s1, s2 = rep["E1_rep_seed"], rep["E2_rep_seed"]
A1 = np.array(e1["actions"][f"thesis-s{s1}-{win}"]); A2 = np.array(e2["actions"][f"fixed-s{s2}-{win}"])
fam = {}
# Family 0: as in the thesis (gross, rf = 0), representative invested seed vs B&H price
fam["F0_thesis_accounting_seed%d_vs_BH_price_rf0" % s1] = {k: lw_test(net(A1, cash=False, cost=False), r, block=k) for k in (5, 10, 20)}
# Family 1: consistent accounting (excess over SARON), thesis-like seed vs three benchmarks
f1 = {"BH price": r - rf, "BH total return": tr - rf, "60/40 real ETFs": r6040 - rf}
res1 = {k: lw_test(net(A1) - rf, v) for k, v in f1.items()}
adj = holm([v["p"] for v in res1.values()]); [res1[k].__setitem__("p_holm", a) for k, a in zip(res1, adj)]
fam["F1_consistent_seed%d" % s1] = res1
# Family 2: consistent accounting, corrected agent median seed vs four comparators
f2 = {"BH price": r - rf, "BH total return": tr - rf, "60/40 real ETFs": r6040 - rf, "RSI 30/70 rule": rsi_net - rf}
res2 = {k: lw_test(net(A2) - rf, v) for k, v in f2.items()}
adj = holm([v["p"] for v in res2.values()]); [res2[k].__setitem__("p_holm", a) for k, a in zip(res2, adj)]
fam["F2_consistent_corrected_seed%d" % s2] = res2
out["families"] = fam
# Across seeds: how often would the H1 comparison reject at 5%?
def across(actions, cfg, gross):
    rows = []
    for k, A in actions.items():
        if not k.startswith(cfg) or not k.endswith(win): continue
        A = np.array(A); a = net(A, cash=False, cost=False) if gross else net(A) - rf; b = r if gross else r - rf
        t = lw_test(a, b, M=1999)
        rows.append(dict(seed=int(k.split("-s")[1].split("-")[0]), invests=bool((A > 0.01).mean() > 0), p=None if t is None else t["p"], diff=None if t is None else t["sharpe_diff_ann"]))
    return rows
E1g = across(e1["actions"], "thesis", gross=True); E2c = across(e2["actions"], "fixed", gross=False)
out["across_seeds"] = {
    "thesis_config_30_seeds_gross_rf0_vs_BH_price": dict(n=len(E1g), testable=sum(x["p"] is not None for x in E1g), reject_5pct=sum((x["p"] or 1) < 0.05 for x in E1g), rows=E1g),
    "corrected_10_seeds_consistent_vs_BH_price": dict(n=len(E2c), reject_5pct=sum((x["p"] or 1) < 0.05 for x in E2c), rows=E2c)}
json.dump(out, open(f"{RES}/stats_tests.json", "w"), indent=1, default=float)
def fmt(t): return "n/a (zero variance)" if t is None else f"diff {t['sharpe_diff_ann']:+.2f} (se {t['se_ann']:.2f}), p = {t['p']:.3f}" + (f", Holm {t['p_holm']:.3f}" if 'p_holm' in t else "")
for fk, fv in fam.items():
    print("==", fk)
    for k, v in fv.items(): print("  ", k, ":", fmt(v))
for k, v in out["across_seeds"].items(): print("==", k, {kk: vv for kk, vv in v.items() if kk != "rows"})
