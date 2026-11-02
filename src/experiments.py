"""Post-submission robustness experiments for the thesis defence.
Reproduces the Appendix A pipeline (ThesisEnv) and extends it:
  E1  30-seed replication of the thesis configuration + consistent-accounting re-evaluation
  E2  10-seed 'fixed' agent: turnover cost, cash earns SARON proxy, return-based state incl. previous exposure
  E3  walk-forward stress windows (2020, 2022) for both configurations
  E4  naive baselines, risk-free sensitivity, cost sweep, block-bootstrap CI
Outputs: results/*.csv|json, figures/*.png, results/summary.json
"""
import os, sys, json, math, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import yfinance as yf, ta
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from multiprocessing import Pool
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
torch.set_num_threads(1)
COST = 0.001
TEST_N = 500

# ---------------------------------------------------------------- data
SNB = [("2010-01-01", 0.0010), ("2011-08-03", 0.0000), ("2015-01-15", -0.0075), ("2022-06-16", -0.0025),
       ("2022-09-22", 0.0050), ("2022-12-15", 0.0100), ("2023-03-23", 0.0150), ("2023-06-22", 0.0175),
       ("2024-03-21", 0.0150), ("2024-06-20", 0.0125), ("2024-09-26", 0.0100), ("2024-12-12", 0.0050)]

def saron_proxy(index):
    s = pd.Series(np.nan, index=index)
    for d, r in SNB:
        s[s.index >= pd.Timestamp(d)] = r
    return s.ffill().bfill() / 252.0  # per trading day

def flat(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def load_data():
    p = os.path.join(RES, "data.pkl")
    if os.path.exists(p):
        return pd.read_pickle(p)
    smi = flat(yf.download("^SSMI", start="2010-01-01", end="2025-01-01", progress=False, auto_adjust=False))
    df = smi[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd = ta.trend.MACD(df["Close"]); df["MACD"] = macd.macd(); df["MACD_Signal"] = macd.macd_signal()
    df["ret1"] = df["Close"].pct_change(); df["ret5"] = df["Close"].pct_change(5); df["ret20"] = df["Close"].pct_change(20)
    df["vol20"] = df["ret1"].rolling(20).std(); df["sma200"] = df["Close"].rolling(200).mean()
    for tk, col in [("CSSMI.SW", "CSSMI"), ("CSBGC0.SW", "CSBGC0")]:
        e = flat(yf.download(tk, start="2010-01-01", end="2025-01-01", progress=False, auto_adjust=False))
        df[col] = e["Adj Close"].reindex(df.index).ffill()
    df["rf"] = saron_proxy(df.index)
    df = df.dropna(subset=["RSI", "MACD", "MACD_Signal"]).copy()  # thesis warm-up drop
    df["r_next"] = df["Close"].shift(-1) / df["Close"] - 1
    df["r_cssmi_next"] = df["CSSMI"].shift(-1) / df["CSSMI"] - 1
    df["r_bond_next"] = df["CSBGC0"].shift(-1) / df["CSBGC0"] - 1
    df["rf_next"] = df["rf"].shift(-1)
    df.to_pickle(p)
    return df

# ---------------------------------------------------------------- envs
class ThesisEnv(gym.Env):
    """Exact replica of Appendix A: obs = [Close/15000, RSI/100, MACD/100]; reward = A*dP - 0.001*A*P."""
    def __init__(self, df):
        super().__init__()
        self.close = df["Close"].to_numpy(); self.rsi = df["RSI"].to_numpy(); self.macd = df["MACD"].to_numpy()
        self.n = len(df); self.t = 0
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    def _obs(self):
        return np.array([self.close[self.t] / 15000.0, self.rsi[self.t] / 100.0, self.macd[self.t] / 100.0], dtype=np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.t = 0; return self._obs(), {}
    def step(self, action):
        a = float(action[0]); self.t += 1
        p, pp = self.close[self.t], self.close[self.t - 1]
        reward = a * (p - pp) - p * a * COST
        done = self.t >= self.n - 1
        return self._obs(), float(reward), done, False, {}

class FixedEnv(gym.Env):
    """Turnover cost, cash earns rf, stationary return-based state, previous exposure in state; reward in percent."""
    FEATS = ["ret1", "ret5", "ret20", "vol20", "RSI", "MACD", "Close"]
    def __init__(self, df, cost=COST):
        super().__init__()
        self.f = np.column_stack([df["ret1"] * 100, df["ret5"] * 100, df["ret20"] * 100, df["vol20"] * 100,
                                  df["RSI"] / 100.0, df["MACD"] / df["Close"] * 100]).astype(np.float32)
        self.r = df["r_next"].to_numpy(); self.rf = df["rf_next"].to_numpy(); self.cost = cost
        self.n = len(df); self.t = 0; self.prev = 0.0
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
    def _obs(self):
        return np.concatenate([self.f[self.t], [self.prev]]).astype(np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.t = 0; self.prev = 0.0; return self._obs(), {}
    def step(self, action):
        a = float(np.clip(action[0], 0, 1))
        reward = 100.0 * (a * self.r[self.t] + (1 - a) * self.rf[self.t] - self.cost * abs(a - self.prev))
        self.prev = a; self.t += 1
        done = self.t >= self.n - 1
        return self._obs(), float(reward), done, False, {}

def policy_actions(model, env):
    """Deterministic roll-out; returns exposure A_t for t = 0..n-2 (applied to the return from t to t+1)."""
    obs, _ = env.reset(); acts = []
    for _ in range(env.n - 1):
        a, _ = model.predict(obs, deterministic=True)
        a = float(np.clip(a[0], 0, 1)); acts.append(a)
        obs, _, done, _, _ = env.step(np.array([a], dtype=np.float32))
        if done: break
    return np.array(acts)

# ---------------------------------------------------------------- metrics
def metrics(ret, rf=None):
    ret = np.asarray(ret, dtype=float); rf = np.zeros_like(ret) if rf is None else np.asarray(rf, dtype=float)
    ex = ret - rf; sd = ret.std(ddof=1)
    w = np.cumprod(1 + ret); dd = w / np.maximum.accumulate(w) - 1
    out = dict(cum=float(w[-1] - 1), ann_ret=float((w[-1]) ** (252 / len(ret)) - 1), ann_vol=float(sd * math.sqrt(252)),
               sharpe_rf0=float(math.sqrt(252) * ret.mean() / sd) if sd > 0 else 0.0,
               sharpe_rf=float(math.sqrt(252) * ex.mean() / ex.std(ddof=1)) if ex.std(ddof=1) > 0 else 0.0,
               mdd=float(dd.min()))
    neg = ret[ret < 0]
    out["sortino_rf0"] = float(math.sqrt(252) * ret.mean() / neg.std(ddof=1)) if len(neg) > 1 and neg.std(ddof=1) > 0 else float("nan")
    out["calmar"] = float(out["ann_ret"] / abs(out["mdd"])) if out["mdd"] < 0 else float("nan")
    return out

def strategy_returns(A, r, rf, cost=COST, cash_yield=True, turnover_cost=True):
    A = np.asarray(A, float); dA = np.abs(np.diff(np.concatenate([[0.0], A])))
    ret = A * r + ((1 - A) * rf if cash_yield else 0.0) - (cost * dA if turnover_cost else 0.0)
    return ret, dA

def full_eval(A, test, label, cost=COST):
    r = test["r_next"].to_numpy()[: len(A)]; rf = test["rf_next"].to_numpy()[: len(A)]
    A = np.asarray(A, float)
    gross, dA = strategy_returns(A, r, rf, cost, cash_yield=False, turnover_cost=False)   # thesis convention
    net, _ = strategy_returns(A, r, rf, cost, cash_yield=True, turnover_cost=True)         # consistent convention
    cashonly, _ = strategy_returns(A, r, rf, cost, cash_yield=True, turnover_cost=False)
    mg, mn, mc = metrics(gross, rf), metrics(net, rf), metrics(cashonly, rf)
    rf25 = np.full_like(r, 0.025 / 252)
    row = dict(label=label,
               gross_cum=mg["cum"], gross_sharpe_rf0=mg["sharpe_rf0"], gross_sharpe_saron=mg["sharpe_rf"], gross_mdd=mg["mdd"], gross_vol=mg["ann_vol"],
               gross_sharpe_rf25=metrics(gross, rf25)["sharpe_rf"],
               cashcredit_cum=mc["cum"], cashcredit_sharpe_saron=mc["sharpe_rf"],
               net_cum=mn["cum"], net_sharpe_rf0=mn["sharpe_rf0"], net_sharpe_saron=mn["sharpe_rf"], net_mdd=mn["mdd"], net_vol=mn["ann_vol"],
               net_sortino=mn["sortino_rf0"], net_calmar=mn["calmar"],
               mean_exposure=float(A.mean()), pct_days_invested=float((A > 0.01).mean()), pct_days_full=float((A > 0.99).mean()),
               turnover=float(dA.sum()), n_trades=int((dA > 0.01).sum()), cost_paid=float((cost * dA).sum()))
    return row, gross, net

def block_bootstrap_sharpe_diff(ret_a, ret_b, rf, n=3000, block=20, seed=0):
    rng = np.random.default_rng(seed); T = len(ret_a); diffs = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, T - block + 1, size=T // block + 1)])[:T]
        a, b, f = ret_a[idx], ret_b[idx], rf[idx]
        sa = math.sqrt(252) * (a - f).mean() / (a - f).std(ddof=1) if (a - f).std(ddof=1) > 0 else 0
        sb = math.sqrt(252) * (b - f).mean() / (b - f).std(ddof=1) if (b - f).std(ddof=1) > 0 else 0
        diffs.append(sa - sb)
    d = np.array(diffs)
    return dict(mean=float(d.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), p_le0=float((d <= 0).mean()))

# ---------------------------------------------------------------- one training job
def train_job(args):
    cfg, seed, train_end, test_start, test_end, steps = args
    df = load_data()
    train = df.loc[:train_end].iloc[:-1] if train_end else df.iloc[:-TEST_N]
    test = df.loc[test_start:test_end] if test_start else df.iloc[-TEST_N:]
    train = train.dropna(subset=["r_next"]); test = test.copy()
    Env = ThesisEnv if cfg == "thesis" else FixedEnv
    if cfg == "fixed":
        train = train.dropna(subset=["ret20", "vol20"])
    t0 = time.time()
    env = Env(train.reset_index(drop=True))
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu")
    model.learn(total_timesteps=steps)
    A = policy_actions(model, Env(test.reset_index(drop=True)))
    row, gross, net = full_eval(A, test, f"{cfg}-s{seed}")
    row.update(cfg=cfg, seed=seed, window=f"{test.index[0].date()}..{test.index[-1].date()}", steps=steps, train_time=time.time() - t0)
    return row, A.tolist()

def run_pool(jobs, tag):
    p = os.path.join(RES, f"{tag}.json")
    if os.path.exists(p):
        return json.load(open(p))
    with Pool(min(8, os.cpu_count() or 4)) as pool:
        out = pool.map(train_job, jobs)
    rows = [o[0] for o in out]; acts = {f"{r['cfg']}-s{r['seed']}-{r['window']}": a for r, (_, a) in zip(rows, out)}
    json.dump({"rows": rows, "actions": acts}, open(p, "w"))
    return {"rows": rows, "actions": acts}

# ---------------------------------------------------------------- baselines
def baselines(test):
    r = test["r_next"].to_numpy(); rf = test["rf_next"].to_numpy(); n = len(r) - 1
    r, rf = r[:n], rf[:n]  # match agent horizon (T-1 returns)
    rows = {}
    def add(name, ret, A=None):
        m = metrics(ret, rf); m25 = metrics(ret, np.full_like(ret, 0.025 / 252))
        rows[name] = dict(cum=m["cum"], sharpe_rf0=m["sharpe_rf0"], sharpe_saron=m["sharpe_rf"], sharpe_rf25=m25["sharpe_rf"],
                          mdd=m["mdd"], vol=m["ann_vol"], sortino=m["sortino_rf0"], calmar=m["calmar"],
                          mean_exposure=float(np.mean(A)) if A is not None else None)
    add("Buy-and-Hold SMI (price index, as thesis)", r, np.ones(n))
    add("Buy-and-Hold SMI total return (CSSMI ETF adj.)", test["r_cssmi_next"].to_numpy()[:n], np.ones(n))
    add("60/40 stylized (thesis: 0.6 SMI + 0.4 x 0.01%/day)", 0.6 * r + 0.4 * 0.0001, np.full(n, 0.6))
    add("60/40 real ETFs (CSSMI / CSBGC0, daily rebalanced)", 0.6 * test["r_cssmi_next"].to_numpy()[:n] + 0.4 * test["r_bond_next"].to_numpy()[:n], np.full(n, 0.6))
    add("Cash stylized (0.01%/day, thesis)", np.full(n, 0.0001), np.zeros(n))
    add("Cash at SARON proxy", rf, np.zeros(n))
    for k in [0.10, 0.25, 0.50]:
        ret, _ = strategy_returns(np.full(n, k), r, rf, cash_yield=True, turnover_cost=True)
        add(f"Constant exposure {int(k*100)}% (cash at SARON, net)", ret, np.full(n, k))
    rsi = test["RSI"].to_numpy()[:n]; macd = test["MACD"].to_numpy()[:n]; sig = test["MACD_Signal"].to_numpy()[:n]
    close = test["Close"].to_numpy()[:n]; sma = test["sma200"].to_numpy()[:n]
    # RSI 30/70 state machine
    A = np.zeros(n); inpos = False
    for i in range(n):
        if not inpos and rsi[i] < 30: inpos = True
        elif inpos and rsi[i] > 70: inpos = False
        A[i] = 1.0 if inpos else 0.0
    ret, _ = strategy_returns(A, r, rf); add("Rule: RSI<30 buy / RSI>70 sell (net)", ret, A)
    A = (rsi > 50).astype(float); ret, _ = strategy_returns(A, r, rf); add("Rule: long when RSI>50 (net)", ret, A)
    A = (macd > sig).astype(float); ret, _ = strategy_returns(A, r, rf); add("Rule: long when MACD>signal (net)", ret, A)
    A = (macd > 0).astype(float); ret, _ = strategy_returns(A, r, rf); add("Rule: long when MACD>0 (net)", ret, A)
    A = (close > sma).astype(float); ret, _ = strategy_returns(A, r, rf); add("Rule: long when Close>SMA200 (net)", ret, A)
    rng = np.random.default_rng(0); rs = []
    for _ in range(500):
        A = rng.uniform(0, 1, n); ret, _ = strategy_returns(A, r, rf); rs.append(metrics(ret, rf))
    rows["Random exposure U(0,1), 500 draws (median; net)"] = dict(cum=float(np.median([m["cum"] for m in rs])), sharpe_rf0=float(np.median([m["sharpe_rf0"] for m in rs])),
        sharpe_saron=float(np.median([m["sharpe_rf"] for m in rs])), sharpe_rf25=None, mdd=float(np.median([m["mdd"] for m in rs])), vol=float(np.median([m["ann_vol"] for m in rs])),
        sortino=None, calmar=None, mean_exposure=0.5, sharpe_rf0_p95=float(np.percentile([m["sharpe_rf0"] for m in rs], 95)), mdd_p05=float(np.percentile([m["mdd"] for m in rs], 5)))
    return rows

# ---------------------------------------------------------------- main
def main():
    df = load_data()
    test = df.iloc[-TEST_N:].copy()
    print("data", df.index[0].date(), df.index[-1].date(), len(df), "| test", test.index[0].date(), test.index[-1].date(), flush=True)

    # E1: replication, 30 seeds
    e1 = run_pool([("thesis", s, None, None, None, 10_000) for s in range(30)], "E1_thesis_30seeds")
    print("E1 done", flush=True)
    # E2: fixed agent, 10 seeds x 200k
    e2 = run_pool([("fixed", s, None, None, None, 200_000) for s in range(10)], "E2_fixed_10seeds")
    print("E2 done", flush=True)
    # E3: walk-forward stress windows
    e3 = run_pool([("thesis", s, "2019-12-31", "2020-01-01", "2020-12-31", 10_000) for s in range(10)] +
                  [("thesis", s, "2021-12-31", "2022-01-01", "2022-12-31", 10_000) for s in range(10)] +
                  [("fixed", s, "2019-12-31", "2020-01-01", "2020-12-31", 200_000) for s in range(5)] +
                  [("fixed", s, "2021-12-31", "2022-01-01", "2022-12-31", 200_000) for s in range(5)], "E3_walkforward")
    print("E3 done", flush=True)

    base = baselines(test); json.dump(base, open(os.path.join(RES, "baselines.json"), "w"), indent=1)
    pd.DataFrame(base).T.to_csv(os.path.join(RES, "baselines.csv"))

    d1 = pd.DataFrame(e1["rows"]); d1.to_csv(os.path.join(RES, "E1_thesis_30seeds.csv"), index=False)
    d2 = pd.DataFrame(e2["rows"]); d2.to_csv(os.path.join(RES, "E2_fixed_10seeds.csv"), index=False)
    d3 = pd.DataFrame(e3["rows"]); d3.to_csv(os.path.join(RES, "E3_walkforward.csv"), index=False)

    # representative seeds (median gross Sharpe for E1, median net Sharpe for E2)
    def median_seed(d, col):
        return int(d.iloc[(d[col] - d[col].median()).abs().argsort().iloc[0]]["seed"])
    s1, s2 = median_seed(d1, "gross_sharpe_rf0"), median_seed(d2, "net_sharpe_saron")
    win = f"{test.index[0].date()}..{test.index[-1].date()}"
    A1 = np.array(e1["actions"][f"thesis-s{s1}-{win}"]); A2 = np.array(e2["actions"][f"fixed-s{s2}-{win}"])
    r = test["r_next"].to_numpy()[: len(A1)]; rf = test["rf_next"].to_numpy()[: len(A1)]

    # cost sweep on E2 actions (ex post) and on E1 actions
    sweep = []
    for c in [0.0, 0.0005, 0.001, 0.00175, 0.0025]:
        for name, A in [("thesis-median", A1), ("fixed-median", A2)]:
            ret, _ = strategy_returns(A, r, rf, cost=c); m = metrics(ret, rf)
            sweep.append(dict(agent=name, cost=c, cum=m["cum"], sharpe_saron=m["sharpe_rf"], mdd=m["mdd"]))
    pd.DataFrame(sweep).to_csv(os.path.join(RES, "cost_sweep.csv"), index=False)

    # bootstrap: agent vs BH (price) Sharpe difference, consistent accounting
    bh = r
    net1, _ = strategy_returns(A1, r, rf); net2, _ = strategy_returns(A2, r, rf); g1, _ = strategy_returns(A1, r, rf, cash_yield=False, turnover_cost=False)
    boot = dict(thesis_median_gross_vs_bh_rf0=block_bootstrap_sharpe_diff(g1, bh, np.zeros_like(rf)),
                thesis_median_net_vs_bh_saron=block_bootstrap_sharpe_diff(net1, bh, rf),
                fixed_median_net_vs_bh_saron=block_bootstrap_sharpe_diff(net2, bh, rf))
    json.dump(boot, open(os.path.join(RES, "bootstrap.json"), "w"), indent=1)

    # summary
    def q(d, col): return dict(median=float(d[col].median()), p10=float(d[col].quantile(.1)), p90=float(d[col].quantile(.9)), min=float(d[col].min()), max=float(d[col].max()), mean=float(d[col].mean()))
    summary = dict(test_window=win, n_test_returns=int(len(A1)),
                   thesis_reported=dict(sharpe=1.17, mdd=-0.0077, cum=0.0518),
                   E1=dict(n=len(d1), gross_sharpe_rf0=q(d1, "gross_sharpe_rf0"), gross_mdd=q(d1, "gross_mdd"), gross_cum=q(d1, "gross_cum"),
                           mean_exposure=q(d1, "mean_exposure"), pct_days_invested=q(d1, "pct_days_invested"), turnover=q(d1, "turnover"),
                           net_sharpe_saron=q(d1, "net_sharpe_saron"), net_cum=q(d1, "net_cum"), cashcredit_cum=q(d1, "cashcredit_cum"),
                           share_seeds_sharpe_above_BH=float((d1["gross_sharpe_rf0"] > base["Buy-and-Hold SMI (price index, as thesis)"]["sharpe_rf0"]).mean()),
                           share_seeds_always_cash=float((d1["pct_days_invested"] < 0.01).mean()), rep_seed=s1),
                   E2=dict(n=len(d2), net_sharpe_saron=q(d2, "net_sharpe_saron"), net_mdd=q(d2, "net_mdd"), net_cum=q(d2, "net_cum"), mean_exposure=q(d2, "mean_exposure"),
                           turnover=q(d2, "turnover"), pct_days_invested=q(d2, "pct_days_invested"), rep_seed=s2,
                           share_seeds_sharpe_above_BH=float((d2["net_sharpe_saron"] > base["Buy-and-Hold SMI (price index, as thesis)"]["sharpe_saron"]).mean())),
                   E3={w: {cfg: dict(n=int(len(g)), net_sharpe_saron=q(g, "net_sharpe_saron"), net_mdd=q(g, "net_mdd"), net_cum=q(g, "net_cum"), mean_exposure=q(g, "mean_exposure"))
                          for cfg, g in d3[d3.window == w].groupby("cfg")} for w in d3.window.unique()},
                   baselines=base, bootstrap=boot, cost_sweep=sweep)
    # BH stats in the stress windows
    for w in d3.window.unique():
        a, b = w.split(".."); tw = df.loc[a:b]; rr = tw["r_next"].to_numpy()[:-1]; ff = tw["rf_next"].to_numpy()[:-1]
        m = metrics(rr, ff); summary["E3"][w]["BH"] = dict(cum=m["cum"], sharpe_saron=m["sharpe_rf"], mdd=m["mdd"], vol=m["ann_vol"])
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=float)

    # ---------------------------------------------------------------- figures
    dates = test.index[: len(A1)]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    # F1 exposure path, thesis config (median seed) vs SMI
    fig, ax = plt.subplots(figsize=(11, 4.5)); ax2 = ax.twinx()
    ax.fill_between(dates, A1 * 100, 0, color="#1f5fa8", alpha=.25, step="mid"); ax.plot(dates, A1 * 100, color="#1f5fa8", lw=1.2, label="Agent exposure (%)")
    ax2.plot(dates, test["Close"].to_numpy()[: len(A1)], color="#555", lw=1, alpha=.8, label="SMI close")
    ax.set_ylabel("Exposure to SMI (%)"); ax2.set_ylabel("SMI level"); ax.set_ylim(0, 105)
    ax.set_title(f"Thesis configuration, seed {s1}: realised exposure path, out-of-sample {win}  |  mean exposure {A1.mean()*100:.1f}%, invested on {(A1>0.01).mean()*100:.0f}% of days")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F1_exposure_thesis_config.png"), dpi=200); plt.close(fig)

    # F2 seed distribution (E1) with thesis-reported values
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    bhm = base["Buy-and-Hold SMI (price index, as thesis)"]
    for ax, col, rep, bhv, title in [(axs[0], "gross_sharpe_rf0", 1.17, bhm["sharpe_rf0"], "Sharpe (rf = 0, gross)"),
                                     (axs[1], "gross_mdd", -0.0077, bhm["mdd"], "Max drawdown"), (axs[2], "gross_cum", 0.0518, bhm["cum"], "Cumulative return")]:
        v = d1[col].to_numpy() * (100 if col != "gross_sharpe_rf0" else 1)
        ax.boxplot(v, vert=True, widths=.5, showfliers=False); ax.scatter(np.random.default_rng(1).normal(1, .04, len(v)), v, s=14, color="#1f5fa8", alpha=.7, zorder=3)
        ax.axhline(rep * (100 if col != "gross_sharpe_rf0" else 1), color="#c0392b", ls="--", lw=1.2, label="thesis reported")
        ax.axhline(bhv * (100 if col != "gross_sharpe_rf0" else 1), color="#7f8c8d", ls=":", lw=1.2, label="Buy-and-Hold")
        ax.set_title(title); ax.set_xticks([]); 
        if col != "gross_sharpe_rf0": ax.set_ylabel("%")
    axs[0].legend(loc="lower right", fontsize=9); fig.suptitle("Thesis configuration re-trained with 30 random seeds (10,000 timesteps each), same test window, thesis accounting", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F2_seed_distribution.png"), dpi=200); plt.close(fig)

    # F3 cumulative returns: agents and baselines, consistent accounting (net, cash at SARON)
    fig, ax = plt.subplots(figsize=(11, 5))
    def cum(x): return (np.cumprod(1 + x) - 1) * 100
    ax.plot(dates, cum(g1), color="#1f5fa8", lw=2, label=f"Thesis config, seed {s1} (gross, cash 0%, as thesis)")
    ax.plot(dates, cum(net1), color="#1f5fa8", lw=1.4, ls="--", label=f"Thesis config, seed {s1} (net, cash at SARON)")
    ax.plot(dates, cum(net2), color="#e67e22", lw=2, label=f"Turnover-cost agent, seed {s2} (net, cash at SARON)")
    ax.plot(dates, cum(r), color="#555", lw=1.2, label="Buy-and-Hold SMI (price)")
    ax.plot(dates, cum(test["r_cssmi_next"].to_numpy()[: len(A1)]), color="#999", lw=1.2, ls="-.", label="Buy-and-Hold SMI total return (ETF)")
    ax.plot(dates, cum(rf), color="#27ae60", lw=1.2, ls=":", label="Cash at SARON proxy")
    ret_sma, _ = strategy_returns((test["Close"].to_numpy()[: len(A1)] > test["sma200"].to_numpy()[: len(A1)]).astype(float), r, rf)
    ax.plot(dates, cum(ret_sma), color="#8e44ad", lw=1.2, label="Rule: long when Close > SMA200 (net)")
    ax.set_ylabel("Cumulative return (%)"); ax.set_title(f"Out-of-sample {win}: agents vs benchmarks"); ax.legend(fontsize=8.5, loc="upper left"); ax.axhline(0, color="k", lw=.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F3_cumulative_returns.png"), dpi=200); plt.close(fig)

    # F4 risk-free sensitivity of Sharpe
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [f"Thesis config\nseed {s1} (gross)", "Buy-and-Hold\nSMI (price)", "60/40\nstylized"]
    vals = {"rf = 0": [metrics(g1)["sharpe_rf0"], bhm["sharpe_rf0"], base["60/40 stylized (thesis: 0.6 SMI + 0.4 x 0.01%/day)"]["sharpe_rf0"]],
            "rf = SARON proxy": [metrics(g1, rf)["sharpe_rf"], bhm["sharpe_saron"], base["60/40 stylized (thesis: 0.6 SMI + 0.4 x 0.01%/day)"]["sharpe_saron"]],
            "rf = 2.5% (thesis cash rate)": [metrics(g1, np.full_like(r, .025 / 252))["sharpe_rf"], bhm["sharpe_rf25"], base["60/40 stylized (thesis: 0.6 SMI + 0.4 x 0.01%/day)"]["sharpe_rf25"]]}
    x = np.arange(3); wdt = .26
    for i, (k, v) in enumerate(vals.items()):
        ax.bar(x + (i - 1) * wdt, v, wdt, label=k, color=["#1f5fa8", "#5dade2", "#aed6f1"][i])
        for xi, vi in zip(x + (i - 1) * wdt, v): ax.text(xi, vi + (0.02 if vi >= 0 else -0.08), f"{vi:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.axhline(0, color="k", lw=.6); ax.set_ylabel("Annualised Sharpe ratio"); ax.legend(fontsize=9)
    ax.set_title("The Sharpe ranking depends on the risk-free rate used")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F4_rf_sensitivity.png"), dpi=200); plt.close(fig)

    # F5 walk-forward stress windows: drawdown of median agents vs BH
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, w in zip(axs, sorted(d3.window.unique())):
        a, b = w.split(".."); tw = df.loc[a:b]; rr = tw["r_next"].to_numpy(); ff = tw["rf_next"].to_numpy()
        def ddc(x): w_ = np.cumprod(1 + x); return (w_ / np.maximum.accumulate(w_) - 1) * 100
        n = None
        for cfg, color, lab in [("thesis", "#1f5fa8", "Thesis config (median seed, net)"), ("fixed", "#e67e22", "Turnover-cost agent (median seed, net)")]:
            g = d3[(d3.window == w) & (d3.cfg == cfg)]
            if len(g) == 0: continue
            sd = int(g.iloc[(g["net_sharpe_saron"] - g["net_sharpe_saron"].median()).abs().argsort().iloc[0]]["seed"])
            A = np.array(e3["actions"][f"{cfg}-s{sd}-{w}"]); n = len(A)
            net, _ = strategy_returns(A, rr[:n], ff[:n]); ax.plot(tw.index[:n], ddc(net), color=color, lw=1.5, label=f"{lab}, mean exp. {A.mean()*100:.0f}%")
        n = n or len(rr) - 1
        ax.plot(tw.index[:n], ddc(rr[:n]), color="#555", lw=1.2, label="Buy-and-Hold SMI")
        ax.set_title(f"Stress window {a[:4]}: drawdown (%)"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F5_walkforward_stress.png"), dpi=200); plt.close(fig)

    # F6 fixed-agent exposure path
    fig, ax = plt.subplots(figsize=(11, 4)); ax2 = ax.twinx()
    ax.fill_between(dates, A2 * 100, 0, color="#e67e22", alpha=.25, step="mid"); ax.plot(dates, A2 * 100, color="#e67e22", lw=1.2, label="Agent exposure (%)")
    ax2.plot(dates, test["Close"].to_numpy()[: len(A2)], color="#555", lw=1, alpha=.8, label="SMI close"); ax.set_ylim(0, 105)
    ax.set_title(f"Turnover-cost agent, seed {s2}: exposure path  |  mean exposure {A2.mean()*100:.1f}%, turnover {np.abs(np.diff(np.r_[0,A2])).sum():.1f}")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F6_exposure_fixed_agent.png"), dpi=200); plt.close(fig)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
