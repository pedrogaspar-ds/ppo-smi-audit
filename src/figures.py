import os, json, math, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from experiments import load_data, strategy_returns, metrics, block_bootstrap_sharpe_diff, RES, FIG, TEST_N
df = load_data(); test = df.iloc[-TEST_N:].copy()
e1 = json.load(open(f"{RES}/E1_thesis_30seeds.json")); e2 = json.load(open(f"{RES}/E2_fixed_10seeds.json")); e3 = json.load(open(f"{RES}/E3_walkforward.json"))
base = json.load(open(f"{RES}/baselines.json")); s = json.load(open(f"{RES}/summary.json"))
d1 = pd.DataFrame(e1["rows"]); d2 = pd.DataFrame(e2["rows"]); d3 = pd.DataFrame(e3["rows"])
win = s["test_window"]
# representative seeds (Appendix A configuration, NOT the model behind Table 5.1): E1 -> invested seed closest to the thesis's reported cumulative return (5.18%); E2 -> median net Sharpe
inv = d1[d1.pct_days_invested > 0.01]
s1 = int(inv.iloc[(inv.gross_cum - 0.0518).abs().argsort().iloc[0]].seed)
s2 = int(d2.iloc[(d2.net_sharpe_saron - d2.net_sharpe_saron.median()).abs().argsort().iloc[0]].seed)
A1 = np.array(e1["actions"][f"thesis-s{s1}-{win}"]); A2 = np.array(e2["actions"][f"fixed-s{s2}-{win}"])
r = test["r_next"].to_numpy()[: len(A1)]; rf = test["rf_next"].to_numpy()[: len(A1)]; dates = test.index[: len(A1)]
g1, _ = strategy_returns(A1, r, rf, cash_yield=False, turnover_cost=False); n1, _ = strategy_returns(A1, r, rf); n2, _ = strategy_returns(A2, r, rf)
# model behind Table 5.1 (the saved model ppo_smi_agent_master_v2.zip, not included): its exposure path on the test window
P0 = pd.read_csv(f"{RES}/original_model_v2_test_path.csv"); A0 = P0["exposure"].to_numpy()[: len(A1)]
assert (pd.to_datetime(P0["date"]).to_numpy()[: len(A1)] == dates.to_numpy()).all()
n0, _ = strategy_returns(A0, r, rf)
rsi = test["RSI"].to_numpy()[: len(A1)]; Ar = np.zeros(len(A1)); inpos = False
for i in range(len(A1)):
    if not inpos and rsi[i] < 30: inpos = True
    elif inpos and rsi[i] > 70: inpos = False
    Ar[i] = 1.0 if inpos else 0.0
nrsi, _ = strategy_returns(Ar, r, rf)
tr = test["r_cssmi_next"].to_numpy()[: len(A1)]; bo = test["r_bond_next"].to_numpy()[: len(A1)]; r6040 = 0.6 * tr + 0.4 * bo
boot = dict(thesis_like_seed_gross_vs_bh_rf0=block_bootstrap_sharpe_diff(g1, r, np.zeros_like(rf)),
            thesis_like_seed_net_vs_bh_saron=block_bootstrap_sharpe_diff(n1, r, rf),
            fixed_median_net_vs_bh_saron=block_bootstrap_sharpe_diff(n2, r, rf),
            fixed_median_net_vs_bh_totalreturn_saron=block_bootstrap_sharpe_diff(n2, tr, rf),
            fixed_median_net_vs_rsi_rule_saron=block_bootstrap_sharpe_diff(n2, nrsi, rf))
rep = dict(E1_rep_seed=s1, E1_rep=dict(gross=metrics(g1, rf), net=metrics(n1, rf), mean_exposure=float(A1.mean()), pct_days_invested=float((A1 > .01).mean()), turnover=float(np.abs(np.diff(np.r_[0, A1])).sum())),
           E2_rep_seed=s2, E2_rep=dict(net=metrics(n2, rf), mean_exposure=float(A2.mean()), pct_days_invested=float((A2 > .01).mean()), turnover=float(np.abs(np.diff(np.r_[0, A2])).sum())),
           bootstrap=boot, saron_cum=float(np.prod(1 + rf) - 1), saron_mean_annual=float(rf.mean() * 252),
           E1_share_never_invested=float((d1.pct_days_invested < 0.01).mean()), E1_invested_seeds=dict(n=int(len(inv)), gross_sharpe_min=float(inv.gross_sharpe_rf0.min()), gross_sharpe_max=float(inv.gross_sharpe_rf0.max()),
           gross_cum_min=float(inv.gross_cum.min()), gross_cum_max=float(inv.gross_cum.max()), gross_mdd_min=float(inv.gross_mdd.min()), mean_exposure_max=float(inv.mean_exposure.max()), net_sharpe_saron_min=float(inv.net_sharpe_saron.min()), net_sharpe_saron_max=float(inv.net_sharpe_saron.max())))
json.dump(rep, open(f"{RES}/representatives.json", "w"), indent=1, default=float)
print(json.dumps(rep, indent=1, default=float))

plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
C = dict(blue="#1f4e79", orange="#d9822b", grey="#555555", lgrey="#9a9a9a", green="#2e8b57", purple="#7d3c98")
# F1 exposure path, model behind Table 5.1
fig, ax = plt.subplots(figsize=(12, 4.6)); ax2 = ax.twinx()
ax.fill_between(dates, A0 * 100, 0, color=C["blue"], alpha=.25, step="mid"); ax.plot(dates, A0 * 100, color=C["blue"], lw=1.3, label="Agent exposure to SMI (%), model behind Table 5.1")
ax2.plot(dates, test["Close"].to_numpy()[: len(A1)], color=C["grey"], lw=1, alpha=.85, label="SMI close (right axis)")
ax.set_ylabel("Exposure (%)"); ax2.set_ylabel("SMI level"); ax.set_ylim(0, 105)
ax.set_title(f"Model behind Table 5.1: invested on {int((A0 > 0).sum())} of {len(A0)} days in three entries, mean exposure {A0.mean()*100:.1f}%, peak {A0.max()*100:.0f}%", fontsize=12)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/F1_exposure_thesis_config.png", dpi=200); plt.close(fig)

# F2 seed distribution
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
bhm = base["Buy-and-Hold SMI (price index, as thesis)"]
for ax, col, bhv, title, scale in [(axs[0], "gross_sharpe_rf0", bhm["sharpe_rf0"], "Sharpe ratio (rf = 0, gross, as thesis)", 1), (axs[1], "gross_mdd", bhm["mdd"], "Maximum drawdown (%)", 100), (axs[2], "gross_cum", bhm["cum"], "Cumulative return (%)", 100)]:
    v = d1[col].to_numpy() * scale; rng = np.random.default_rng(1)
    ax.scatter(rng.normal(0, .05, len(v)), v, s=26, color=C["blue"], alpha=.75, zorder=3, label="30 seeds, Appendix A configuration")
    ax.axhline(bhv * scale, color=C["grey"], ls=":", lw=1.4, label="Buy-and-Hold SMI")
    ax.set_title(title, fontsize=12); ax.set_xticks([]); ax.set_xlim(-.4, .4)
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.01)); fig.suptitle(f"Appendix A configuration (10,000 steps, trained on 2010-2022), 30 random seeds: {int((d1.pct_days_invested<0.01).sum())} of 30 seeds never really invest", fontsize=12)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{FIG}/F2_seed_distribution.png", dpi=200); plt.close(fig)

# F3 cumulative returns, consistent accounting
def cum(x): return (np.cumprod(1 + x) - 1) * 100
fig, ax = plt.subplots(figsize=(12, 5.2))
ax.plot(dates, cum(n0), color=C["blue"], lw=2.2, label="Model behind Table 5.1 (net, cash at SARON)")
ax.plot(dates, cum(n2), color=C["orange"], lw=2.2, label=f"Turnover-cost agent, seed {s2} (net, cash at SARON)")
ax.plot(dates, cum(r), color=C["grey"], lw=1.3, label="Buy-and-Hold SMI price index (as in thesis)")
ax.plot(dates, cum(tr), color=C["lgrey"], lw=1.3, ls="-.", label="Buy-and-Hold SMI total return (iShares SMI ETF)")
ax.plot(dates, cum(r6040), color=C["purple"], lw=1.3, ls="--", label="60/40 with real ETFs (CSSMI / CSBGC0)")
ax.plot(dates, cum(nrsi), color=C["green"], lw=1.3, label="Rule: buy RSI<30, sell RSI>70 (net)")
ax.plot(dates, cum(rf), color="k", lw=1, ls=":", label="Cash at SARON proxy")
ax.set_ylabel("Cumulative return (%)"); ax.set_title(f"Test window {win}, consistent accounting: idle cash earns SARON, 0.1% cost on every change of exposure", fontsize=12); ax.legend(fontsize=9, loc="upper left"); ax.axhline(0, color="k", lw=.5)
fig.tight_layout(); fig.savefig(f"{FIG}/F3_cumulative_returns.png", dpi=200); plt.close(fig)

# F4 consistent-accounting scoreboard (Sharpe in excess of SARON, cum return, MDD)
rows = [("Model behind\nTable 5.1", metrics(n0, rf), C["blue"]), ("Turnover-\ncost agent", metrics(n2, rf), C["orange"]),
        ("B&H\nSMI price", metrics(r, rf), C["grey"]), ("B&H SMI\ntotal return", metrics(tr, rf), C["lgrey"]),
        ("60/40\nreal ETFs", metrics(r6040, rf), C["purple"]), ("RSI 30/70\nrule", metrics(nrsi, rf), C["green"])]
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for ax, key, title, scale in [(axs[0], "sharpe_rf", "Sharpe ratio, excess over SARON", 1), (axs[1], "cum", "Cumulative return (%)", 100), (axs[2], "mdd", "Maximum drawdown (%)", 100)]:
    vals = [m[key] * scale for _, m, _ in rows]; ax.bar(range(len(rows)), vals, color=[c for _, _, c in rows], width=.7)
    for i, v in enumerate(vals): ax.text(i, v + (0.02 * (max(vals) - min(vals)) if v >= 0 else -0.06 * (max(vals) - min(vals))), f"{v:.2f}" if scale == 1 else f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([n for n, _, _ in rows], fontsize=9.5); ax.set_title(title, fontsize=11, pad=8 if key != "mdd" else 34); ax.axhline(0, color="k", lw=.6)
    if key == "mdd": ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0); ax.set_ylim(min(vals) * 1.18, 0.5)
fig.suptitle(f"Same rules for everyone: idle cash earns the SARON proxy, 0.1% on every change of exposure, dividends via ETFs  (test window {win})", fontsize=12); fig.tight_layout(); fig.savefig(f"{FIG}/F4_consistent_scoreboard.png", dpi=200); plt.close(fig)

# F7 E2 seed distribution vs baselines (net Sharpe over SARON and cum)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4)); rng = np.random.default_rng(2)
for ax, col, title, scale in [(axs[0], "net_sharpe_saron", "Sharpe ratio, excess over SARON (net)", 1), (axs[1], "net_cum", "Cumulative return (%, net)", 100)]:
    v = d2[col].to_numpy() * scale; ax.scatter(rng.normal(0, .05, len(v)), v, s=34, color=C["orange"], alpha=.85, zorder=3, label="turnover-cost agent, 10 seeds")
    for name, m, c in [("Buy-and-Hold price", metrics(r, rf), C["grey"]), ("Buy-and-Hold total return", metrics(tr, rf), C["lgrey"]), ("60/40 real ETFs", metrics(r6040, rf), C["purple"]), ("RSI 30/70 rule", metrics(nrsi, rf), C["green"])]:
        ax.axhline(m["sharpe_rf" if col == "net_sharpe_saron" else "cum"] * scale, color=c, ls="--", lw=1.2, label=name)
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_xlim(-.4, .4)
axs[0].legend(fontsize=8.5, loc="lower right"); fig.suptitle("Turnover-cost agent: 10 seeds x 200,000 timesteps, out-of-sample 2023-2024", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIG}/F7_fixed_agent_seeds.png", dpi=200); plt.close(fig)

# F4b risk-free sensitivity of the Sharpe ratio (Appendix A configuration, seed s1, gross as in thesis)
fig, ax = plt.subplots(figsize=(8, 4)); s6040 = base["60/40 stylized (thesis: 0.6 SMI + 0.4 x 0.01%/day)"]
names = [f"Appendix A config,\nseed {s1} (gross)", "Buy-and-Hold\nSMI (price)", "60/40\nstylized"]
vals = {"rf = 0": [metrics(g1)["sharpe_rf0"], bhm["sharpe_rf0"], s6040["sharpe_rf0"]],
        "rf = SARON proxy": [metrics(g1, rf)["sharpe_rf"], bhm["sharpe_saron"], s6040["sharpe_saron"]],
        "rf = 2.5% (thesis cash rate)": [metrics(g1, np.full_like(r, .025 / 252))["sharpe_rf"], bhm["sharpe_rf25"], s6040["sharpe_rf25"]]}
x = np.arange(3); wdt = .26
for i, (k, v) in enumerate(vals.items()):
    ax.bar(x + (i - 1) * wdt, v, wdt, label=k, color=["#1f5fa8", "#5dade2", "#aed6f1"][i])
    for xi, vi in zip(x + (i - 1) * wdt, v): ax.text(xi, vi + (0.02 if vi >= 0 else -0.08), f"{vi:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names); ax.axhline(0, color="k", lw=.6); ax.set_ylabel("Annualised Sharpe ratio"); ax.legend(fontsize=9)
ax.set_title("The Sharpe ranking depends on the risk-free rate used"); fig.tight_layout(); fig.savefig(f"{FIG}/F4_rf_sensitivity.png", dpi=200); plt.close(fig)

# F5 walk-forward stress windows: drawdown of the median-return seeds vs Buy-and-Hold (cash-biased agent = Appendix A configuration, 10,000 steps)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
def ddc(x): w_ = np.cumprod(1 + x); return (w_ / np.maximum.accumulate(w_) - 1) * 100
for ax, w in zip(axs, sorted(d3.window.unique())):
    a, b = w.split(".."); tw = df.loc[a:b]; rr = tw["r_next"].to_numpy(); ff = tw["rf_next"].to_numpy(); m = None
    for cfg, color, lab in [("thesis", "#1f5fa8", "Appendix A config, 10,000 steps"), ("fixed", "#e67e22", "Turnover-cost agent")]:
        g = d3[(d3.window == w) & (d3.cfg == cfg)]
        # median seed = the seed whose net cumulative return is closest to the median (net Sharpe over SARON is undefined/explodes for zero-exposure seeds)
        sd = int(g.iloc[(g["net_cum"] - g["net_cum"].median()).abs().to_numpy().argsort(kind="stable")[0]]["seed"])
        A = np.array(e3["actions"][f"{cfg}-s{sd}-{w}"]); m = len(A)
        nr, _ = strategy_returns(A, rr[:m], ff[:m]); ax.plot(tw.index[:m], ddc(nr), color=color, lw=1.5, label=f"{lab}, seed {sd}: mean exp. {A.mean()*100:.1f}%, max DD {ddc(nr).min():.1f}%")
    ax.plot(tw.index[:m], ddc(rr[:m]), color="#555", lw=1.2, label="Buy-and-Hold SMI")
    ax.set_title(f"Stress window {a[:4]}: drawdown (%), median-return seeds (net)", fontsize=11); ax.legend(fontsize=7.5, loc="lower left")
    import matplotlib.dates as mdates; ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
fig.tight_layout(); fig.savefig(f"{FIG}/F5_walkforward_stress.png", dpi=200); plt.close(fig)
print("figures ok", s1, s2)
