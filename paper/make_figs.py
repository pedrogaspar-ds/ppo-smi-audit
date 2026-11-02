# Figuras da versão de portfólio (todas a partir de ../results). Uso: python3 make_figs.py
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "figs")
INK, GREY, NAVY, ACC, RED = "#222222", "#9aa3ad", "#1f3a5f", "#c8873a", "#b03a2e"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "savefig.bbox": "tight", "savefig.dpi": 300})

# 1. percurso do modelo reportado
p = pd.read_csv(os.path.join(R, "original_model_v2_test_path.csv"), parse_dates=["date"])
fig, (a, b) = plt.subplots(2, 1, figsize=(6.3, 3.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
a.plot(p.date, 100 * ((1 + p.smi_return).cumprod() - 1), color=GREY, lw=1, label="SMI price index (buy-and-hold)")
a.plot(p.date, 100 * ((1 + p.agent_return).cumprod() - 1), color=NAVY, lw=1.6, label="Reported PPO agent (gross)")
a.axhline(0, color=GREY, lw=0.5)
a.set_ylabel("Cumulative return (%)"); a.legend(frameon=False, loc="upper left")
b.fill_between(p.date, 100 * p.exposure, color=ACC, step="post", lw=0)
b.set_ylabel("Exposure (%)"); b.set_ylim(0, 105)
fig.align_ylabels(); fig.savefig(os.path.join(OUT, "f1_path.pdf")); plt.close(fig)

# 2. quadro com contabilidade consistente (Sharpe acima da SARON)
rows = [("Reported agent (in-sample)", 1.04, NAVY), ("Appendix A config., seed 27", 1.24, NAVY),
        ("Turnover-cost agent, median seed", 0.79, NAVY), ("60/40, real ETFs", 0.81, GREY),
        ("RSI 30/70 rule", 0.53, GREY), ("Buy-and-hold, total return", 0.52, GREY),
        ("60/40, stylised (as thesis)", 0.24, GREY), ("Buy-and-hold, price index", 0.18, GREY),
        ("Constant exposure 10/25/50%", 0.17, GREY), ("Long when MACD > signal", 0.10, GREY),
        ("Long when close > SMA200", -0.27, GREY), ("Long when RSI > 50", -0.34, GREY)]
fig, ax = plt.subplots(figsize=(6.3, 3.3))
y = np.arange(len(rows))[::-1]
ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
for yi, r in zip(y, rows):
    ax.text(r[1] + (0.03 if r[1] >= 0 else -0.03), yi, f"{r[1]:.2f}", va="center",
            ha="left" if r[1] >= 0 else "right", fontsize=8)
ax.set_yticks(y, [r[0] for r in rows]); ax.axvline(0, color=INK, lw=0.6)
ax.set_xlabel("Sharpe ratio in excess of SARON, net of 0.1% per change in exposure, 2023–2024")
ax.set_xlim(-0.55, 1.45); fig.savefig(os.path.join(OUT, "f2_scoreboard.pdf")); plt.close(fig)

# 3. sementes: exposição média vs retorno (contabilidade da tese)
e1 = pd.read_csv(os.path.join(R, "E1_thesis_30seeds.csv"))
e4 = json.load(open(os.path.join(R, "E4_control_1M.json")))
fig, ax = plt.subplots(figsize=(6.3, 3.2))
ax.scatter(100 * e1.mean_exposure, 100 * e1.gross_cum, s=22, color=GREY, label="Appendix A config., 10k steps, 30 seeds (out-of-sample)")
oos = [r for r in e4 if r["train"] == "2010-2022"]; ins = [r for r in e4 if r["train"] != "2010-2022"]
ax.scatter([100 * r["mean_exposure"] for r in oos], [100 * r["cum"] for r in oos], s=40, marker="D",
           color=ACC, label="1M steps, trained 2010–2022, 3 seeds (out-of-sample)")
ax.scatter([100 * r["mean_exposure"] for r in ins], [100 * r["cum"] for r in ins], s=40, marker="s",
           facecolor="white", edgecolor=NAVY, label="1M steps, trained 2010–2024, 3 seeds (in-sample)")
ax.scatter([3.2], [5.18], s=90, marker="*", color=NAVY, label="Reported model (in-sample)", zorder=5)
ax.axhline(5.67, color=GREY, ls="--", lw=0.8); ax.text(0.2, 5.82, "SMI buy-and-hold, price index: 5.67%", fontsize=7.5, color="#555")
ax.axhline(2.87, color=GREY, ls=":", lw=0.8); ax.text(0.2, 3.0, "Cash at SARON: 2.87%", fontsize=7.5, color="#555")
ax.set_xlabel("Mean exposure over the test window (%)"); ax.set_ylabel("Cumulative return, gross (%)")
ax.legend(frameon=False, fontsize=7.5, loc="lower right"); ax.set_ylim(-0.3, 7)
fig.savefig(os.path.join(OUT, "f3_seeds.pdf")); plt.close(fig)

# 4. sensibilidade ao custo do agente com custo de rotação
cs = pd.read_csv(os.path.join(R, "cost_sweep.csv")); cs = cs[cs.agent == "fixed-median"].sort_values("cost")
fig, ax = plt.subplots(figsize=(6.3, 2.6))
ax.plot(100 * cs.cost, cs.sharpe_saron, color=NAVY, marker="o", lw=1.6, ms=4, label="Turnover-cost agent, median seed")
for v, lab in [(0.81, "60/40, real ETFs (0.81)"), (0.525, "RSI 30/70 rule (0.53) and buy-and-hold, total return (0.52)")]:
    ax.axhline(v, color=GREY, ls="--", lw=0.8); ax.text(100 * cs.cost.max(), v + 0.02, lab, ha="right", fontsize=7.5, color="#555")
ax.set_xlabel("Cost per unit change in exposure (%)"); ax.set_ylabel("Sharpe over SARON")
ax.legend(frameon=False, loc="lower left"); fig.savefig(os.path.join(OUT, "f4_cost.pdf")); plt.close(fig)

# 5. janelas de stress: drawdown máximo (medianas; números da nota de robustez)
labels = ["Buy-and-hold\n(price)", "Appendix A\nconfig.", "Constant at its\nmean exposure", "Turnover-cost\nagent", "Constant at its\nmean exposure"]
v20 = [-27.5, -6.8, -1.3, -16.8, -7.3]; v22 = [-22.0, -0.65, -0.9, -20.9, -12.9]
fig, ax = plt.subplots(figsize=(6.3, 2.8)); x = np.arange(5); w = 0.38
ax.bar(x - w / 2, v20, w, color=NAVY, label="2020 (trained to end-2019)")
ax.bar(x + w / 2, v22, w, color=ACC, label="2022 (trained to end-2021)")
ax.set_xticks(x, labels, fontsize=7.8); ax.set_ylabel("Maximum drawdown (%)"); ax.axhline(0, color=INK, lw=0.6)
ax.legend(frameon=False, loc="lower right"); fig.savefig(os.path.join(OUT, "f5_stress.pdf")); plt.close(fig)
print("ok")
