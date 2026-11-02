"""Sharpe-difference tests (same method as stats_tests.py) for the SAVED model behind Table 5.1
(ppo_smi_agent_master_v2.zip, exposure path recovered in results/original_model_v2_test_path.csv).
Light: numpy/pandas only. Writes results/stats_saved_model.json."""
import json, math, numpy as np, pandas as pd
src = open("stats_tests.py").read(); exec(src[:src.index("out = {")])
p = pd.read_csv("results/original_model_v2_test_path.csv")
A = p["exposure"].to_numpy()[:n]
dates_ok = (pd.to_datetime(p["date"]).to_numpy()[:n] == test.index.to_numpy()[:n]).mean() if hasattr(test.index, "to_numpy") else None
ret_gap = float(np.abs(p["smi_return"].to_numpy()[:n] - r).max())
out = {"model": "ppo_smi_agent_master_v2 (saved model behind Table 5.1)", "date_alignment_share": None if dates_ok is None else float(dates_ok), "max_abs_return_gap_vs_data_pkl": ret_gap}
g = net(A, cash=False, cost=False); w = np.cumprod(1 + g)
out["thesis_accounting"] = {"sharpe": float(math.sqrt(252) * g.mean() / g.std()), "cum": float(w[-1] - 1), "mdd": float((w / np.maximum.accumulate(w) - 1).min()),
                            "vs_BH_price_rf0": {k: lw_test(g, r, block=k) for k in (5, 10, 20)}}
c = net(A); wc = np.cumprod(1 + c); ex = c - rf
out["consistent_accounting"] = {"cum": float(wc[-1] - 1), "sharpe_excess_saron": float(math.sqrt(252) * ex.mean() / ex.std()),
                                "BH_price_sharpe_excess": float(math.sqrt(252) * (r - rf).mean() / (r - rf).std())}
f1 = {"BH price": r - rf, "BH total return": tr - rf, "60/40 real ETFs": r6040 - rf}
res = {k: lw_test(ex, v) for k, v in f1.items()}
adj = holm([v["p"] for v in res.values()]); [res[k].__setitem__("p_holm", a) for k, a in zip(res, adj)]
out["consistent_accounting"]["tests"] = res
json.dump(out, open("results/stats_saved_model.json", "w"), indent=1)
print(json.dumps(out, indent=1))
