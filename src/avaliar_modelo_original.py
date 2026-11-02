"""Evaluate the ORIGINAL ppo_smi_agent_master_v2.zip without importing torch (numpy forward pass of the SB3 MlpPolicy)."""
import zipfile, pickle, io, json, math, numpy as np, pandas as pd
Z = "original/ppo_smi_agent_master_v2.zip"; CSV = "original/smi_data_final.csv"

def load_state_dict(zpath):
    outer = zipfile.ZipFile(zpath); inner = zipfile.ZipFile(io.BytesIO(outer.read("policy.pth")))
    names = inner.namelist(); root = names[0].split("/")[0]
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                def rebuild(storage, offset, size, stride, *a):
                    n = int(np.prod(size)) if size else 1
                    arr = storage[offset:offset + n]
                    return arr.reshape(size) if size else arr.reshape(())
                return rebuild
            if name == "_rebuild_parameter":
                return lambda data, *a: data
            if name == "OrderedDict": import collections; return collections.OrderedDict
            if mod.startswith("torch"): return name   # storage type names, e.g. 'FloatStorage'
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            _, stype, key, loc, numel = pid
            dt = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64}.get(stype if isinstance(stype, str) else getattr(stype, "__name__", "FloatStorage"), np.float32)
            return np.frombuffer(inner.read(f"{root}/data/{key}"), dtype=dt)
    return U(io.BytesIO(inner.read(f"{root}/data.pkl"))).load()

sd = load_state_dict(Z)
W = {k: np.asarray(v, dtype=np.float32) for k, v in sd.items()}
print("policy tensors:", {k: v.shape for k, v in W.items()})
def act(obs):
    h = np.tanh(W["mlp_extractor.policy_net.0.weight"] @ obs + W["mlp_extractor.policy_net.0.bias"])
    h = np.tanh(W["mlp_extractor.policy_net.2.weight"] @ h + W["mlp_extractor.policy_net.2.bias"])
    mean = W["action_net.weight"] @ h + W["action_net.bias"]
    return float(np.clip(mean[0], 0.0, 1.0)), float(mean[0])

df = pd.read_csv(CSV); test = df.iloc[-500:].reset_index(drop=True)
A, M, R = [], [], []
for i in range(len(test) - 1):
    obs = np.array([test.loc[i, "Close"] / 15000.0, test.loc[i, "RSI"] / 100.0, test.loc[i, "MACD"] / 100.0], dtype=np.float32)
    a, m = act(obs); A.append(a); M.append(m)
    R.append((test.loc[i + 1, "Close"] - test.loc[i, "Close"]) / test.loc[i, "Close"])
A, M, R = map(np.array, (A, M, R)); ag = A * R
def sharpe(x): return math.sqrt(252) * x.mean() / x.std()
def mdd(x): w = np.cumprod(1 + x); return (w / np.maximum.accumulate(w) - 1).min()
cum = np.cumprod(1 + ag) - 1
out = dict(sharpe=sharpe(ag), mdd=mdd(ag), cum_day490=cum[489], cum_499=cum[-1], bh_sharpe=sharpe(R), bh_mdd=mdd(R), bh_cum_day490=(np.cumprod(1 + R) - 1)[489], bh_cum_499=(np.cumprod(1 + R) - 1)[-1],
           mean_exposure=A.mean(), pct_days_invested=(A > 0.01).mean(), n_days_invested=int((A > 0.01).sum()), max_exposure=A.max(), n_entries=int(((A[1:] > 0.01) & (A[:-1] <= 0.01)).sum() + (A[0] > 0.01)),
           turnover=float(np.abs(np.diff(np.r_[0, A])).sum()))
print(json.dumps({k: round(float(v), 5) for k, v in out.items()}, indent=1))
print("checkpoints (day: agent cum %):", {d: round(cum[d - 1] * 100, 2) for d in (10, 50, 60, 200, 210, 220, 470, 480, 490)})
inv = np.where(A > 0.01)[0]; print("invested days (index: exposure):", [(int(i), round(float(A[i]), 3)) for i in inv][:60])
pd.DataFrame({"date": test.iloc[:-1, 0].values, "exposure": A, "policy_mean": M, "smi_return": R, "agent_return": ag}).to_csv("results/original_model_v2_test_path.csv", index=False)
json.dump({k: float(v) for k, v in out.items()}, open("results/original_model_v2_eval.json", "w"), indent=1)
