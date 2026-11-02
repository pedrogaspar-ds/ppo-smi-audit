"""Control experiment for the original configuration (1,000,000 timesteps, holding cost, raw-level state):
   A) trained on 2010-2022 only (df.iloc[:-500])  -> genuinely out-of-sample on 2023-2024
   B) trained on the full 2010-2024 data, as the original model was -> test window seen in training
   3 seeds each, sequential, one process, 2 torch threads, low priority. Results appended to results/E4_control_1M.json"""
import os, sys, json, math, time, numpy as np, pandas as pd
import torch; torch.set_num_threads(2)
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
OUT = "results/E4_control_1M.json"
df = pd.read_csv("original/smi_data_final.csv")
class Env(gym.Env):
    def __init__(self, d):
        super().__init__(); self.c = d["Close"].to_numpy(); self.r = d["RSI"].to_numpy(); self.m = d["MACD"].to_numpy(); self.n = len(d); self.t = 0
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    def _o(self): return np.array([self.c[self.t] / 15000.0, self.r[self.t] / 100.0, self.m[self.t] / 100.0], dtype=np.float32)
    def reset(self, seed=None, options=None): super().reset(seed=seed); self.t = 0; return self._o(), {}
    def step(self, a):
        a = float(a[0]); self.t += 1; p, pp = self.c[self.t], self.c[self.t - 1]
        return self._o(), float(a * (p - pp) - p * a * 0.001), self.t >= self.n - 1, False, {}
test = df.iloc[-500:].reset_index(drop=True)
R = test["Close"].pct_change().shift(-1).to_numpy()[:-1]
def evaluate(model):
    env = Env(test); obs, _ = env.reset(); A = []
    for i in range(len(test) - 1):
        a, _ = model.predict(obs, deterministic=True); a = float(np.clip(a[0], 0, 1)); A.append(a); obs, *_ = env.step(np.array([a], dtype=np.float32))
    A = np.array(A); g = A * R; w = np.cumprod(1 + g)
    sd = g.std(); return dict(sharpe=float(math.sqrt(252) * g.mean() / sd) if sd > 0 else 0.0, mdd=float((w / np.maximum.accumulate(w) - 1).min()), cum=float(w[-1] - 1),
                              mean_exposure=float(A.mean()), days_invested=int((A > 0.01).sum()), actions=A.round(4).tolist())
res = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {(x["train"], x["seed"]) for x in res}
for seed in (0, 1, 2):
    for train in ("2010-2022", "2010-2024 (as original)"):
        if (train, seed) in done: continue
        d = df.iloc[:-500].reset_index(drop=True) if train == "2010-2022" else df
        t0 = time.time(); model = PPO("MlpPolicy", Env(d), seed=seed, device="cpu", verbose=0); model.learn(total_timesteps=1_000_000)
        row = dict(train=train, seed=seed, minutes=round((time.time() - t0) / 60, 1), **evaluate(model)); res.append(row)
        json.dump(res, open(OUT, "w")); print(time.strftime("%H:%M"), train, "seed", seed, {k: row[k] for k in ("sharpe", "mdd", "cum", "mean_exposure", "days_invested", "minutes")}, flush=True)
print("ALL DONE", flush=True)
