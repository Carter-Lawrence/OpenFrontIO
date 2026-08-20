"""
openfront_env.py — a Gymnasium environment wrapping the OpenFront headless sim.

This is the ONLY thing you talk to from Python. Under the hood it launches the
TypeScript bridge (sim_server.ts) as a subprocess and exchanges one line of JSON
per call, but you never see that — you just use the standard Gym interface:

    env = OpenFrontEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()

Observation is a dict:
    "grid"    : (GRID, GRID) int array. 0=ocean, 1=neutral land, 2=you, 3=enemy.
    "scalars" : [your_tiles, your_troops, your_gold, enemies_alive,
                 enemy_tiles, tick]

Action is Discrete(4):
    0 = do nothing
    1 = expand into neutral land
    2 = attack the WEAKEST bordering enemy
    3 = attack the STRONGEST bordering enemy

Requirements:  pip install gymnasium numpy
(Node must be installed and `npx tsx sim_server.ts` must run from this folder —
which you already confirmed when the tests passed.)
"""

import json
import os
import subprocess
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
GRID = 24  # must match GRID in sim_server.ts
N_SCALARS = 6


class OpenFrontEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0, high=3, shape=(GRID, GRID), dtype=np.int8),
                # Loose upper bounds — just so the space is well-formed.
                "scalars": spaces.Box(
                    low=0.0, high=np.inf, shape=(N_SCALARS,), dtype=np.float32
                ),
            }
        )
        self.proc = None
        self._mask = None  # last mask received from sim_server

    # -- subprocess plumbing -------------------------------------------------
    def _start(self):
        # On Windows the executable is "npx.cmd"; elsewhere "npx".
        npx = "npx.cmd" if os.name == "nt" else "npx"
        self.proc = subprocess.Popen(
            [npx, "tsx", "sim_server.ts"],
            cwd=REPO_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # let the bridge's stderr flow to your terminal (for debugging)
            text=True,
            bufsize=1,  # line-buffered
        )

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _recv(self):
        # Read lines until we get a valid JSON protocol message (skip any noise).
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(
                    "sim_server closed unexpectedly. Check the errors printed above."
                )
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # not a protocol line; ignore
            if isinstance(msg, dict) and "type" in msg:
                if msg["type"] == "error":
                    raise RuntimeError("sim_server error: " + msg.get("error", "?"))
                return msg

    def _decode_obs(self, obs):
        grid = np.array(obs["grid"], dtype=np.int8).reshape(GRID, GRID)
        scalars = np.array(obs["scalars"], dtype=np.float32)
        return {"grid": grid, "scalars": scalars}

    # -- Gym API -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        self._send({"cmd": "reset"})
        msg = self._recv()
        self._mask = msg.get("mask")
        return self._decode_obs(msg["obs"]), {"meta": msg.get("meta", {})}

    def step(self, action):
        self._send({"cmd": "step", "action": int(action)})
        msg = self._recv()
        self._mask = msg.get("mask")
        obs = self._decode_obs(msg["obs"])
        reward = float(msg["reward"])
        info = msg.get("info", {})
        done = bool(msg["done"])
        # timeout = truncation (PPO bootstraps); win/loss = real termination
        timed_out = info.get("reason") == "max_ticks"
        terminated = done and not timed_out
        truncated = done and timed_out
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self._send({"cmd": "close"})
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None

    def action_masks(self):
        if self._mask is None:
            return np.ones(self.action_space.n, dtype=bool)
        return np.array(self._mask, dtype=bool)

# ---------------------------------------------------------------------------
# Smoke test: run a few episodes with a RANDOM agent and print what happens.
# Run:  python openfront_env.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = OpenFrontEnv()
    print("Launching sim (first run compiles TypeScript, ~a few seconds)...")
    for ep in range(3):
        obs, info = env.reset()
        total_r = 0.0
        steps = 0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            steps += 1
            if terminated or truncated:
                break
        print(
            f"episode {ep}: steps={steps} total_reward={total_r:.2f} "
            f"final_tiles={info.get('tiles')} enemies_left={info.get('enemies')} "
            f"end={info.get('reason')}"
        )
    env.close()
    print("Smoke test done. If you see 3 episode lines above, the loop works.")
    sys.exit(0)