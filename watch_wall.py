"""
watch_wall.py — watch many OpenFront games at once, as a live wall of maps.

Each little square is one game. Your agent's territory is bright green; bots are
other colors; grey is unclaimed land; dark is ocean. Games restart when they end.

Usage:
    python watch_wall.py                              # random agent, 3x3 wall
    python watch_wall.py --model ppo_openfront_final  # watch one saved model
    python watch_wall.py --model checkpoints          # watch-while-training: auto-
                                                       # reloads the newest checkpoint
    python watch_wall.py --games 16                   # 4x4 wall
    python watch_wall.py --snapshot wall.png          # headless: save a PNG and exit

Requirements:  pip install pygame-ce numpy      (pygame-ce, not pygame, on Python 3.13+)
(plus stable-baselines3 only if you pass --model)

It prefers sim_server's high-res per-player "render" map, but if that's missing it
falls back to the low-res observation grid, so the wall is never blank.
"""

import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
GRID = 24  # obs grid size, must match sim_server.ts


class GameProc:
    def __init__(self):
        npx = "npx.cmd" if os.name == "nt" else "npx"
        env = dict(os.environ)
        if not os.environ.get("OF_WALL_NORENDER"):
            env["OF_RENDER"] = "1"  # ask sim_server for the high-res map
        self.proc = subprocess.Popen(
            [npx, "tsx", "sim_server.ts"],
            cwd=REPO_DIR, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1,
        )
        self.render_size = None
        self.agent_small_id = None

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def send_reset(self):
        self._send({"cmd": "reset"})

    def send_step(self, action):
        self._send({"cmd": "step", "action": int(action)})

    def recv(self):
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("a game process closed unexpectedly")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and "type" in msg:
                meta = msg.get("meta")
                if meta:
                    self.render_size = meta.get("renderSize") or 0
                    self.agent_small_id = meta.get("agentSmallID")
                return msg

    def close(self):
        try:
            self._send({"cmd": "close"})
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def obs_for_model(msg):
    o = msg["obs"]
    return {
        "grid": np.array(o["grid"], dtype=np.int8).reshape(GRID, GRID),
        "scalars": np.array(o["scalars"], dtype=np.float32),
    }


def player_color_table(agent_small_id):
    """For the high-res render map: -1 ocean, 0 neutral, >0 = player smallID."""
    rng = np.random.default_rng(7)
    table = {-1: (18, 24, 40), 0: (90, 90, 90)}
    for sid in range(1, 64):
        table[sid] = tuple(int(c) for c in rng.integers(60, 230, size=3))
    if agent_small_id is not None:
        table[agent_small_id] = (60, 255, 90)  # your agent = bright green
    return table


# For the fallback obs grid: 0 ocean, 1 neutral, 2 agent, 3 enemy.
OBS_TABLE = {0: (18, 24, 40), 1: (90, 90, 90), 2: (60, 255, 90), 3: (220, 70, 70)}


def frame_rgb(msg, game):
    """Return (size, size, 3) RGB from the render map if present, else the obs grid."""
    render = msg.get("render")
    if render and game.render_size:
        arr = np.array(render, dtype=np.int32).reshape(game.render_size, game.render_size)
        table = player_color_table(game.agent_small_id)
    else:
        arr = np.array(msg["obs"]["grid"], dtype=np.int32).reshape(GRID, GRID)
        table = OBS_TABLE
    img = np.empty(arr.shape + (3,), dtype=np.uint8)
    for val in np.unique(arr):
        img[arr == val] = table.get(int(val), (200, 200, 200))
    return img


def newest_checkpoint(path):
    if path is None:
        return None
    if os.path.isfile(path):
        return path
    if os.path.isfile(path + ".zip"):
        return path + ".zip"
    zips = glob.glob(os.path.join(path, "*.zip"))
    return max(zips, key=os.path.getmtime) if zips else None


def try_load(path):
    from stable_baselines3 import PPO
    import model_def  # noqa: F401  makes the custom network class importable
    try:
        return PPO.load(path, device="cpu")
    except Exception as ex:
        print(f"  (couldn't load {os.path.basename(path)} yet: {ex})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=9)
    ap.add_argument("--model", type=str, default=None, help="a .zip OR a checkpoint folder to watch live")
    ap.add_argument("--reload-secs", type=float, default=30.0)
    ap.add_argument("--tile", type=int, default=200)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--snapshot", type=str, default=None, help="headless: save one PNG and exit")
    args = ap.parse_args()

    if args.snapshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    cols = int(np.ceil(np.sqrt(args.games)))
    rows = int(np.ceil(args.games / cols))
    pygame.init()
    screen = pygame.display.set_mode((cols * args.tile, rows * args.tile))
    pygame.display.set_caption("OpenFront — wall of games")
    font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()
    GAP = max(3, args.tile // 40)   # gutter between tiles = the border
    BORDER = (55, 55, 68)

    # Launch all game processes FIRST (so their TypeScript compiles happen in
    # parallel), showing a loading screen, then collect their first frames.
    print(f"Launching {args.games} games in parallel (first launch compiles TypeScript)...")
    screen.fill((25, 25, 30))
    msg = font.render("Launching games… (compiling, ~15s)", True, (230, 230, 230))
    screen.blit(msg, (20, 20))
    pygame.display.flip()

    games = [GameProc() for _ in range(args.games)]
    for g in games:
        g.send_reset()
    states = []
    for i, g in enumerate(games):
        states.append(g.recv())
        for e in pygame.event.get():  # keep the window responsive during boot
            pass

    model, model_path, last_check = None, None, 0.0
    steps, running = 0, True
    screen.fill(BORDER)  # gutters between tiles show through as borders
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        if args.model and time.time() - last_check > args.reload_secs:
            last_check = time.time()
            newest = newest_checkpoint(args.model)
            if newest and newest != model_path:
                m = try_load(newest)
                if m is not None:
                    model, model_path = m, newest
                    print(f"↻ now watching: {os.path.basename(newest)}")

        for i, g in enumerate(games):
            action = (model.predict(obs_for_model(states[i]), deterministic=True)[0]
                      if model else np.random.randint(4))
            g.send_step(action)
            msg = g.recv()
            if msg.get("done"):
                g.send_reset()
                msg = g.recv()
            states[i] = msg

            rgb = frame_rgb(msg, g)
            surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            inner = args.tile - 2 * GAP
            surf = pygame.transform.scale(surf, (inner, inner))
            x = (i % cols) * args.tile + GAP
            y = (i // cols) * args.tile + GAP
            screen.blit(surf, (x, y))
            # subtle outline around each game so tiles are easy to tell apart
            pygame.draw.rect(screen, (110, 110, 130), (x, y, inner, inner), width=2)

        pygame.display.flip()
        clock.tick(args.fps)
        steps += 1

        if steps == 1:
            has_render = bool(states[0].get("render"))
            print(f"[status] drawing OK. high-res render: {'YES' if has_render else 'NO (using obs-grid fallback)'}")

        if args.snapshot and steps >= 40:
            pygame.image.save(screen, args.snapshot)
            print(f"Saved snapshot to {args.snapshot}")
            running = False

    for g in games:
        g.close()
    pygame.quit()


if __name__ == "__main__":
    main()