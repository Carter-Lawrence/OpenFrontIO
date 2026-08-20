/**
 * sim_server.ts — OpenFront headless bridge for reinforcement learning.
 *
 * You do NOT need to understand or edit the game itself. This file is the only
 * TypeScript in your project, and it just does three things:
 *   1. Builds a headless OpenFront game (one "agent" player + some Impossible bots).
 *   2. Reads one JSON command per line from stdin  (sent by your Python code).
 *   3. Writes one JSON reply per line to stdout     (read by your Python code).
 *
 * Protocol (all messages are a single line of JSON):
 *   IN  {"cmd":"reset"}            -> OUT {"type":"reset","obs":{...},"meta":{...}}
 *   IN  {"cmd":"step","action":N}  -> OUT {"type":"step","obs":{...},"reward":R,"done":B,"info":{...}}
 *   IN  {"cmd":"close"}            -> process exits
 *
 * Normally you never run it directly — the Python environment launches it.
 */

import path from "path";
import readline from "readline";
import { fileURLToPath } from "url";

// NOTE: these paths are "./" because this file lives in the REPO ROOT, next to
// the game's src/ and tests/ folders. If you ever move this file into a
// subfolder, change each "./" to "../" (one "../" per folder level deeper).
import { setup } from "./tests/util/Setup";
import { AttackExecution } from "./src/core/execution/AttackExecution";
import { SpawnExecution } from "./src/core/execution/SpawnExecution";
import { NationExecution } from "./src/core/execution/NationExecution";
import {
  Cell,
  Difficulty,
  Game,
  GameMapType,
  Nation,
  Player,
  PlayerInfo,
  PlayerType,
} from "./src/core/game/Game";

// The game core occasionally prints to console.log. That would corrupt our
// stdout JSON channel, so we redirect ALL console output to stderr.
console.log = (...a: unknown[]) => process.stderr.write(a.join(" ") + "\n");
console.debug = () => {};

// ============================================================================
// CONFIG — the knobs you'll actually want to change. Everything below is plumbing.
// ============================================================================
const MAP = "big_plains"; // 200x200, all land. Small + dense = fast, forces conflict.
const GAME_MAP_TYPE = GameMapType.Pangaea; // metadata; real geometry comes from MAP's .bin
const DIFFICULTY = Difficulty.Impossible; // the opponent you ultimately want to beat
const NUM_BOTS = 5; // start small (1-5). More bots = harder + slower.
const MAX_TICKS = 5000; // episode length cap (~8 min of real game at 10 ticks/sec)
const TICKS_PER_STEP = 10; // "action repeat": each RL step advances 1 second of game
const GRID = 24; // observation is downsampled to GRID x GRID cells

const AGENT_ID = "agent";
const AGENT_SPAWN: [number, number] = [40, 40];
// Bot spawn points spread across the 200x200 map, away from the agent.
const BOT_SPAWNS: [number, number][] = [
  [160, 160],
  [40, 160],
  [160, 40],
  [100, 110],
  [160, 100],
  [100, 40],
  [40, 100],
];

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MAP_DIR = path.join(HERE, "tests", "util"); // repo-root/tests/util (map files)

// ----------------------------------------------------------------------------
// Episode state
// ----------------------------------------------------------------------------
let game: Game;
let agent: Player;
let prevTiles = 0;
let episode = 0;

function isPlayerObj(o: unknown): o is Player {
  return !!o && typeof (o as Player).isPlayer === "function" && (o as Player).isPlayer();
}

function aliveEnemies(): Player[] {
  return game
    .players()
    .filter((p) => p.id() !== AGENT_ID && p.isAlive());
}

// Enemies that border the agent, sorted weakest -> strongest by troops.
function borderingEnemies(): Player[] {
  const near = agent
    .nearby()
    .filter(
      (p): p is Player =>
        isPlayerObj(p) && p.isAlive() && p.id() !== AGENT_ID && !agent.isFriendly(p),
    );
  return near.sort((a, b) => a.troops() - b.troops());
}

// ----------------------------------------------------------------------------
// Observation: a downsampled map grid + a few scalar features.
//   grid cell values: 0 = ocean, 1 = neutral land, 2 = agent, 3 = enemy
// ----------------------------------------------------------------------------
function buildObs() {
  const w = game.width();
  const h = game.height();
  const grid: number[] = new Array(GRID * GRID);
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      const x = Math.min(w - 1, Math.floor(((gx + 0.5) * w) / GRID));
      const y = Math.min(h - 1, Math.floor(((gy + 0.5) * h) / GRID));
      const tile = game.ref(x, y);
      let v = 0;
      if (game.isLand(tile)) {
        const o = game.owner(tile);
        if (isPlayerObj(o)) v = o.id() === AGENT_ID ? 2 : 3;
        else v = 1;
      }
      grid[gy * GRID + gx] = v;
    }
  }
  const enemies = aliveEnemies();
  const enemyTiles = enemies.reduce((s, p) => s + p.numTilesOwned(), 0);
  return {
    grid,
    scalars: [
      agent.numTilesOwned(),
      Math.round(agent.troops()),
      Number(agent.gold()),
      enemies.length,
      enemyTiles,
      game.ticks(),
    ],
  };
}

// ----------------------------------------------------------------------------
// Higher-resolution per-player ownership map for the wall-of-games viewer.
// Only built when the OF_RENDER env var is set, so it never slows training.
//   cell values: -1 = ocean, 0 = neutral land, >0 = that player's smallID
// Set OF_RENDER=1 for the default 80x80, or OF_RENDER=120 for a finer map.
// ----------------------------------------------------------------------------
const RENDER = !!process.env.OF_RENDER;
const RENDER_SIZE = Number(process.env.OF_RENDER) > 1 ? Number(process.env.OF_RENDER) : 80;
let agentSmallID = -1;

function buildRender(): number[] | undefined {
  if (!RENDER) return undefined;
  const w = game.width();
  const h = game.height();
  const out: number[] = new Array(RENDER_SIZE * RENDER_SIZE);
  for (let gy = 0; gy < RENDER_SIZE; gy++) {
    for (let gx = 0; gx < RENDER_SIZE; gx++) {
      const x = Math.min(w - 1, Math.floor(((gx + 0.5) * w) / RENDER_SIZE));
      const y = Math.min(h - 1, Math.floor(((gy + 0.5) * h) / RENDER_SIZE));
      const tile = game.ref(x, y);
      let v = -1;
      if (game.isLand(tile)) {
        const o = game.owner(tile);
        v = isPlayerObj(o) ? o.smallID() : 0;
      }
      out[gy * RENDER_SIZE + gx] = v;
    }
  }
  return out;
}

// ----------------------------------------------------------------------------
// reset / step
// ----------------------------------------------------------------------------
async function reset() {
  episode++;
  const gameID = `rl_ep_${episode}_${Math.floor(Math.random() * 1e9)}`;

  game = await setup(
    MAP,
    { gameMap: GAME_MAP_TYPE, difficulty: DIFFICULTY },
    [],
    MAP_DIR,
    undefined,
    false, // don't auto-end the spawn phase; we manage it
  );

  // IMPORTANT ordering: add the bots FIRST and let them spawn. The spawn phase
  // auto-ends once a *human* has spawned, so if we added the agent first it would
  // close the phase before the bots ever claimed territory.
  for (let i = 0; i < Math.min(NUM_BOTS, BOT_SPAWNS.length); i++) {
    const [x, y] = BOT_SPAWNS[i];
    const info = new PlayerInfo(`bot_${i}`, PlayerType.Nation, null, `bot_${i}`);
    game.addExecution(new NationExecution(gameID, new Nation(new Cell(x, y), info)));
  }
  // Let the bots claim their starting territory (they spawn within ~3 ticks).
  for (let i = 0; i < 6; i++) game.executeNextTick();

  // Now add the agent (a normal Human player) and spawn it.
  const agentInfo = new PlayerInfo(AGENT_ID, PlayerType.Human, null, AGENT_ID);
  game.addPlayer(agentInfo);
  game.addExecution(new SpawnExecution(gameID, agentInfo, game.ref(...AGENT_SPAWN)));
  for (let i = 0; i < 4; i++) game.executeNextTick(); // agent spawns; phase auto-ends
  game.endSpawnPhase(); // safety: ensure we're out of the spawn phase

  agent = game.player(AGENT_ID);
  agentSmallID = agent.smallID();
  prevTiles = agent.numTilesOwned();

  return {
    type: "reset",
    obs: buildObs(),
    render: buildRender(),
    meta: {
      grid: GRID,
      nActions: 4,
      nScalars: 6,
      map: MAP,
      bots: NUM_BOTS,
      renderSize: RENDER ? RENDER_SIZE : 0,
      agentSmallID,
    },
  };
}

function applyAction(action: number) {
  if (!agent.isAlive()) return;
  const troops = Math.floor(agent.troops() * 0.5);
  if (troops <= 0) return;

  if (action === 1) {
    // Expand into neutral territory (attack with null target).
    game.addExecution(new AttackExecution(troops, agent, null, null));
  } else if (action === 2 || action === 3) {
    const enemies = borderingEnemies();
    if (enemies.length > 0) {
      const target = action === 2 ? enemies[0] : enemies[enemies.length - 1];
      game.addExecution(new AttackExecution(troops, agent, target.id(), null));
    }
  }
  // action === 0 is no-op.
}

function step(action: number) {
  applyAction(action);

  for (let i = 0; i < TICKS_PER_STEP; i++) {
    game.executeNextTick();
    if (!agent.isAlive()) break;
    if (game.ticks() >= MAX_TICKS) break;
  }

  // Reward: shaped by change in territory, with terminal bonuses.
  const tiles = agent.numTilesOwned();
  let reward = (tiles - prevTiles) * 0.01;
  prevTiles = tiles;

  let done = false;
  let reason = "";
  if (!agent.isAlive()) {
    reward -= 10;
    done = true;
    reason = "agent_died";
  } else if (aliveEnemies().length === 0) {
    reward += 10;
    done = true;
    reason = "agent_won";
  } else if (game.ticks() >= MAX_TICKS) {
    done = true;
    reason = "max_ticks";
  }

  return {
    type: "step",
    obs: buildObs(),
    render: buildRender(),
    reward,
    done,
    info: { reason, tiles, ticks: game.ticks(), enemies: aliveEnemies().length },
  };
}

// ----------------------------------------------------------------------------
// stdio loop: one JSON command per line, replies serialized in order.
// ----------------------------------------------------------------------------
const rl = readline.createInterface({ input: process.stdin });
let chain: Promise<void> = Promise.resolve();

function send(msg: unknown) {
  process.stdout.write(JSON.stringify(msg) + "\n");
}

rl.on("line", (line) => {
  const text = line.trim();
  if (!text) return;
  chain = chain.then(async () => {
    try {
      const cmd = JSON.parse(text);
      if (cmd.cmd === "reset") send(await reset());
      else if (cmd.cmd === "step") send(step(cmd.action | 0));
      else if (cmd.cmd === "close") process.exit(0);
      else send({ type: "error", error: `unknown cmd: ${cmd.cmd}` });
    } catch (e) {
      send({ type: "error", error: String((e as Error)?.stack ?? e) });
    }
  });
});

process.stderr.write("[sim_server] ready\n");