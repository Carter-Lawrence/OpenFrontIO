/**
 * sim_server.ts — OpenFront headless bridge for reinforcement learning.
 *
 * Protocol (one line of JSON each):
 *   IN  {"cmd":"reset"}            -> OUT {"type":"reset","obs":{...},"meta":{...}}
 *   IN  {"cmd":"step","action":N}  -> OUT {"type":"step","obs":{...},"reward":R,"done":B,"info":{...}}
 *   IN  {"cmd":"close"}            -> process exits
 */

import path from "path";
import readline from "readline";
import { fileURLToPath } from "url";

import { setup } from "./tests/util/Setup";
import { AttackExecution } from "./src/core/execution/AttackExecution";
import { ConstructionExecution } from "./src/core/execution/ConstructionExecution";
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
  UnitType,
} from "./src/core/game/Game";
// Redirect console output to stderr so it can't corrupt the stdout JSON channel.
console.log = (...a: unknown[]) => process.stderr.write(a.join(" ") + "\n");
console.debug = () => {};

// ============================================================================
// CONFIG
// ============================================================================
const MAP = "big_plains"; // 200x200, all land.
const GAME_MAP_TYPE = GameMapType.Pangaea;
const DIFFICULTY = Difficulty.Impossible;
const NUM_BOTS = 5;
const MAX_TICKS = 25000; // 25000 ticks / TICKS_PER_STEP(10) = 2500 RL steps/episode
const TICKS_PER_STEP = 10;
const GRID = 24;

const AGENT_ID = "agent";
const AGENT_SPAWN: [number, number] = [40, 40];
// Bots ring the agent (at 40,40) so conflict is unavoidable within a couple
// hundred ticks, instead of exiled to the far corners where the agent could
// farm free land forever and never fight.
const BOT_SPAWNS: [number, number][] = [
  [40, 10],
  [10, 40],
  [90, 40],
  [40, 90],
  [80, 80],
  [10, 80],
  [80, 10],
];

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MAP_DIR = path.join(HERE, "tests", "util");

// ----------------------------------------------------------------------------
// Episode state
// ----------------------------------------------------------------------------
let game: Game;
let agent: Player;
let episode = 0;

// Reward bookkeeping — declared at module scope so reset() and step() share them.
let totalLand = 1; // land tiles on the map; set in reset()
let prevOwnFrac = 0; // agent's fraction of the land last step
let prevEnemies = 0; // enemies alive last step

function isPlayerObj(o: unknown): o is Player {
  return !!o && typeof (o as Player).isPlayer === "function" && (o as Player).isPlayer();
}

function aliveEnemies(): Player[] {
  return game.players().filter((p) => p.id() !== AGENT_ID && p.isAlive());
}

function borderingEnemies(): Player[] {
  const near = agent
    .nearby()
    .filter(
      (p): p is Player =>
        isPlayerObj(p) && p.isAlive() && p.id() !== AGENT_ID && !agent.isFriendly(p),
    );
  return near.sort((a, b) => a.troops() - b.troops());
}

// Action indices 4-8 map to structures the agent can build. Placement is
// heuristic (the bridge finds a valid owned tile); the agent only picks WHAT.
const BUILD_ACTIONS: Record<number, UnitType> = {
  4: UnitType.City,        // economy: more population/troop income
  5: UnitType.Port,        // economy + unlocks naval
  6: UnitType.Factory,     // economy/production
  7: UnitType.DefensePost, // defensive bonus on nearby tiles
  8: UnitType.SAMLauncher, // shoots down incoming nukes
};

// Try to build `type` on one of the agent's tiles. Returns true if queued.
function tryBuild(type: UnitType): boolean {
  const border = Array.from(agent.borderTiles());
  if (border.length === 0) return false;
  const stride = Math.max(1, Math.floor(border.length / 40)); // sample ~40 tiles
  for (let i = 0; i < border.length; i += stride) {
    const spot = agent.canBuild(type, border[i]); // TileRef if legal, else false
    if (spot !== false) {
      game.addExecution(new ConstructionExecution(agent, type, spot));
      return true;
    }
  }
  return false;
}

// Boolean legality per action, so PPO never wastes a step on an illegal move.
function computeMask(): boolean[] {
  // [noop, expand, atkWeak, atkStrong, city, port, factory, defense, sam]
  if (!agent || !agent.isAlive()) {
    return [true, false, false, false, false, false, false, false, false];
  }
  const mask = [true, true, false, false, false, false, false, false, false];
  const hasEnemyBorder = borderingEnemies().length > 0;
  mask[2] = hasEnemyBorder;
  mask[3] = hasEnemyBorder;

  const border = Array.from(agent.borderTiles());
  if (border.length > 0) {
    const gold = agent.gold();
    const list = agent.buildableUnits(border[0], [
      UnitType.City,
      UnitType.Port,
      UnitType.Factory,
      UnitType.DefensePost,
      UnitType.SAMLauncher,
    ]);
    const idx: Record<string, number> = {
      [UnitType.City]: 4,
      [UnitType.Port]: 5,
      [UnitType.Factory]: 6,
      [UnitType.DefensePost]: 7,
      [UnitType.SAMLauncher]: 8,
    };
    for (const b of list) {
      const i = idx[b.type];
      if (i !== undefined) mask[i] = gold >= b.cost; // affordable = legal
    }
  }
  return mask;
}
// ----------------------------------------------------------------------------
// Observation grid (0=ocean, 1=neutral, 2=agent, 3=enemy) + scalars
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
// High-res per-player map for the wall viewer (only when OF_RENDER is set)
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
    false,
  );

  // Bots FIRST (the spawn phase auto-ends once a human spawns).
  for (let i = 0; i < Math.min(NUM_BOTS, BOT_SPAWNS.length); i++) {
    const [x, y] = BOT_SPAWNS[i];
    const info = new PlayerInfo(`bot_${i}`, PlayerType.Nation, null, `bot_${i}`);
    game.addExecution(new NationExecution(gameID, new Nation(new Cell(x, y), info)));
  }
  for (let i = 0; i < 6; i++) game.executeNextTick();

  // Then the agent.
  const agentInfo = new PlayerInfo(AGENT_ID, PlayerType.Human, null, AGENT_ID);
  game.addPlayer(agentInfo);
  game.addExecution(new SpawnExecution(gameID, agentInfo, game.ref(...AGENT_SPAWN)));
  for (let i = 0; i < 4; i++) game.executeNextTick();
  game.endSpawnPhase();

  agent = game.player(AGENT_ID);
  agentSmallID = agent.smallID();

  // Count land tiles once so territory can be a fraction in [0,1].
  totalLand = 0;
  const w = game.width();
  const h = game.height();
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      if (game.isLand(game.ref(x, y))) totalLand++;
    }
  }
  if (totalLand < 1) totalLand = 1;
  prevOwnFrac = agent.numTilesOwned() / totalLand;
  prevEnemies = aliveEnemies().length;

  return {
    type: "reset",
    obs: buildObs(),
    render: buildRender(),
    mask: computeMask(),
    meta: {
      grid: GRID,
      nActions: 9,
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

  if (action === 1) {
    const troops = Math.floor(agent.troops() * 0.5);
    if (troops > 0) game.addExecution(new AttackExecution(troops, agent, null, null));
  } else if (action === 2 || action === 3) {
    const troops = Math.floor(agent.troops() * 0.5);
    if (troops <= 0) return;
    const enemies = borderingEnemies();
    if (enemies.length > 0) {
      const target = action === 2 ? enemies[0] : enemies[enemies.length - 1];
      game.addExecution(new AttackExecution(troops, agent, target.id(), null));
    }
  } else if (action >= 4 && action <= 8) {
    const type = BUILD_ACTIONS[action];
    if (type) tryBuild(type);
  }
  // action === 0 is no-op
}

function step(action: number) {
  applyAction(action);

  for (let i = 0; i < TICKS_PER_STEP; i++) {
    game.executeNextTick();
    if (!agent.isAlive()) break;
    if (game.ticks() >= MAX_TICKS) break;
  }

  // ---- Reward (lives HERE in step(), not reset()) ----
  const tiles = agent.numTilesOwned();
  const ownFrac = tiles / totalLand;
  const enemiesNow = aliveEnemies().length;

  let reward = (ownFrac - prevOwnFrac) * 5.0; // territory share swing
  reward += (prevEnemies - enemiesNow) * 2.0; // +2 per enemy eliminated
  reward -= 0.005; // small time cost

  prevOwnFrac = ownFrac;
  prevEnemies = enemiesNow;

  let done = false;
  let reason = "";
  if (!agent.isAlive()) {
    reward -= 5;
    done = true;
    reason = "agent_died";
  } else if (enemiesNow === 0) {
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
    mask: computeMask(),
    reward,
    done,
    info: { reason, tiles, ticks: game.ticks(), enemies: enemiesNow },
  };
}

// ----------------------------------------------------------------------------
// stdio loop
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