# Ship RPG — Project Plan

A Game Boy-style top-down RPG set on a derelict spaceship, where NPC robots
are driven by a small LLM (target: sub-100M-parameter Qwen3, running in the
browser via WebGPU). Crew is in hibernation; a virus disabled the maintenance
bots; the player's job is to repair/de-virus robots and, with their help,
fix the ship.

The game and the LLM harness are developed together so the same world API
drives both player input and agent tool calls. Frontier models play episodes
headless to produce the fine-tuning dataset for the small model.

## Roadmap

| Phase | Scope                                                                                   | Status       |
| ----- | --------------------------------------------------------------------------------------- | ------------ |
| 1     | Phaser + Vite scaffold; tile renderer; world API; player movement; one dummy robot      | **done**     |
| 2     | Agent harness: scratchpad, tool registry, prompt/obs/parser, mock + Anthropic adapters, headless episode runner, JSONL trace | **done**     |
| 3     | Data collection: run N episodes with frontier models across goals + virus modifiers; filter; produce training corpus | next         |
| 4     | Fine-tune: pretrain on public tool-use/reasoning data, then SFT on collected trajectories | pending      |
| 5     | Browser deployment: quantize to Q4/Q8, WebGPU inference, in-game "AI plays" mode         | pending      |

## Design decisions locked in

| Decision                                               | Why it matters                                                                 |
| ------------------------------------------------------ | ------------------------------------------------------------------------------ |
| One **World API** for player and agent                 | Same code path on both sides means data collection reuses gameplay logic.      |
| `World` has **zero Phaser imports**                    | Headless LLM harness constructs a world directly; browser renders the same state. |
| Every mutation returns `{ tick, action, ok, ... }` and appends to `world.log` | Replay + training dataset are the same artifact.                             |
| **Two time scales**: LLM turns vs game ticks            | `move_to` consumes many ticks but one LLM turn — tokens stay cheap.            |
| Multi-tile `move_to` + **interrupts on new entity in sight** | Agents react to the world without wasting a call per tile.                  |
| Bounded scratchpad: `GOAL ≤120`, `TASK ≤100`, `NOTES ≤500` chars | Forces small models to use context deliberately instead of rambling.      |
| Response format `<think>…</think><action>{json}</action>` | Small-model-friendly; parser tolerates surrounding prose / missing `<think>`. |
| **4096-token target context** for the small LLM         | Matches typical tiny-LLM training budgets; drives all size decisions.           |
| Deterministic world (seeded `mulberry32` RNG)           | Replays and regression tests are bit-identical.                                |
| Anthropic adapter uses **top-level prompt caching**     | Wired now, silently no-ops under Haiku 4.5's 4096-token minimum; engages as prompts grow. |
| **Procedural textures** in `BootScene`                  | No binary assets in the repo.                                                  |

## Tools exposed to the agent

| Tool           | Signature                             | Notes                                               |
| -------------- | ------------------------------------- | --------------------------------------------------- |
| `look`         | `look(radius?:1..3)`                  | Returns a (2r+1)² grid. Default r=3 (7×7).          |
| `move_to`      | `move_to(x, y)`                       | A* pathfind; multi-tile; halts on block or new entity in sight. |
| `step`         | `step(dir:N\|S\|E\|W)`                | Single-tile move.                                   |
| `talk`         | `talk(to:int, msg:str≤200)`           | Delivered to target's INBOX.                        |
| `take`         | `take(item:int)`                      | Adjacent-only.                                      |
| `give`         | `give(to:int, item:int)`              | Adjacent-only.                                      |
| `set_task`     | `set_task(task:str≤100)`              | Updates scratchpad TASK.                            |
| `update_notes` | `update_notes(notes:str≤500)`         | Replaces scratchpad NOTES (only persistent memory across turns). |
| `wait`         | `wait()`                              | Skip a turn.                                        |
| `done`         | `done(reason:str≤200)`                | Terminates the episode.                             |

## Observation format

Each turn, the agent sees exactly:

```
TICK n   POS (x,y)
GOAL: …
TASK: …
NOTES: …

LOOK r=3:
  <7 lines of single-char symbols>

NEAR:
  <name> (id=N) <kind> @ (x,y)     ← only entities within radius
  ...

INBOX:
  from <name>(id=N): <msg>         ← only messages since last turn
  ...
```

Single-char symbols:

```
.  floor        #  wall          /  door
C  console      =  power line    x  broken power line
@  you          R  robot         P  player        !  item
```

## Repository layout

```
game/
├── package.json, vite.config.js, index.html
├── PLAN.md                        ← this file
├── src/
│   ├── main.js                    ← Phaser boot
│   ├── palette.js                 ← GB DMG palette
│   ├── rng.js                     ← mulberry32 (seedable)
│   ├── world/
│   │   ├── tiles.js               ← tile ids + symbols + passability
│   │   ├── Entity.js              ← entity factory (scratchpad first-class)
│   │   ├── World.js               ← authoritative state, Phaser-free
│   │   │                             look / step / findPath / moveTo / advanceMove
│   │   │                             talk / take / give / setGoal / setTask / updateNotes
│   │   │                             entitiesInRadius / inbox
│   │   └── ship-map.js            ← 24×16 ship, buildShip(seed)
│   ├── scenes/
│   │   ├── BootScene.js           ← procedural tile + sprite textures
│   │   └── ShipScene.js           ← renders World, wires arrows/WASD
│   ├── agent/
│   │   ├── scratchpad.js          ← limits (goal/task/notes)
│   │   ├── tools.js               ← tool registry + executor
│   │   ├── prompt.js              ← system prompt, observation formatter, response parser
│   │   ├── Agent.js               ← RobotAgent: adapter + history window + act()
│   │   └── adapters/
│   │       ├── base.js            ← interface contract
│   │       ├── mock.js            ← deterministic scripted adapter
│   │       └── anthropic.js       ← /v1/messages via fetch, top-level cache_control
│   └── harness/
│       ├── episode.js             ← runs world+agent; auto-walk, interrupts, trace
│       ├── headless.js            ← CLI entry
│       └── __smoke__.js           ← end-to-end tests
```

## How to run

```bash
cd game
npm install

# Browser — player-controlled, for manual testing
npm run dev                        # http://localhost:5173, arrows/WASD

# Headless smoke tests (no API, ~1s)
npm run smoke

# Headless deterministic demo (no API)
npm run play:mock

# Headless agent play (needs ANTHROPIC_API_KEY; default model claude-haiku-4-5)
npm run play:claude -- \
  --goal "walk to the wrench at (10,10) and pick it up" \
  --max-turns 25 \
  --out trajectories/ep001.jsonl
```

Trajectory JSONL is the per-event log from `episode.js` — one line per
game event (turn, auto_step, interrupt, parse_error, adapter_error,
episode_start, episode_end). Training data extractor (Phase 4) will filter
to LLM turns.

## What Phase 3 needs

- **Scenario library**: a handful of parameterised goals (`patrol`, `fetch item`, `repair power line`, `deliver X to Y`, `build base`) plus virus modifiers (slow, forgetful, hoarder, greedy, chatty) driven from the CLI.
- **Additional adapters**: OpenAI-compatible (for OpenRouter / xAI) and Gemini, so we can sample across model families. Anthropic is already wired.
- **Episode batch runner**: parallelise N episodes per scenario, write per-episode JSONL under `trajectories/<scenario>/<model>/ep<NNN>.jsonl`.
- **Quality filter**: heuristic pass rejecting episodes with too many `parse_error`s, pure-`wait` loops, or no progress toward goal.
- **Evaluation set**: ~20 fixed scenarios with success criteria. Run after every fine-tune to measure progress.

Deferred to later phases (explicitly out of scope for Phase 3):
record/replay UI, LoRA per robot, richer item/crafting tree beyond the
replicator shortcut, multi-robot coordination beyond `talk`.

## Key gotchas

- **Prompt caching on Haiku 4.5** requires a ≥4096-token prefix. The current
  system prompt is ~600 tokens, so `cache_control` silently no-ops until the
  prompt + history grows. Expected — no code change needed.
- **Silent cache invalidators**: never put `Date.now()`, UUIDs, or
  per-episode IDs into the system prompt. The agent's system prompt is built
  once per episode from static inputs only.
- **Interrupts require re-issued `move_to`**: when auto-walk halts on a new
  entity, the LLM must decide what to do next. Frontier models handle this
  naturally; scripted mock responses need explicit re-issues (see
  `__smoke__.js` for an example).
- **Entity symbol collisions**: the observing entity always renders as `@`;
  other entities use their own `symbol` field. Don't reuse `@` on any other
  entity.
