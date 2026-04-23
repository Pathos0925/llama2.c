#!/usr/bin/env node
// Headless episode runner. Usage:
//
//   node src/harness/headless.js \
//     --adapter anthropic \
//     --model claude-haiku-4-5 \
//     --goal  "reach the broken power line in engineering" \
//     --seed  42 \
//     --max-turns 25 \
//     --out trajectories/ep001.jsonl
//
// Adapters:
//   mock       deterministic, no API calls (default)
//   anthropic  requires ANTHROPIC_API_KEY
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

import { buildShip } from '../world/ship-map.js';
import { RobotAgent } from '../agent/Agent.js';
import { MockAdapter, scriptedAction } from '../agent/adapters/mock.js';
import { AnthropicAdapter } from '../agent/adapters/anthropic.js';
import { runEpisode } from './episode.js';

function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) { out[key] = true; }
    else { out[key] = next; i++; }
  }
  return out;
}

function makeAdapter(args) {
  const kind = args.adapter || 'mock';
  if (kind === 'anthropic') {
    return new AnthropicAdapter({ model: args.model || 'claude-haiku-4-5' });
  }
  if (kind === 'mock') {
    // A small deterministic demo script that does: look -> set_task ->
    // move_to player -> talk -> done.
    return new MockAdapter({
      script: [
        scriptedAction('look',         { radius: 3 }, 'orient'),
        scriptedAction('set_task',     { task: 'approach player' }),
        scriptedAction('move_to',      { x: 3, y: 7 }, 'walk toward player'),
        scriptedAction('talk',         { to: 1, msg: 'hello captain' }),
        scriptedAction('update_notes', { notes: 'met the captain at (3,7)' }),
        scriptedAction('done',         { reason: 'greeted player' }),
      ],
    });
  }
  throw new Error(`unknown adapter: ${kind}`);
}

async function main() {
  const args = parseArgs(process.argv);
  const seed = parseInt(args.seed ?? '42', 10);
  const maxTurns = parseInt(args['max-turns'] ?? '25', 10);
  const maxTicks = parseInt(args['max-ticks'] ?? '500', 10);
  const goal = args.goal || 'explore the ship and greet the player';

  const world = buildShip(seed);
  const robot = world.entities.find(e => e.kind === 'robot');
  if (!robot) throw new Error('no robot entity in map');
  world.setGoal(robot, goal);

  const adapter = makeAdapter(args);
  const agent = new RobotAgent({ adapter, entity: robot });

  const { trace, llmTurns, ticks, terminal } = await runEpisode({
    world, agent,
    maxLlmTurns: maxTurns,
    maxTicks,
    onEvent(e) {
      if (args.verbose) {
        const tag = e.type.padEnd(12);
        if (e.type === 'turn')  console.error(`t${e.tick} #${e.llmTurn} ${tag} ${e.action.tool}(${JSON.stringify(e.action.args)}) -> ${e.result.ok ? 'ok' : 'FAIL:' + (e.result.reason||'?')}`);
        else if (e.type === 'auto_step') console.error(`t${e.tick}      ${tag} ${e.result.ok ? 'ok' : 'blocked'}${e.result.done ? ' (done)' : ''}`);
        else if (e.type !== 'episode_start' && e.type !== 'episode_end') console.error(`t${e.tick}      ${tag} ${JSON.stringify(e).slice(0, 120)}`);
      }
    },
  });

  console.error(`\ndone. llm_turns=${llmTurns} ticks=${ticks} terminal=${!!terminal}`);
  if (adapter.usage) {
    const u = adapter.usage;
    console.error(`tokens: in=${u.input} out=${u.output} cache_read=${u.cacheRead} cache_write=${u.cacheWrite}`);
  }

  if (args.out) {
    mkdirSync(dirname(args.out), { recursive: true });
    const jsonl = trace.map(e => JSON.stringify(e)).join('\n') + '\n';
    writeFileSync(args.out, jsonl);
    console.error(`wrote ${trace.length} events -> ${args.out}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
