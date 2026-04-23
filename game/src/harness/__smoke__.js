// End-to-end smoke test for Phase 2. No API calls; uses MockAdapter.
// Run: npm run smoke
import { buildShip } from '../world/ship-map.js';
import { RobotAgent } from '../agent/Agent.js';
import { MockAdapter, scriptedAction } from '../agent/adapters/mock.js';
import { runEpisode } from './episode.js';
import { parseResponse, buildSystemPrompt, buildObservation } from '../agent/prompt.js';

function assert(cond, msg) { if (!cond) { console.error('FAIL:', msg); process.exit(1); } else { console.log('ok', msg); } }

// ---- unit: parseResponse ----
{
  const good = `<think>plan</think>\n<action>{"tool":"look","args":{"radius":2}}</action>`;
  const p = parseResponse(good);
  assert(p.ok && p.action.tool === 'look' && p.action.args.radius === 2 && p.think === 'plan', 'parseResponse: happy path');

  const noThink = `<action>{"tool":"wait","args":{}}</action>`;
  const p2 = parseResponse(noThink);
  assert(p2.ok && p2.action.tool === 'wait' && p2.think === null, 'parseResponse: action-only');

  const prose = `I think I should move. <action>{"tool":"step","args":{"dir":"N"}}</action> Going north now.`;
  const p3 = parseResponse(prose);
  assert(p3.ok && p3.action.tool === 'step', 'parseResponse: tolerates surrounding prose');

  const bad = `I have no plan.`;
  const p4 = parseResponse(bad);
  assert(!p4.ok && p4.reason === 'no_action_block', 'parseResponse: detects missing action');
}

// ---- unit: prompt + observation shape ----
{
  const w = buildShip(42);
  const robot = w.entities.find(e => e.kind === 'robot');
  w.setGoal(robot, 'test-goal');
  const sys = buildSystemPrompt({ robotName: robot.name });
  assert(sys.includes('R01') && sys.includes('<action>'), 'system prompt contains robot name and format tags');
  const obs = buildObservation({ world: w, entity: robot, radius: 3 });
  assert(obs.includes('GOAL: test-goal') && obs.includes('LOOK r=3:'), 'observation contains goal and look block');
}

// ---- integration: uninterrupted multi-tile move ----
// Robot starts (18,8). Walks west to (15,8) — no entities within r=3
// along that path, so auto-walk completes without interrupts.
{
  const world = buildShip(42);
  const robot = world.entities.find(e => e.kind === 'robot');
  const player = world.entities.find(e => e.kind === 'player');
  world.setGoal(robot, 'patrol corridor');

  const adapter = new MockAdapter({
    script: [
      scriptedAction('look',     { radius: 3 }),
      scriptedAction('set_task', { task: 'walk west' }),
      scriptedAction('move_to',  { x: 15, y: 8 }),
      scriptedAction('talk',     { to: player.id, msg: 'status nominal' }),
      scriptedAction('update_notes', { notes: 'reported to captain' }),
      scriptedAction('done',     { reason: 'patrol leg complete' }),
    ],
  });
  const agent = new RobotAgent({ adapter, entity: robot });
  const result = await runEpisode({ world, agent, maxLlmTurns: 20 });

  assert(result.terminal && result.terminal.action === 'done', 'episode ended via done()');
  assert(result.llmTurns === 6, `used 6 LLM turns (got ${result.llmTurns})`);
  assert(robot.x === 15 && robot.y === 8, `robot walked to (15,8); got (${robot.x},${robot.y})`);
  assert(robot.scratchpad.task === 'walk west', 'task set via tool');
  assert(robot.scratchpad.notes.startsWith('reported'), 'notes updated via tool');
  const msg = world.messages.find(m => m.from === robot.id && m.to === player.id);
  assert(msg && msg.msg.includes('nominal'), 'talk recorded');

  const turns = result.trace.filter(e => e.type === 'turn').length;
  const autos = result.trace.filter(e => e.type === 'auto_step').length;
  const interrupts = result.trace.filter(e => e.type === 'interrupt').length;
  assert(turns === 6, `6 turn events (got ${turns})`);
  assert(autos === 3, `3 auto_step events for 3-tile walk (got ${autos})`);
  assert(interrupts === 0, `no interrupts (got ${interrupts})`);

  // determinism: re-running produces identical tick and turn counts
  const w2 = buildShip(42);
  const r2 = w2.entities.find(e => e.kind === 'robot');
  const p2 = w2.entities.find(e => e.kind === 'player');
  w2.setGoal(r2, 'patrol corridor');
  const a2 = new MockAdapter({
    script: [
      scriptedAction('look',     { radius: 3 }),
      scriptedAction('set_task', { task: 'walk west' }),
      scriptedAction('move_to',  { x: 15, y: 8 }),
      scriptedAction('talk',     { to: p2.id, msg: 'status nominal' }),
      scriptedAction('update_notes', { notes: 'reported to captain' }),
      scriptedAction('done',     { reason: 'patrol leg complete' }),
    ],
  });
  const r2r = await runEpisode({ world: w2, agent: new RobotAgent({ adapter: a2, entity: r2 }) });
  assert(r2r.ticks === result.ticks && r2r.llmTurns === result.llmTurns, 'deterministic on re-run');
}

// ---- integration: auto-walk halts when a new entity enters view ----
// Robot starts (18,8). Walks to (3,7), which crosses the item at (10,10).
// The item enters the radius-3 window at (13,8), triggering an interrupt.
// A second move_to then completes the rest (once item is already in view,
// no further 'new entity' events fire for it).
{
  const world = buildShip(42);
  const robot = world.entities.find(e => e.kind === 'robot');
  world.setGoal(robot, 'reach bridge');

  const adapter = new MockAdapter({
    script: [
      scriptedAction('move_to', { x: 3, y: 7 }),  // will be interrupted
      scriptedAction('move_to', { x: 3, y: 7 }),  // resume after interrupt
      scriptedAction('done',    { reason: 'arrived' }),
    ],
  });
  const agent = new RobotAgent({ adapter, entity: robot });
  const r = await runEpisode({ world, agent, maxLlmTurns: 10 });

  const interrupts = r.trace.filter(e => e.type === 'interrupt');
  assert(interrupts.length >= 1, `at least one interrupt fired (got ${interrupts.length})`);
  assert(r.terminal && r.terminal.action === 'done', 'episode terminated');
}

// ---- unit: bad actions fail softly (don't crash the episode) ----
{
  const world = buildShip(42);
  const robot = world.entities.find(e => e.kind === 'robot');
  const adapter = new MockAdapter({
    script: [
      '<action>{"tool":"move_to","args":{"x":0,"y":0}}</action>',  // (0,0) is wall
      scriptedAction('wait', {}),
      scriptedAction('done', { reason: 'fine' }),
    ],
  });
  const agent = new RobotAgent({ adapter, entity: robot });
  const r = await runEpisode({ world, agent, maxLlmTurns: 10 });
  assert(r.terminal && r.terminal.action === 'done', 'episode survives a bad tool call');
}

console.log('\nall smoke tests passed.');
