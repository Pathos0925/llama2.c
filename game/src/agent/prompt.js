import { toolsSchemaText } from './tools.js';

// The response format is deliberately minimal so a small model can learn it.
// The agent must emit exactly:
//
//   <think>optional short reasoning</think>
//   <action>{"tool":"<name>","args":{...}}</action>
//
// Parser accepts an action block without <think>, and tolerates prose
// around the blocks (frontier models sometimes add commentary).

export function buildSystemPrompt({ robotName = 'R01', radius = 3 } = {}) {
  return [
    `You are ${robotName}, a maintenance robot on a derelict spaceship.`,
    `The crew was in hibernation; you and other bots kept the ship running`,
    `until a virus disabled most of the fleet. You follow your given GOAL.`,
    '',
    'Emit exactly one action per turn in this format:',
    '<think>brief reasoning</think>',
    '<action>{"tool":"<name>","args":{...}}</action>',
    '',
    'Nothing outside the two tags will be parsed.',
    '',
    'Map legend (single-char tiles and entities):',
    '  .  floor        #  wall          /  door',
    '  C  console      =  power line    x  broken power line',
    '  @  you (self)   R  robot         P  player         !  item',
    '',
    `Observations show a ${2 * radius + 1}x${2 * radius + 1} grid centred on you.`,
    'Anything outside is unknown. You also receive the list of entity ids',
    'in view (so tools taking an id work), an INBOX, and your SCRATCHPAD.',
    '',
    'Scratchpad rules:',
    '  GOAL  stays fixed (the reason you exist this episode).',
    '  TASK  is short-term (<=100 chars), what you are doing right now.',
    '  NOTES is your only persistent memory across turns (<=500 chars).',
    '        If a fact matters later, put it in NOTES.',
    '',
    'Tools:',
    toolsSchemaText(),
    '',
    'Guidelines:',
    '- Call look() early if the grid is mostly unknown. Default r=3 is fine.',
    '- Prefer move_to over multiple step() — it is multi-tile and cheap.',
    '- Use set_task when your subgoal changes. Use update_notes to remember.',
    '- Call done() only when the goal is genuinely complete.',
  ].join('\n');
}

// Format the per-turn observation the agent sees as the user message.
export function buildObservation({ world, entity, sinceTick = 0, radius = 3 }) {
  const sp = entity.scratchpad;
  const grid = world.look(entity, radius);
  const near = world.entitiesInRadius(entity, radius).map(e =>
    `  ${e.name} (id=${e.id}) ${e.kind} @ (${e.x},${e.y})`
  );
  const inbox = world.inbox(entity, sinceTick).map(m => {
    const from = world.entityById(m.from);
    const label = from ? `${from.name}(id=${from.id})` : `#${m.from}`;
    return `  from ${label}: ${m.msg}`;
  });

  const lines = [
    `TICK ${world.tick}   POS (${entity.x},${entity.y})`,
    `GOAL: ${sp.goal || '(none)'}`,
    `TASK: ${sp.task || '(none)'}`,
    `NOTES: ${sp.notes || '(empty)'}`,
    '',
    `LOOK r=${radius}:`,
    ...grid.map(r => '  ' + r),
    '',
    'NEAR:',
    ...(near.length ? near : ['  (none in view)']),
    '',
    'INBOX:',
    ...(inbox.length ? inbox : ['  (empty)']),
  ];
  return lines.join('\n');
}

// Parse an assistant response. Returns
//   { ok: true,  think, action: { tool, args } }
//   { ok: false, reason, text? }
export function parseResponse(text) {
  if (typeof text !== 'string') return { ok: false, reason: 'not_string' };

  const think = (text.match(/<think>([\s\S]*?)<\/think>/)?.[1] || '').trim();
  const actionRaw = text.match(/<action>([\s\S]*?)<\/action>/)?.[1];

  let jsonStr = actionRaw?.trim();
  if (!jsonStr) {
    // fallback: last top-level JSON object containing "tool"
    const m = text.match(/\{[^{}]*"tool"[^{}]*(\{[^{}]*\})?[^{}]*\}/);
    jsonStr = m?.[0];
  }
  if (!jsonStr) return { ok: false, reason: 'no_action_block' };

  let action;
  try { action = JSON.parse(jsonStr); }
  catch (e) { return { ok: false, reason: 'invalid_json', detail: e.message }; }

  if (!action || typeof action !== 'object' || typeof action.tool !== 'string') {
    return { ok: false, reason: 'missing_tool' };
  }
  if (!action.args || typeof action.args !== 'object') action.args = {};
  return { ok: true, think: think || null, action };
}
