import { executeTool } from '../agent/tools.js';

// Run one episode against a world and a RobotAgent. The loop has two
// time scales:
//
//   - LLM turn: one Agent.act() call. Counted toward `maxLlmTurns`.
//   - Game tick: one world mutation. Counted toward `maxTicks`.
//
// Multi-tile moves auto-consume via advanceMove() between LLM turns —
// no LLM call per tile. An auto-walk halts on:
//   - path blocked
//   - a new entity entered the look radius (interrupt re-prompts)
//   - path consumed
//
// Returns { trace, llmTurns, ticks, terminal }
// `trace` is the JSONL-ready per-event log.
export async function runEpisode({
  world, agent,
  maxLlmTurns = 30,
  maxTicks    = 500,
  onEvent     = () => {},
}) {
  const trace = [];
  let llmTurns = 0;
  let terminal = null;

  const push = (e) => {
    const stamped = { ts: Date.now(), tick: world.tick, ...e };
    trace.push(stamped);
    onEvent(stamped);
    return stamped;
  };

  push({ type: 'episode_start', entity: agent.entity.id, goal: agent.entity.scratchpad.goal });

  while (llmTurns < maxLlmTurns && world.tick < maxTicks && !terminal) {
    // Auto-walk: if the agent has a path, take one step without calling the LLM.
    if (agent.entity.path) {
      const pre = new Set(world.entitiesInRadius(agent.entity, agent.radius).map(e => e.id));
      const r = world.advanceMove(agent.entity);
      push({ type: 'auto_step', result: r });
      if (!r.ok || r.done) {
        // path finished or blocked — next loop iteration will call the LLM
        continue;
      }
      // interrupt on new entity in sight
      const post = new Set(world.entitiesInRadius(agent.entity, agent.radius).map(e => e.id));
      for (const id of post) {
        if (!pre.has(id)) {
          agent.entity.path = null;
          push({ type: 'interrupt', reason: 'entity_entered', entity_id: id });
          break;
        }
      }
      continue;
    }

    // LLM turn.
    llmTurns++;
    let act;
    try {
      act = await agent.act(world);
    } catch (e) {
      push({ type: 'adapter_error', llmTurn: llmTurns, error: e.message });
      break;
    }
    const { observation, response, parsed } = act;

    if (!parsed.ok) {
      push({ type: 'parse_error', llmTurn: llmTurns, observation, response, error: parsed.reason });
      continue;
    }

    const result = executeTool(world, agent.entity, parsed.action);
    push({
      type: 'turn',
      llmTurn: llmTurns,
      observation,
      think: parsed.think,
      action: parsed.action,
      response,
      result,
    });

    if (result.terminal) {
      terminal = result;
      break;
    }
  }

  push({ type: 'episode_end', llmTurns, ticks: world.tick, terminal: !!terminal });
  return { trace, llmTurns, ticks: world.tick, terminal };
}
