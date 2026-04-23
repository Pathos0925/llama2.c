import { buildObservation, buildSystemPrompt, parseResponse } from './prompt.js';

// RobotAgent owns: the adapter (LLM), the system prompt, and a rolling
// chat history of turns. It does NOT own world state — the episode
// runner drives it.
//
// A "turn" = one LLM call. The history alternates:
//   user: observation text
//   assistant: raw response (<think>...</think><action>...</action>)
// We keep the last `maxTurns` exchanges so the context budget is bounded.
export class RobotAgent {
  constructor({ adapter, entity, systemPrompt, maxTurns = 8, radius = 3 }) {
    if (!adapter) throw new Error('RobotAgent: adapter required');
    if (!entity)  throw new Error('RobotAgent: entity required');
    this.adapter = adapter;
    this.entity = entity;
    this.radius = radius;
    this.systemPrompt = systemPrompt || buildSystemPrompt({
      robotName: entity.name, radius,
    });
    this.history = [];
    this.maxTurns = maxTurns;
    this.lastInboxTick = 0;
  }

  observe(world) {
    return buildObservation({
      world, entity: this.entity,
      sinceTick: this.lastInboxTick, radius: this.radius,
    });
  }

  async act(world) {
    const observation = this.observe(world);
    this.history.push({ role: 'user', content: observation });
    this._trim();

    const response = await this.adapter.complete({
      system: this.systemPrompt,
      messages: this.history.slice(),
      maxTokens: 400,
    });

    this.history.push({ role: 'assistant', content: response });
    this.lastInboxTick = world.tick;

    const parsed = parseResponse(response);
    return { observation, response, parsed };
  }

  // Drop oldest complete turns until under budget. Always keep even number
  // of entries so pairs stay aligned.
  _trim() {
    const maxMsgs = this.maxTurns * 2;
    if (this.history.length <= maxMsgs) return;
    const drop = this.history.length - maxMsgs;
    this.history.splice(0, drop - (drop % 2));
  }
}
