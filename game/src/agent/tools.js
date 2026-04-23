// Tool registry. One source of truth for: the LLM's system-prompt
// description, argument validation, and execution against the World.
//
// Each tool declares a concise `signature` (used in the prompt) and an
// `execute(world, entity, args)` function (used by the episode runner).

const DIRS = { N: [0, -1], S: [0, 1], E: [1, 0], W: [-1, 0] };

export const TOOLS = {
  look: {
    signature: 'look(radius?:1..3)',
    description: 'Return a (2r+1)x(2r+1) grid of your surroundings. Default r=3.',
    execute(world, entity, args) {
      const r = Math.max(1, Math.min(3, args.radius ?? 3));
      return { ok: true, radius: r, grid: world.look(entity, r) };
    },
  },

  move_to: {
    signature: 'move_to(x:int, y:int)',
    description: 'Walk to (x,y). Multi-tile; halts if path blocks or a new entity appears.',
    execute(world, entity, args) {
      const x = args.x | 0, y = args.y | 0;
      return world.moveTo(entity, x, y);
    },
  },

  step: {
    signature: 'step(dir:N|S|E|W)',
    description: 'Move one tile in a cardinal direction.',
    execute(world, entity, args) {
      const d = DIRS[args.dir];
      if (!d) return { ok: false, reason: 'bad_dir' };
      return world.step(entity, d[0], d[1]);
    },
  },

  talk: {
    signature: 'talk(to:int, msg:str<=200)',
    description: 'Send a message to entity id `to`. They see it in their INBOX next turn.',
    execute(world, entity, args) {
      return world.talk(entity, args.to | 0, String(args.msg || '').slice(0, 200));
    },
  },

  take: {
    signature: 'take(item:int)',
    description: 'Pick up an adjacent item by id.',
    execute(world, entity, args) {
      return world.take(entity, args.item | 0);
    },
  },

  give: {
    signature: 'give(to:int, item:int)',
    description: 'Hand an inventory item to an adjacent entity.',
    execute(world, entity, args) {
      return world.give(entity, args.to | 0, args.item | 0);
    },
  },

  set_task: {
    signature: 'set_task(task:str<=100)',
    description: 'Update your short-term TASK (what you are doing *right now*).',
    execute(world, entity, args) {
      return world.setTask(entity, String(args.task || ''));
    },
  },

  update_notes: {
    signature: 'update_notes(notes:str<=500)',
    description: 'Replace your NOTES. Keep only facts you will need again.',
    execute(world, entity, args) {
      return world.updateNotes(entity, String(args.notes || ''));
    },
  },

  wait: {
    signature: 'wait()',
    description: 'Do nothing this turn.',
    execute(world, entity) {
      return world._record({ action: 'wait', entity: entity.id, ok: true });
    },
  },

  done: {
    signature: 'done(reason:str<=200)',
    description: 'Declare your goal complete. Ends the episode.',
    execute(world, entity, args) {
      const r = world._record({
        action: 'done', entity: entity.id,
        reason: String(args.reason || '').slice(0, 200), ok: true,
      });
      return { ...r, terminal: true };
    },
  },
};

export function toolsSchemaText() {
  const lines = [];
  for (const [name, tool] of Object.entries(TOOLS)) {
    lines.push(`  - ${tool.signature}  ${tool.description}`);
  }
  return lines.join('\n');
}

export function executeTool(world, entity, action) {
  const tool = TOOLS[action.tool];
  if (!tool) return { ok: false, reason: 'unknown_tool', tool: action.tool };
  try {
    return tool.execute(world, entity, action.args || {});
  } catch (e) {
    return { ok: false, reason: 'tool_threw', error: e.message };
  }
}
