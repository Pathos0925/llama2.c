let nextId = 1;

// The scratchpad is intentionally small. Enforced by World.setGoal /
// setTask / updateNotes; keep limits in sync with src/agent/scratchpad.js.
export function makeEntity({ kind, x, y, name, symbol, goal, task, notes, meta }) {
  return {
    id: nextId++,
    kind,                       // 'player' | 'robot' | 'item'
    name: name || kind,
    symbol: symbol || '?',
    x, y,
    inventory: [],
    scratchpad: {
      goal:  goal  || '',
      task:  task  || '',
      notes: notes || '',
    },
    path: null,                 // set by moveTo, consumed by advanceMove
    meta: meta || {},
  };
}
