import { TILE, TILE_INFO } from './tiles.js';
import { clip } from '../agent/scratchpad.js';

// The World holds all authoritative game state. It is deliberately
// independent of Phaser — a headless LLM-play harness will instantiate
// it directly with no renderer.
//
// All mutations go through named methods (step, moveTo, talk, ...).
// Each mutation appends to `this.log` so episodes can be replayed.
export class World {
  constructor({ width, height, tiles, entities, rng, seed }) {
    this.width = width;
    this.height = height;
    this.tiles = tiles;         // Uint8Array, length = w*h
    this.entities = entities;   // array of entity objects
    this.rng = rng;
    this.seed = seed;
    this.tick = 0;
    this.log = [];              // [{ tick, action, ...args, result }]
    this.messages = [];         // recent talk() events: [{ tick, from, to, msg }]
  }

  // ---- queries ----
  inBounds(x, y) {
    return x >= 0 && y >= 0 && x < this.width && y < this.height;
  }

  tileAt(x, y) {
    if (!this.inBounds(x, y)) return TILE.WALL;
    return this.tiles[y * this.width + x];
  }

  setTile(x, y, t) {
    if (!this.inBounds(x, y)) return;
    this.tiles[y * this.width + x] = t;
  }

  entityAt(x, y) {
    for (const e of this.entities) {
      if (e.x === x && e.y === y) return e;
    }
    return null;
  }

  entityById(id) {
    return this.entities.find(e => e.id === id) || null;
  }

  isPassable(x, y, ignoreEntity = null) {
    const t = this.tileAt(x, y);
    if (!TILE_INFO[t].passable) return false;
    const occ = this.entityAt(x, y);
    if (occ && occ !== ignoreEntity) return false;
    return true;
  }

  // ---- observation ----
  // Returns a (2r+1) x (2r+1) grid of single-char symbols.
  // The observing entity itself is marked '@'; other entities use their own symbol.
  look(entity, radius = 3) {
    const rows = [];
    for (let dy = -radius; dy <= radius; dy++) {
      const row = [];
      for (let dx = -radius; dx <= radius; dx++) {
        const x = entity.x + dx;
        const y = entity.y + dy;
        if (dx === 0 && dy === 0) { row.push('@'); continue; }
        const other = this.entityAt(x, y);
        if (other) { row.push(other.symbol || '?'); continue; }
        row.push(TILE_INFO[this.tileAt(x, y)].symbol);
      }
      rows.push(row.join(''));
    }
    return rows;
  }

  // ---- mutations ----
  step(entity, dx, dy) {
    if (Math.abs(dx) + Math.abs(dy) !== 1) {
      return this._record({ action: 'step', entity: entity.id, dx, dy, ok: false, reason: 'invalid_delta' });
    }
    const nx = entity.x + dx;
    const ny = entity.y + dy;
    if (!this.isPassable(nx, ny, entity)) {
      return this._record({ action: 'step', entity: entity.id, dx, dy, ok: false, reason: 'blocked' });
    }
    entity.x = nx;
    entity.y = ny;
    return this._record({ action: 'step', entity: entity.id, dx, dy, ok: true });
  }

  // A* pathfind. Returns an array of {x,y} from (not including) the start
  // to (including) the goal, or null if unreachable.
  findPath(entity, gx, gy) {
    if (!this.inBounds(gx, gy)) return null;
    if (entity.x === gx && entity.y === gy) return [];
    const key = (x, y) => y * this.width + x;
    const open = new Map();        // key -> { x, y, g, f, parent }
    const closed = new Set();
    const start = { x: entity.x, y: entity.y, g: 0, f: 0, parent: null };
    start.f = Math.abs(gx - start.x) + Math.abs(gy - start.y);
    open.set(key(start.x, start.y), start);

    const DIRS = [[1,0],[-1,0],[0,1],[0,-1]];

    while (open.size) {
      // pop lowest f
      let best = null, bestKey = null;
      for (const [k, v] of open) {
        if (!best || v.f < best.f) { best = v; bestKey = k; }
      }
      open.delete(bestKey);
      closed.add(bestKey);

      if (best.x === gx && best.y === gy) {
        const path = [];
        let cur = best;
        while (cur.parent) { path.push({ x: cur.x, y: cur.y }); cur = cur.parent; }
        path.reverse();
        return path;
      }

      for (const [dx, dy] of DIRS) {
        const nx = best.x + dx, ny = best.y + dy;
        const nk = key(nx, ny);
        if (closed.has(nk)) continue;
        // allow goal tile even if occupied by target; block otherwise
        const isGoal = (nx === gx && ny === gy);
        if (!isGoal && !this.isPassable(nx, ny, entity)) continue;
        if (isGoal && !this.inBounds(nx, ny)) continue;
        const g = best.g + 1;
        const h = Math.abs(gx - nx) + Math.abs(gy - ny);
        const existing = open.get(nk);
        if (existing && existing.g <= g) continue;
        open.set(nk, { x: nx, y: ny, g, f: g + h, parent: best });
      }
    }
    return null;
  }

  // Computes a path and stores it on the entity. The actual stepping
  // happens one tile at a time via advanceMove(), so the caller (renderer
  // or agent loop) can animate or insert interrupts.
  moveTo(entity, gx, gy) {
    const path = this.findPath(entity, gx, gy);
    if (!path) {
      return this._record({ action: 'moveTo', entity: entity.id, gx, gy, ok: false, reason: 'unreachable' });
    }
    entity.path = path;
    return this._record({ action: 'moveTo', entity: entity.id, gx, gy, ok: true, steps: path.length });
  }

  // Consume one step of a prior moveTo. Returns
  //   { ok: true, done, interrupted? }
  // 'interrupted' is true if the next tile became blocked since pathing.
  advanceMove(entity) {
    if (!entity.path || entity.path.length === 0) return { ok: true, done: true };
    const next = entity.path[0];
    if (!this.isPassable(next.x, next.y, entity)) {
      entity.path = null;
      return { ok: false, done: true, interrupted: true, reason: 'blocked' };
    }
    entity.path.shift();
    const dx = next.x - entity.x;
    const dy = next.y - entity.y;
    const r = this.step(entity, dx, dy);
    if (entity.path.length === 0) entity.path = null;
    return { ...r, done: !entity.path };
  }

  // ---- stubs (Phase 1: wired enough to record; Phase 2+ fleshes out) ----
  talk(entity, targetId, msg) {
    const target = this.entityById(targetId);
    if (!target) return this._record({ action: 'talk', entity: entity.id, targetId, msg, ok: false, reason: 'no_target' });
    this.messages.push({ tick: this.tick, from: entity.id, to: target.id, msg });
    return this._record({ action: 'talk', entity: entity.id, targetId, msg, ok: true });
  }

  take(entity, itemId) {
    const item = this.entityById(itemId);
    if (!item || item.kind !== 'item') return this._record({ action: 'take', entity: entity.id, itemId, ok: false, reason: 'no_item' });
    const adj = Math.abs(item.x - entity.x) + Math.abs(item.y - entity.y) <= 1;
    if (!adj) return this._record({ action: 'take', entity: entity.id, itemId, ok: false, reason: 'not_adjacent' });
    this.entities = this.entities.filter(e => e !== item);
    entity.inventory.push(item);
    return this._record({ action: 'take', entity: entity.id, itemId, ok: true });
  }

  // ---- scratchpad mutators (tool-callable) ----
  setGoal(entity, goal) {
    entity.scratchpad.goal = clip('goal', goal);
    return this._record({ action: 'set_goal', entity: entity.id, goal: entity.scratchpad.goal, ok: true });
  }

  setTask(entity, task) {
    entity.scratchpad.task = clip('task', task);
    return this._record({ action: 'set_task', entity: entity.id, task: entity.scratchpad.task, ok: true });
  }

  updateNotes(entity, notes) {
    entity.scratchpad.notes = clip('notes', notes);
    return this._record({ action: 'update_notes', entity: entity.id, notes: entity.scratchpad.notes, ok: true });
  }

  // ---- query helpers for observations ----
  entitiesInRadius(entity, radius) {
    return this.entities.filter(e =>
      e !== entity &&
      Math.abs(e.x - entity.x) <= radius &&
      Math.abs(e.y - entity.y) <= radius
    );
  }

  inbox(entity, sinceTick = 0) {
    return this.messages.filter(m => m.to === entity.id && m.tick > sinceTick);
  }

  give(entity, targetId, itemId) {
    const target = this.entityById(targetId);
    const item = entity.inventory.find(i => i.id === itemId);
    if (!target || !item) return this._record({ action: 'give', entity: entity.id, targetId, itemId, ok: false, reason: 'missing' });
    const adj = Math.abs(target.x - entity.x) + Math.abs(target.y - entity.y) <= 1;
    if (!adj) return this._record({ action: 'give', entity: entity.id, targetId, itemId, ok: false, reason: 'not_adjacent' });
    entity.inventory = entity.inventory.filter(i => i !== item);
    target.inventory.push(item);
    return this._record({ action: 'give', entity: entity.id, targetId, itemId, ok: true });
  }

  // ---- internal ----
  _record(event) {
    this.tick++;
    const rec = { tick: this.tick, ...event };
    this.log.push(rec);
    return rec;
  }
}
