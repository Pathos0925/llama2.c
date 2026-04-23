import Phaser from 'phaser';
import { buildShip } from '../world/ship-map.js';
import { TILE } from '../world/tiles.js';
import { GB_CSS } from '../palette.js';
import { TILE_SIZE as TS } from './BootScene.js';

const TILE_TEX = {
  [TILE.FLOOR]:        't_floor',
  [TILE.WALL]:         't_wall',
  [TILE.DOOR]:         't_door',
  [TILE.CONSOLE]:      't_console',
  [TILE.POWER_OK]:     't_power',
  [TILE.POWER_BROKEN]: 't_power_broken',
};

const MOVE_COOLDOWN_MS = 140;

export class ShipScene extends Phaser.Scene {
  constructor() { super('Ship'); }

  create() {
    this.world = buildShip(42);
    this.player = this.world.entities.find(e => e.kind === 'player');

    this._drawTiles();
    this._drawEntities();
    this._setupInput();
    this._setupHud();

    this.moveCooldown = 0;

    // Camera: follow the player, clamp to world bounds.
    const W = this.world.width  * TS;
    const H = this.world.height * TS;
    this.cameras.main.setBounds(0, 0, W, H);
    this.cameras.main.startFollow(this.playerSprite, true, 0.2, 0.2);
    this.cameras.main.setBackgroundColor(GB_CSS.darkest);
  }

  _drawTiles() {
    this.tileLayer = this.add.container(0, 0);
    for (let y = 0; y < this.world.height; y++) {
      for (let x = 0; x < this.world.width; x++) {
        const t = this.world.tileAt(x, y);
        const tex = TILE_TEX[t] || 't_floor';
        const img = this.add.image(x * TS + TS / 2, y * TS + TS / 2, tex);
        this.tileLayer.add(img);
      }
    }
  }

  _drawEntities() {
    this.spritesById = new Map();
    for (const e of this.world.entities) {
      const tex = e.kind === 'player' ? 's_player'
                : e.kind === 'item'   ? 's_item'
                :                       's_robot';
      const s = this.add.image(e.x * TS + TS / 2, e.y * TS + TS / 2, tex);
      s.setDepth(10);
      this.spritesById.set(e.id, s);
    }
  }

  _setupInput() {
    this.cursors = this.input.keyboard.createCursorKeys();
    this.wasd = this.input.keyboard.addKeys({
      up: Phaser.Input.Keyboard.KeyCodes.W,
      down: Phaser.Input.Keyboard.KeyCodes.S,
      left: Phaser.Input.Keyboard.KeyCodes.A,
      right: Phaser.Input.Keyboard.KeyCodes.D,
    });
  }

  _setupHud() {
    this.hud = this.add.text(4, 4, '', {
      fontFamily: 'ui-monospace, monospace',
      fontSize: '10px',
      color: GB_CSS.lightest,
      backgroundColor: '#0f380fcc',
      padding: { x: 4, y: 3 },
    }).setScrollFactor(0).setDepth(1000);
  }

  get playerSprite() { return this.spritesById.get(this.player.id); }

  _readInputDir() {
    const c = this.cursors, w = this.wasd;
    if (c.left.isDown  || w.left.isDown)  return [-1, 0];
    if (c.right.isDown || w.right.isDown) return [ 1, 0];
    if (c.up.isDown    || w.up.isDown)    return [ 0,-1];
    if (c.down.isDown  || w.down.isDown)  return [ 0, 1];
    return null;
  }

  _syncSprite(entity) {
    const s = this.spritesById.get(entity.id);
    if (!s) return;
    s.x = entity.x * TS + TS / 2;
    s.y = entity.y * TS + TS / 2;
  }

  update(_time, delta) {
    this.moveCooldown -= delta;
    if (this.moveCooldown <= 0) {
      const dir = this._readInputDir();
      if (dir) {
        const r = this.world.step(this.player, dir[0], dir[1]);
        if (r.ok) this._syncSprite(this.player);
        this.moveCooldown = MOVE_COOLDOWN_MS;
      }
    }

    // Update HUD with a live look() readout — this is exactly what the
    // agent harness will feed to the LLM, so it's useful to eyeball.
    const view = this.world.look(this.player, 3);
    this.hud.setText([
      `tick ${this.world.tick}   pos (${this.player.x},${this.player.y})`,
      'look r=3:',
      ...view,
    ].join('\n'));
  }
}
