import Phaser from 'phaser';
import { GB } from '../palette.js';

const TS = 16;

// Generate all tile / sprite textures procedurally so the repo stays
// binary-free. Each texture is TSxTS with the classic GB palette.
export class BootScene extends Phaser.Scene {
  constructor() { super('Boot'); }

  create() {
    this.makeTileTextures();
    this.scene.start('Ship');
  }

  _tex(key, draw) {
    const g = this.add.graphics();
    draw(g);
    g.generateTexture(key, TS, TS);
    g.destroy();
  }

  makeTileTextures() {
    // floor: light with a scattering of lightest pixels
    this._tex('t_floor', g => {
      g.fillStyle(GB.light, 1).fillRect(0, 0, TS, TS);
      g.fillStyle(GB.lightest, 1);
      for (const [x, y] of [[3, 5], [10, 2], [12, 11], [5, 13], [14, 8]]) {
        g.fillRect(x, y, 1, 1);
      }
    });

    // wall: dark with a highlight top row and crosshatch
    this._tex('t_wall', g => {
      g.fillStyle(GB.darkest, 1).fillRect(0, 0, TS, TS);
      g.fillStyle(GB.dark, 1).fillRect(0, 0, TS, 2);
      g.fillStyle(GB.dark, 1);
      for (let y = 4; y < TS; y += 4) g.fillRect(0, y, TS, 1);
      for (let x = 4; x < TS; x += 4) g.fillRect(x, 2, 1, TS - 2);
    });

    // door: floor with a vertical slit
    this._tex('t_door', g => {
      g.fillStyle(GB.light, 1).fillRect(0, 0, TS, TS);
      g.fillStyle(GB.dark, 1).fillRect(7, 0, 2, TS);
      g.fillStyle(GB.darkest, 1).fillRect(0, 0, TS, 1);
      g.fillStyle(GB.darkest, 1).fillRect(0, TS - 1, TS, 1);
    });

    // console: blocky dark with a light "screen"
    this._tex('t_console', g => {
      g.fillStyle(GB.light, 1).fillRect(0, 0, TS, TS);
      g.fillStyle(GB.darkest, 1).fillRect(2, 3, TS - 4, TS - 6);
      g.fillStyle(GB.lightest, 1).fillRect(4, 5, TS - 8, TS - 10);
      g.fillStyle(GB.dark, 1).fillRect(2, TS - 3, TS - 4, 2);
    });

    // power line (intact): horizontal cable
    this._tex('t_power', g => {
      g.fillStyle(GB.light, 1).fillRect(0, 0, TS, TS);
      g.fillStyle(GB.darkest, 1).fillRect(0, 7, TS, 2);
      g.fillStyle(GB.lightest, 1).fillRect(0, 9, TS, 1);
    });

    // broken power line: gap + sparks
    this._tex('t_power_broken', g => {
      g.fillStyle(GB.light, 1).fillRect(0, 0, TS, TS);
      g.fillStyle(GB.darkest, 1).fillRect(0, 7, 5, 2);
      g.fillStyle(GB.darkest, 1).fillRect(TS - 5, 7, 5, 2);
      g.fillStyle(GB.dark, 1).fillRect(6, 4, 1, 1);
      g.fillStyle(GB.dark, 1).fillRect(9, 11, 1, 1);
      g.fillStyle(GB.dark, 1).fillRect(7, 9, 2, 1);
    });

    // player sprite: small figure
    this._tex('s_player', g => {
      g.fillStyle(GB.darkest, 1).fillRect(5, 2, 6, 4);   // head
      g.fillStyle(GB.dark,    1).fillRect(4, 6, 8, 6);   // body
      g.fillStyle(GB.darkest, 1).fillRect(5, 12, 2, 3);  // left leg
      g.fillStyle(GB.darkest, 1).fillRect(9, 12, 2, 3);  // right leg
      g.fillStyle(GB.lightest,1).fillRect(7, 3, 2, 1);   // eyes
    });

    // item sprite: small highlighted thing
    this._tex('s_item', g => {
      g.fillStyle(GB.darkest, 1).fillRect(5, 5, 6, 6);
      g.fillStyle(GB.lightest, 1).fillRect(6, 6, 4, 4);
      g.fillStyle(GB.dark, 1).fillRect(7, 11, 2, 2);
    });

    // robot sprite: boxier
    this._tex('s_robot', g => {
      g.fillStyle(GB.dark,    1).fillRect(4, 1, 8, 5);   // head
      g.fillStyle(GB.lightest,1).fillRect(6, 3, 1, 1);
      g.fillStyle(GB.lightest,1).fillRect(9, 3, 1, 1);
      g.fillStyle(GB.darkest, 1).fillRect(3, 6, 10, 7);  // chassis
      g.fillStyle(GB.dark,    1).fillRect(5, 9, 6, 2);   // vent
      g.fillStyle(GB.darkest, 1).fillRect(3, 13, 3, 2);
      g.fillStyle(GB.darkest, 1).fillRect(10, 13, 3, 2);
    });
  }
}

export const TILE_SIZE = TS;
