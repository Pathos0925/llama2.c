import Phaser from 'phaser';
import { BootScene } from './scenes/BootScene.js';
import { ShipScene } from './scenes/ShipScene.js';
import { GB_CSS } from './palette.js';

const game = new Phaser.Game({
  type: Phaser.AUTO,
  parent: 'game',
  width: 480,
  height: 320,
  zoom: 2,
  pixelArt: true,
  backgroundColor: GB_CSS.darkest,
  scene: [BootScene, ShipScene],
});

// Handy for tinkering in the browser console.
window.__game = game;
