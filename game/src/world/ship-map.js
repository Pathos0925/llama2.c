import { World } from './World.js';
import { parseChar, TILE } from './tiles.js';
import { makeEntity } from './Entity.js';
import { createRng } from '../rng.js';

// Hand-authored starter ship.
//   # wall    . floor    / door
//   C console = intact power line    x broken power line
//
// 24 wide x 16 tall. Top half has three rooms (bridge / reactor / bay)
// separated by walls with doors, a corridor across the middle, and two
// rooms on the bottom (storage + hibernation).
const RAW_MAP = [
  '########################', // 0
  '#......#.......#.......#', // 1
  '#..C...#.......#..C....#', // 2
  '#......#...=...#.......#', // 3
  '#......#.......#.......#', // 4
  '#......#.......#.......#', // 5
  '###/#######/#######/####', // 6
  '#......................#', // 7  corridor
  '#......................#', // 8  corridor
  '########/####/##########', // 9
  '#...........#..........#', // 10
  '#...x.......#......x...#', // 11
  '#...........#..........#', // 12
  '#...........#....C.....#', // 13
  '#...........#..........#', // 14
  '########################', // 15
];

export function buildShip(seed = 42) {
  const height = RAW_MAP.length;
  const width = RAW_MAP[0].length;
  const tiles = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const row = RAW_MAP[y];
    if (row.length !== width) {
      throw new Error(`ship-map: row ${y} is ${row.length} chars, expected ${width}`);
    }
    for (let x = 0; x < width; x++) {
      tiles[y * width + x] = parseChar(row[x]);
    }
  }

  const entities = [
    makeEntity({ kind: 'player', x: 2,  y: 7, name: 'player', symbol: 'P' }),
    makeEntity({ kind: 'robot',  x: 18, y: 8, name: 'R01',    symbol: 'R',
                 goal: 'await instructions', task: 'idle' }),
    makeEntity({ kind: 'item',   x: 10, y: 10, name: 'wrench', symbol: '!' }),
  ];

  const rng = createRng(seed);
  return new World({ width, height, tiles, entities, rng, seed });
}

// Re-export so callers only need this module.
export { TILE };
