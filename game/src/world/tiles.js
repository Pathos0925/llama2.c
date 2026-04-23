export const TILE = {
  FLOOR:        0,
  WALL:         1,
  DOOR:         2,
  CONSOLE:      3,
  POWER_OK:     4,
  POWER_BROKEN: 5,
};

// symbol: single-char rep for text/grid output (for LLM look() and debug).
// passable: can an entity stand on it.
export const TILE_INFO = {
  [TILE.FLOOR]:        { passable: true,  symbol: '.', label: 'floor' },
  [TILE.WALL]:         { passable: false, symbol: '#', label: 'wall' },
  [TILE.DOOR]:         { passable: true,  symbol: '/', label: 'door' },
  [TILE.CONSOLE]:      { passable: false, symbol: 'C', label: 'console' },
  [TILE.POWER_OK]:     { passable: true,  symbol: '=', label: 'power_line' },
  [TILE.POWER_BROKEN]: { passable: true,  symbol: 'x', label: 'broken_power_line' },
};

// Parse a single char in a RAW_MAP string into a tile id.
export function parseChar(ch) {
  switch (ch) {
    case '#': return TILE.WALL;
    case '.': return TILE.FLOOR;
    case '/': return TILE.DOOR;
    case 'C': return TILE.CONSOLE;
    case '=': return TILE.POWER_OK;
    case 'x': return TILE.POWER_BROKEN;
    default:  return TILE.FLOOR;
  }
}
