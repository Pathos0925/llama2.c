// Deterministic adapter: returns responses from a scripted list. Once the
// script is exhausted it falls back to a wait action, so episodes always
// terminate cleanly.
export class MockAdapter {
  constructor({ script = [], fallback } = {}) {
    this.script = script;
    this.i = 0;
    this.fallback = fallback || '<action>{"tool":"wait","args":{}}</action>';
    this.calls = [];
  }

  async complete(req) {
    this.calls.push(req);
    const response = this.i < this.script.length ? this.script[this.i++] : this.fallback;
    return response;
  }
}

// Helper for building scripted responses without writing the XML by hand.
export function scriptedAction(tool, args = {}, think = '') {
  const t = think ? `<think>${think}</think>\n` : '';
  return `${t}<action>${JSON.stringify({ tool, args })}</action>`;
}
