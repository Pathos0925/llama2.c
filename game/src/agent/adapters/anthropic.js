// Anthropic /v1/messages adapter using Node's built-in fetch.
//
// Caching strategy: top-level auto-caching. The API places one
// cache_control marker on the last cacheable block each request, which
// caches the full prefix (system + message history) up to that point.
// On Haiku 4.5 the minimum cacheable prefix is 4096 tokens — under that
// it silently no-ops, so early short episodes won't see cache hits.
//
// Silent invalidators to avoid upstream: timestamps, UUIDs, or any
// per-request bytes in the system prompt or early message content.
export class AnthropicAdapter {
  constructor({
    apiKey = process.env.ANTHROPIC_API_KEY,
    model = 'claude-haiku-4-5',
    baseUrl = 'https://api.anthropic.com',
    debug = !!process.env.ADAPTER_DEBUG,
  } = {}) {
    if (!apiKey) throw new Error('AnthropicAdapter: ANTHROPIC_API_KEY not set');
    this.apiKey = apiKey;
    this.model = model;
    this.baseUrl = baseUrl;
    this.debug = debug;
    this.usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
  }

  async complete({ system, messages, maxTokens = 400 }) {
    const body = {
      model: this.model,
      max_tokens: maxTokens,
      system,
      messages,
      cache_control: { type: 'ephemeral' },
    };

    const res = await fetch(`${this.baseUrl}/v1/messages`, {
      method: 'POST',
      headers: {
        'x-api-key': this.apiKey,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Anthropic ${res.status}: ${text.slice(0, 500)}`);
    }

    const data = await res.json();
    const u = data.usage || {};
    this.usage.input      += u.input_tokens         || 0;
    this.usage.output     += u.output_tokens        || 0;
    this.usage.cacheRead  += u.cache_read_input_tokens    || 0;
    this.usage.cacheWrite += u.cache_creation_input_tokens || 0;

    if (this.debug) {
      console.error(`[adapter] in=${u.input_tokens} out=${u.output_tokens} cache_read=${u.cache_read_input_tokens ?? 0} cache_write=${u.cache_creation_input_tokens ?? 0}`);
    }

    const textBlock = (data.content || []).find(b => b.type === 'text');
    return textBlock?.text || '';
  }
}
