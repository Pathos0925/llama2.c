// Adapter interface doc. Every adapter implements:
//
//   async complete({ system, messages, maxTokens, stop? }) -> string
//
// where `messages` is an array of { role: 'user' | 'assistant', content: str }.
// Adapters translate this to whatever the underlying provider wants and
// return just the assistant text.
//
// Failures should throw; callers treat that as "retry later".
export {};
