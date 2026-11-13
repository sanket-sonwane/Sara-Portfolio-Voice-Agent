/**
 * Pure turn-timing calculations.
 *
 * These live outside the controller so they can be tested without a browser,
 * and so the rule for each number is visible in one place instead of buried in
 * an event handler.
 */

/**
 * The visitor's own end-of-speech → audible latency, in milliseconds.
 *
 * Both stamps are `performance.now()` values from this browser. The number is
 * only meaningful when both belong to the same turn and audio came after the
 * speech ended; a typed turn, a greeting, or a barge-in has no pair and must
 * report `undefined` rather than a plausible-looking zero.
 */
export function clientEosToAudible(
  eos: number | null | undefined,
  audible: number | null | undefined,
): number | undefined {
  if (eos === null || eos === undefined) return undefined;
  if (audible === null || audible === undefined) return undefined;
  if (!(audible >= eos)) return undefined;
  return Math.round(audible - eos);
}
