/**
 * Single source for showcase-specific constants (timing, motion, derived
 * colors). Colors are derived from the shared design tokens in $lib/theme so
 * there are no one-off hex values scattered across components.
 */

// ── Cycle / transition timing (ms) ──────────────────────────────────────
/** How long each model is shown before auto-advancing. */
export const CYCLE_MS = 8000;
/** Model swap: fly-out of the old model (ease-in, accelerating away) and
 *  fly-in of the new one (ease-out, settling on the fit view). */
export const FLY_OUT_MS = 450;
export const FLY_IN_MS = 650;
/** Auto-cycling resumes this long after the last manual interaction. */
export const RESUME_MS = 30000;

// ── Camera motion ───────────────────────────────────────────────────────
/** Idle orbit speed around the model, radians per second. */
export const ORBIT_SPEED = 0.13;
