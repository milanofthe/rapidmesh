/**
 * Single source for showcase-specific constants (timing, motion, derived
 * colors). Colors are derived from the shared design tokens in $lib/theme so
 * there are no one-off hex values scattered across components.
 */

import { palette } from '$lib/theme';

// ── Cycle / transition timing (ms) ──────────────────────────────────────
/** How long each model is shown before auto-advancing. */
export const CYCLE_MS = 8000;
/** Fade-to-black duration for one half of a model transition. */
export const FADE_MS = 650;
/** Auto-cycling resumes this long after the last manual interaction. */
export const RESUME_MS = 30000;

// ── Camera motion ───────────────────────────────────────────────────────
/** Idle orbit speed around the model, radians per second. */
export const ORBIT_SPEED = 0.13;

// ── Renderer colors (derived from theme tokens) ─────────────────────────
/** Surface fill is dimmed so the bright wireframe overlay reads clearly. */
export const SURFACE_FILL_DIM = 0.45;
/** Surface wireframe color: the light body text token, so the mesh pops. */
export const SURFACE_WIRE_COLOR = palette.text;
