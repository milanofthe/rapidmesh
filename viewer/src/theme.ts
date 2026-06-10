/** Minimal token set for the mesh comparison viewer. */

export const canvas = {
	bg: '#14171c'
};

export const colors = {
	text: '#d8dde6',
	textDim: '#8a93a3',
	panelBg: '#1b1f26',
	border: '#2a3038',
	accent: '#e8a33d'
};

/** Muted region palette (region tag 1, 2, 3, ...). */
export const regionPalette: [number, number, number][] = [
	[0.42, 0.5, 0.62],
	[0.36, 0.56, 0.44],
	[0.6, 0.45, 0.38],
	[0.5, 0.42, 0.6],
	[0.58, 0.56, 0.38]
];

/** PEC / tagged sheet faces. */
export const pecColor: [number, number, number] = [0.91, 0.64, 0.24];

export const wireSurface: [number, number, number] = [0.08, 0.09, 0.11];
export const wireTets: [number, number, number] = [0.35, 0.55, 0.75];
