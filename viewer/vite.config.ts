import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

// `$lib` mirrors the SvelteKit alias so the rapidfem library files
// (canvas3d.ts, theme.ts, components) work verbatim.
export default defineConfig({
	plugins: [svelte()],
	resolve: {
		alias: {
			$lib: fileURLToPath(new URL('./src/lib', import.meta.url))
		}
	},
	server: {
		port: 5199
	}
});
