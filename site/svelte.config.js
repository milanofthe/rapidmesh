import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// adapter-static emits plain files for GitHub Pages. The custom domain
// (mesh.rapidpassives.org via the static/CNAME file) serves from root, so no
// base path is needed. A single prerendered route ships index.html + assets.
export default {
	preprocess: vitePreprocess(),
	kit: {
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: undefined,
			precompress: false,
			strict: true
		})
	}
};
