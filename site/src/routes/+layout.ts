// Fully static, client-rendered showcase: prerender the HTML shell and skip
// SSR (the viewer needs WebGL / window at runtime).
export const prerender = true;
export const ssr = false;
