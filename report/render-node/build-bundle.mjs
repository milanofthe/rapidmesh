import * as esbuild from 'esbuild'
import path from 'node:path'
const LIB = path.resolve('../../site/src/lib')
await esbuild.build({
  entryPoints: ['entry.ts'], bundle: true, format: 'esm', platform: 'node',
  outfile: 'bundle.mjs', logLevel: 'info',
  alias: { '$lib': LIB },
})
console.log('bundle.mjs built')
