// WGSL shaders for the WebGPU mesh renderer -- a faithful 1:1 port of the
// WebGL2 GLSL in `canvas3d.ts` (mesh + line passes). Shared by the browser
// (navigator.gpu) and the headless Node rasterizer (@kmamal/gpu), so there is
// ONE renderer for live + offscreen.
//
// Bind-group split mirrors the original's per-frame vs per-draw uniforms:
//   @group(0) = per-frame  (camera/light/clip, set once per render3D)
//   @group(1) = per-object (the mesh's flat color + colormap flag, or line color)
// so render3D draws each addMesh/addLineMesh with its OWN color exactly like the
// WebGL2 path (no per-vertex colour -- bit-for-bit the same look).

export const MESH_WGSL = /* wgsl */ `
struct Frame {
  mvp        : mat4x4<f32>,
  normalMat  : mat3x3<f32>,
  lightDir   : vec3<f32>,
  ambient    : f32,
  clipPlane  : vec4<f32>,
  clipEnable : f32,
  zFlip      : f32,
};
struct Obj { color : vec3<f32>, colormap : f32 };
@group(0) @binding(0) var<uniform> F : Frame;
@group(1) @binding(0) var<uniform> O : Obj;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) normal : vec3<f32>,
  @location(1) scalar : f32,
  @location(2) world  : vec3<f32>,
};

@vertex
fn vs(@location(0) aPos : vec3<f32>, @location(1) aNormal : vec3<f32>, @location(2) aScalar : f32) -> VSOut {
  var o : VSOut;
  var n = aNormal; n.z = n.z * F.zFlip;
  o.normal = normalize(F.normalMat * n);
  var p = aPos; p.z = p.z * F.zFlip;
  o.world = p;
  o.pos = F.mvp * vec4<f32>(p, 1.0);
  o.scalar = aScalar;
  return o;
}

fn inferno(t0 : f32) -> vec3<f32> {
  let t = clamp(t0, 0.0, 1.0);
  let c0 = vec3<f32>(0.0002, 0.0016, -0.0194);
  let c1 = vec3<f32>(0.1065, 0.5639, 3.9327);
  let c2 = vec3<f32>(11.6024, -3.972, -15.9423);
  let c3 = vec3<f32>(-41.7039, 17.4363, 44.354);
  let c4 = vec3<f32>(77.1629, -33.4023, -81.8073);
  let c5 = vec3<f32>(-71.319, 32.6261, 73.2095);
  let c6 = vec3<f32>(25.1311, -12.2426, -23.0703);
  return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

@fragment
fn fs(i : VSOut) -> @location(0) vec4<f32> {
  if (F.clipEnable > 0.5) {
    if (dot(i.world, F.clipPlane.xyz) > F.clipPlane.w) { discard; }
  }
  let diff = abs(dot(normalize(i.normal), F.lightDir));
  let base = mix(O.color, inferno(i.scalar), O.colormap);
  let lit = base * (F.ambient + (1.0 - F.ambient) * diff);
  return vec4<f32>(lit, 1.0);
}
`;

// Thick wireframe: WebGPU has no native lineWidth, so each segment is expanded
// to a screen-space quad (6 verts per instance, aspect-correct).
export const LINE_WGSL = /* wgsl */ `
struct Frame { mvp : mat4x4<f32>, viewport : vec2<f32>, halfPx : f32, _pad : f32 };
struct Obj { color : vec4<f32> };
@group(0) @binding(0) var<uniform> F : Frame;
@group(1) @binding(0) var<uniform> O : Obj;
const ENDS = array<u32,6>(0u,0u,1u,0u,1u,1u);
const SIDE = array<f32,6>(-1.0,1.0,1.0,-1.0,1.0,-1.0);
@vertex
fn vs(@builtin(vertex_index) vi : u32, @location(0) A : vec3<f32>, @location(1) B : vec3<f32>) -> @builtin(position) vec4<f32> {
  let ca = F.mvp * vec4<f32>(A, 1.0);
  let cb = F.mvp * vec4<f32>(B, 1.0);
  let na = ca.xy / ca.w; let nb = cb.xy / cb.w;
  let sa = na * F.viewport * 0.5; let sb = nb * F.viewport * 0.5;
  var sdir = sb - sa; let ln = length(sdir); sdir = select(vec2<f32>(1.0, 0.0), sdir / ln, ln > 1e-6);
  let sperp = vec2<f32>(-sdir.y, sdir.x);
  let end = ENDS[vi]; let side = SIDE[vi];
  let clip = select(ca, cb, end == 1u);
  let ndc = select(na, nb, end == 1u);
  let off = (sperp * side * F.halfPx) / (F.viewport * 0.5);
  return vec4<f32>((ndc + off) * clip.w, clip.z, clip.w);
}
@fragment fn fs() -> @location(0) vec4<f32> { return O.color; }
`;
