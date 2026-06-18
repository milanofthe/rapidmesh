"""
Python launcher for rapidmesh's EXISTING 3D mesh viewer.

This does NOT reimplement anything. It serves the already-built SvelteKit
showcase (site/build) and points a browser at the `/embed` route, which mounts
the verbatim `MeshViewer.svelte` component. Two entry points:

    inspect(mesh, **camera)   -> opens an interactive window (orbit/zoom/pan)
                                  and blocks until you close it.
    render(mesh, out_png, ..) -> headless screenshot with a transparent
                                  background (PNG with a real alpha channel).

The only viewer-code change this relies on is the `transparentBackground`
prop added to MeshViewer.svelte; everything else (scene, materials, lighting,
region colouring, camera) is the unchanged component.

Build the viewer once (re-run if site/src changes):
    cd site && npm run build

Python deps:
    pip install playwright pillow
    playwright install chromium
    pip install pywebview          # optional, for a cleaner native inspect() window
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import os
import socket
import threading
import urllib.parse
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent                      # rapidmesh/
BUILD_DIR = _REPO / "site" / "build"             # vite build output
COMPARE_DIR = BUILD_DIR / "meshes" / "compare"   # bundled sample meshes
# SvelteKit's client router matches on location.pathname, so the page must be
# served at "/embed" (not "/embed.html"); the handler maps it to the
# prerendered embed.html file.
EMBED_ROUTE = "embed"
EMBED_FILE = "embed.html"                         # adapter-static output


# ── Mesh resolution ────────────────────────────────────────────────────────
def _resolve_mesh(mesh: str | os.PathLike, engine: str) -> Path:
    """Resolve `mesh` to an absolute .json file.

    Accepts either a path to a mesh JSON file, or a bare geometry name such as
    "box" / "bunny" which maps to the bundled
    site/build/meshes/compare/<name>.<engine>.json.
    """
    p = Path(mesh)
    if p.is_file():
        return p.resolve()
    cand = COMPARE_DIR / f"{mesh}.{engine}.json"
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(
        f"Could not resolve mesh {mesh!r}. Pass a path to a mesh JSON file, "
        f"or a bundled name (e.g. 'box', 'bunny'); expected {cand}"
    )


# ── Local static server (serves the built viewer + the chosen mesh) ─────────
class _Handler(http.server.SimpleHTTPRequestHandler):
    mesh_path: str = ""  # absolute path served at /__mesh__.json

    def translate_path(self, path: str) -> str:  # noqa: D102
        clean = urllib.parse.urlparse(path).path
        if clean == "/__mesh__.json" and self.mesh_path:
            return self.mesh_path
        if clean in ("/embed", "/embed/"):
            return str(BUILD_DIR / EMBED_FILE)
        return super().translate_path(path)

    def log_message(self, *args) -> None:  # silence per-request logging
        pass

    def end_headers(self) -> None:
        # Always-fresh, so a re-render after a rebuild never serves stale JS.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


@contextlib.contextmanager
def _serve(mesh_file: Path):
    if not (BUILD_DIR / EMBED_FILE).is_file():
        raise FileNotFoundError(
            f"Built viewer not found at {BUILD_DIR / EMBED_FILE}. "
            f"Build it first:  cd site && npm run build"
        )
    handler = functools.partial(_Handler, directory=str(BUILD_DIR))
    _Handler.mesh_path = str(mesh_file)
    # Pick a free port.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


# ── URL builder ─────────────────────────────────────────────────────────────
def _embed_url(
    base: str,
    *,
    transparent: bool,
    controls: bool,
    azim: float,
    elev: float,
    dist: float | None,
    wireframe: bool,
    tets: bool,
    edges: bool,
    defects: bool,
    clip: float | None,
    clip_axis: int,
    width: int | None,
    height: int | None,
) -> str:
    q: dict[str, str] = {
        "mesh": "/__mesh__.json",
        "transparent": "1" if transparent else "0",
        "controls": "1" if controls else "0",
        "azim": f"{azim:g}",
        "elev": f"{elev:g}",
        "wire": "1" if wireframe else "0",
        "tets": "1" if tets else "0",
        "edges": "1" if edges else "0",
        "defects": "1" if defects else "0",
    }
    if dist is not None:
        q["dist"] = f"{dist:g}"
    if clip is not None:
        q["clip"] = "1"
        q["clipt"] = f"{clip:g}"
        q["clipaxis"] = str(int(clip_axis))
    else:
        q["clip"] = "0"
    if width:
        q["width"] = str(int(width))
    if height:
        q["height"] = str(int(height))
    return f"{base}/{EMBED_ROUTE}?" + urllib.parse.urlencode(q)


# ── (B) Headless render ─────────────────────────────────────────────────────
def render(
    mesh_json: str | os.PathLike,
    out_png: str | os.PathLike,
    *,
    transparent: bool = True,
    engine: str = "rapidmesh",
    azim: float = 30,
    elev: float = 20,
    dist: float | None = None,
    wireframe: bool = True,
    opacity: float = 1.0,        # accepted; see note below
    width: int = 1200,
    height: int = 900,
    region_colors=None,          # accepted; see note below
    clip: float | None = None,
    clip_axis: int = 1,
    tets: bool = True,
    edges: bool = True,
    defects: bool = False,
    controls: bool = False,
    timeout_ms: int = 30000,
) -> Path:
    """Headlessly render a mesh figure to a PNG.

    With transparent=True the PNG has a real alpha channel (omit_background),
    so it drops onto any report background cleanly. Deterministic: a fixed
    camera produces an identical PNG.

    Note: `opacity` and `region_colors` are accepted for API completeness but
    are NOT honoured — region colouring and surface opacity are owned by the
    unchanged MeshViewer (the 1:1-reuse hard requirement forbids editing them).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Playwright is required for render(). Install with:\n"
            "    pip install playwright\n"
            "    playwright install chromium"
        ) from e

    mesh_file = _resolve_mesh(mesh_json, engine)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with _serve(mesh_file) as base:
        url = _embed_url(
            base,
            transparent=transparent,
            controls=controls,
            azim=azim,
            elev=elev,
            dist=dist,
            wireframe=wireframe,
            tets=tets,
            edges=edges,
            defects=defects,
            clip=clip,
            clip_axis=clip_axis,
            width=width,
            height=height,
        )
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": int(width), "height": int(height)},
                # 2x pixel density: the canvas is devicePixelRatio-aware, so this
                # renders at double resolution (crisp PNGs) for the same framing.
                device_scale_factor=2,
            )
            page.goto(url, wait_until="load")
            page.wait_for_function(
                "() => document.body && document.body.hasAttribute('data-viewer-ready')",
                timeout=timeout_ms,
            )
            state = page.get_attribute("body", "data-viewer-ready")
            if state == "error":
                err = page.get_attribute("body", "data-viewer-error")
                browser.close()
                raise RuntimeError(f"Viewer failed to load mesh: {err}")
            # Let the final framed camera settle before snapping.
            page.wait_for_timeout(400)
            page.screenshot(
                path=str(out_png),
                omit_background=bool(transparent),
                clip={"x": 0, "y": 0, "width": int(width), "height": int(height)},
            )
            browser.close()
    return out_png


# ── (A) Interactive window ───────────────────────────────────────────────────
def inspect(
    mesh_json: str | os.PathLike,
    *,
    engine: str = "rapidmesh",
    azim: float = 30,
    elev: float = 20,
    dist: float | None = None,
    wireframe: bool = True,
    clip: float | None = None,
    clip_axis: int = 1,
    tets: bool = True,
    edges: bool = True,
    defects: bool = False,
    title: str = "rapidmesh viewer",
) -> None:
    """Open the mesh in an interactive window and block until it is closed.

    Uses pywebview (clean native window) if installed, else a headed
    Playwright (Chromium) window. Orbit = drag, pan = right-drag, zoom = wheel;
    the viewer's own toolbar (Fit / Tets / Wire / Edge / Clip) is available.
    """
    mesh_file = _resolve_mesh(mesh_json, engine)
    with _serve(mesh_file) as base:
        url = _embed_url(
            base,
            transparent=False,
            controls=True,
            azim=azim,
            elev=elev,
            dist=dist,
            wireframe=wireframe,
            tets=tets,
            edges=edges,
            defects=defects,
            clip=clip,
            clip_axis=clip_axis,
            width=None,
            height=None,
        )

        # Preferred: native pywebview window.
        try:
            import webview  # type: ignore

            webview.create_window(title, url, width=1280, height=900)
            webview.start()
            return
        except ImportError:
            pass

        # Fallback: headed Playwright (Chromium) window.
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "inspect() needs either pywebview or Playwright. Install one:\n"
                "    pip install pywebview\n"
                "  or\n"
                "    pip install playwright && playwright install chromium"
            ) from e

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=False)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(url, wait_until="load")
            print(f"[viewer] interactive window open: {title}\n"
                  f"         close the window to return.")
            try:
                # Block until the user closes the window/tab.
                page.wait_for_event("close", timeout=0)
            except Exception:
                pass
            with contextlib.suppress(Exception):
                browser.close()


# ── Demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    figures = _THIS.parent / "figures"
    demo_meshes = ["bunny", "gear", "bracket"]

    print("rapidmesh viewer.py demo")
    print(f"  build dir : {BUILD_DIR}")
    print(f"  figures   : {figures}\n")

    results = []
    for name in demo_meshes:
        out = figures / f"{name}.png"
        try:
            render(name, out, transparent=True, azim=35, elev=22,
                   width=1000, height=750)
            print(f"  rendered  {name:10s} -> {out.name}")
            results.append(out)
        except Exception as e:
            print(f"  FAILED    {name:10s}: {e}")

    # Verify transparency with Pillow.
    print("\nTransparency check (Pillow):")
    try:
        from PIL import Image

        for out in results:
            with Image.open(out) as im:
                rgba = im.convert("RGBA") if im.mode != "RGBA" else im
                alpha = rgba.getchannel("A")
                lo, hi = alpha.getextrema()
                ok = im.mode == "RGBA" and lo < 255
                print(f"  {out.name:14s} mode={im.mode:5s} "
                      f"alpha[min={lo} max={hi}] -> "
                      f"{'TRANSPARENT OK' if ok else 'NOT transparent'}")
    except ImportError:
        print("  Pillow not installed (pip install pillow) — skipped.")

    print("\nTo inspect a mesh interactively, run e.g.:")
    print("  python -c \"import viewer; viewer.inspect('bunny')\"")
    print("  python -c \"import viewer; viewer.inspect(r'C:/path/to/mesh.json')\"")

    if not results:
        sys.exit(1)
