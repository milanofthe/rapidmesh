"""Bundled interactive-viewer launcher for ``Mesh.show()``.

Serves the viewer that ships in this package (``_viewer_dist/`` -- the SvelteKit
``MeshViewer`` build, minus the showcase sample meshes) on a transient localhost
server, points a window at its ``/embed`` route with the given mesh, and blocks
until the window is closed. A native window via ``pywebview`` if installed, else
a headed Playwright (Chromium) window -- no browser tab to manage, no Tauri.

The viewer ships in the wheel; a window backend does not (it is heavy and
GUI-only), so ``inspect`` raises a clear install hint if neither is present.
"""
from __future__ import annotations

import contextlib
import functools
import http.server
import socket
import threading
import urllib.parse
from pathlib import Path

_DIST = Path(__file__).resolve().parent / "_viewer_dist"
_EMBED_FILE = "embed.html"


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Static handler over ``_viewer_dist`` with two virtual routes: ``/embed``
    (the prerendered viewer page) and ``/__mesh__.json`` (the mesh under view)."""

    mesh_path: str = ""

    def translate_path(self, path: str) -> str:  # noqa: D102
        clean = urllib.parse.urlparse(path).path
        if clean == "/__mesh__.json" and self.mesh_path:
            return self.mesh_path
        if clean in ("/embed", "/embed/"):
            return str(_DIST / _EMBED_FILE)
        return super().translate_path(path)

    def log_message(self, *args) -> None:  # silence per-request logging
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


@contextlib.contextmanager
def _serve(mesh_file: Path):
    if not (_DIST / _EMBED_FILE).is_file():
        raise RuntimeError(
            f"bundled viewer missing at {_DIST}. In a source checkout, build and "
            f"sync it once:  cd site && npm run build && python report/sync_viewer_dist.py"
        )
    _Handler.mesh_path = str(mesh_file)
    handler = functools.partial(_Handler, directory=str(_DIST))
    with socket.socket() as s:  # grab a free port
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


def _embed_url(base, *, azim, elev, dist, wireframe, tets, edges, defects, clip, clip_axis):
    q = {
        "mesh": "/__mesh__.json",
        "transparent": "0",
        "controls": "1",
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
    return f"{base}/embed?" + urllib.parse.urlencode(q)


def inspect(
    mesh_json,
    *,
    azim: float = 30,
    elev: float = 20,
    dist: float | None = None,
    wireframe: bool = True,
    tets: bool = True,
    edges: bool = True,
    defects: bool = False,
    clip: float | None = 0.6,
    clip_axis: int = 1,
    title: str = "rapidmesh viewer",
) -> None:
    """Open ``mesh_json`` (a path to a viewer-schema JSON) in an interactive
    window and block until it is closed. Orbit = drag, pan = right-drag, zoom =
    wheel; the viewer's own toolbar (Fit / Tets / Wire / Edge / Clip / legend /
    export) is available."""
    mesh_file = Path(mesh_json).resolve()
    with _serve(mesh_file) as base:
        url = _embed_url(
            base, azim=azim, elev=elev, dist=dist, wireframe=wireframe, tets=tets,
            edges=edges, defects=defects, clip=clip, clip_axis=clip_axis,
        )
        # Preferred: a native pywebview window.
        try:
            import webview  # type: ignore

            webview.create_window(title, url, width=1280, height=900)
            webview.start()
            return
        except ImportError:
            pass
        # Fallback: a headed Playwright (Chromium) window.
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise RuntimeError(
                "mesh.show() needs a window backend. Install one:\n"
                "    pip install pywebview\n"
                "  or\n"
                "    pip install playwright && playwright install chromium"
            ) from e
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=False)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(url, wait_until="load")
            print(f"[rapidmesh] viewer open: {title} -- close the window to return.")
            with contextlib.suppress(Exception):
                page.wait_for_event("close", timeout=0)
            with contextlib.suppress(Exception):
                browser.close()
