#!/usr/bin/env python3
"""qidk_adb.py - adb-driven checks that need nothing from the app except what is on screen.

  python3 tools/qidk_adb.py geometry --serial 8f27557f [--tol 45]
      Manual mode: swipes three short strokes at known screen positions and checks that the red mask overlay appears
      where the finger was (this is what "selected the boat on the other side of the screen" would fail).
  python3 tools/qidk_adb.py lasso --serial 8f27557f --cx 0.50 --cy 0.60 --rx 0.30 --ry 0.20
      Smart mode: draws a lasso (ellipse; centre/radii are fractions of the canvas, or --bmp CX CY RX RY in bitmap pixels,
      converted through the 'QIDK.Touch geom' line the canvas logs) with `input motionevent`,
      waits, then reports the overlay: bounding box, pixels, share inside the lasso and fill ratio (a fill ratio near
      1.0 means the mask is just your circle, i.e. nothing was snapped).
Requires: adb, python3, numpy, opencv-python. The canvas must be the view with resource-id *inpaintCanvasView*.
"""
import argparse, math, re, subprocess, sys, time
import numpy as np
import cv2


def adb(serial, *args, binary=False):
    r = subprocess.run(["adb", "-s", serial, *args], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)} failed: {r.stderr.decode(errors='replace')}")
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def screenshot(serial):
    png = adb(serial, "exec-out", "screencap", "-p", binary=True)
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def parse_node(xml, id_suffix):
    """(resource-id, (left, top, right, bottom)) of the first node whose resource-id ends with id_suffix, else None."""
    for m in re.finditer(r'<node [^>]*?resource-id="([^"]*)"[^>]*?bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', xml):
        if m.group(1).endswith(id_suffix):
            return m.group(1), tuple(int(m.group(i)) for i in range(2, 6))
    return None


def parse_bounds(xml, id_suffix):
    n = parse_node(xml, id_suffix)
    return n[1] if n else None


GEOM_RE = re.compile(r"s=\(([-\d.eE]+),([-\d.eE]+)\) t=\(([-\d.eE]+),([-\d.eE]+)\)")


def parse_geom(log_text):
    """(sx, sy, tx, ty) of the LAST 'QIDK.Touch ... geom' line (ViewportMapper.toString), else None."""
    found = None
    for line in log_text.splitlines():
        if "geom" in line:
            m = GEOM_RE.search(line)
            if m:
                found = tuple(float(m.group(i)) for i in range(1, 5))
    return found


def ui_node(serial, id_suffix):
    adb(serial, "shell", "uiautomator", "dump", "/sdcard/qidk_ui.xml")
    n = parse_node(adb(serial, "shell", "cat", "/sdcard/qidk_ui.xml"), id_suffix)
    if n is None:
        raise RuntimeError(f"no view with resource-id ending in '{id_suffix}' on screen")
    return n


def ui_bounds(serial, id_suffix):
    return ui_node(serial, id_suffix)[1]


def overlay_mask(before, after, bounds, thresh=40):
    """Binary mask (full-screen size) of pixels inside `bounds` that changed between two screenshots."""
    l, t, r, b = bounds
    d = cv2.absdiff(after, before).max(axis=2)
    m = np.zeros(d.shape, np.uint8)
    m[t:b, l:r] = (d[t:b, l:r] > thresh).astype(np.uint8)
    return m


def bbox(mask):
    if mask.sum() == 0:
        return None
    x, y, w, h = cv2.boundingRect(mask)
    return (x, y, x + w, y + h)


def ellipse_points(cx, cy, rx, ry, n=36, sweep=2.1 * math.pi, wobble=0.05):
    return [(cx + rx * (1 + wobble * math.sin(3 * a + 1.0)) * math.cos(a),
             cy + ry * (1 + wobble * math.sin(3 * a + 1.0)) * math.sin(a))
            for a in (sweep * i / (n - 1) for i in range(n))]


def tap(serial, bounds):
    adb(serial, "shell", "input", "tap", str((bounds[0] + bounds[2]) // 2), str((bounds[1] + bounds[3]) // 2))


def play_polyline(serial, pts):
    """DOWN, MOVE..., UP through the real input pipeline (needs Android 12+)."""
    cmds = [f"input motionevent DOWN {pts[0][0]:.0f} {pts[0][1]:.0f}"]
    cmds += [f"input motionevent MOVE {x:.0f} {y:.0f}" for x, y in pts[1:-1]]
    cmds.append(f"input motionevent UP {pts[-1][0]:.0f} {pts[-1][1]:.0f}")
    adb(serial, "shell", "; ".join(cmds))


def cmd_geometry(a):
    l, t, r, b = ui_bounds(a.serial, "inpaintCanvasView")
    clear = None
    try:
        clear = ui_bounds(a.serial, "btnClear")
    except RuntimeError:
        pass
    W, H = r - l, b - t
    print(f"canvas bounds = {(l, t, r, b)} ({W}x{H}px)")
    cases = [("upper-left diagonal", (0.20, 0.20), (0.35, 0.40)),
             ("centre vertical", (0.50, 0.45), (0.50, 0.85)),
             ("lower-right horizontal", (0.65, 0.80), (0.92, 0.80))]
    bad = 0
    for name, p0, p1 in cases:
        if clear:
            tap(a.serial, clear); time.sleep(0.4)
        x0, y0, x1, y1 = l + p0[0] * W, t + p0[1] * H, l + p1[0] * W, t + p1[1] * H
        before = screenshot(a.serial)
        adb(a.serial, "shell", "input", "swipe", f"{x0:.0f}", f"{y0:.0f}", f"{x1:.0f}", f"{y1:.0f}", "450")
        time.sleep(0.8)
        after = screenshot(a.serial)
        bb = bbox(overlay_mask(before, after, (l, t, r, b)))
        if bb is None:
            print(f"FAIL {name}: no overlay appeared"); bad += 1; continue
        exp = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        err = max(abs(bb[0] - (exp[0])), abs(bb[1] - exp[1]), abs(bb[2] - exp[2]), abs(bb[3] - exp[3]))
        # the brush adds half its width on every side, so compare the CENTRE of the overlay box with the centre of the finger path
        cerr = math.hypot((bb[0] + bb[2]) / 2 - (exp[0] + exp[2]) / 2, (bb[1] + bb[3]) / 2 - (exp[1] + exp[3]) / 2)
        ok = cerr <= a.tol
        bad += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {name}: finger box {tuple(int(v) for v in exp)}  overlay box {bb}  centre error {cerr:.0f}px (tol {a.tol})")
    if clear:
        tap(a.serial, clear)
    return 1 if bad else 0


def cmd_lasso(a):
    rid, (l, t, r, b) = ui_node(a.serial, "inpaintCanvasView")
    W, H = r - l, b - t
    if a.bmp:
        # bitmap pixels -> screen pixels through the geometry the canvas logs (ViewportMapper: view = bitmap * s + t)
        pkg = rid.split(":")[0]
        adb(a.serial, "shell", "am", "broadcast", "-a", "qidk.debug.MODE", "-p", pkg, "--es", "mode", "smart")
        time.sleep(1.0)
        g = parse_geom(adb(a.serial, "logcat", "-d", "-s", "QIDK.Touch:I"))
        if g is None:
            print("FAIL no 'QIDK.Touch geom ...' line in logcat: log mapper.toString() in onSizeChanged and in the MODE hook"); return 1
        sx, sy, tx, ty = g
        bx, by, brx, bry = a.bmp
        cx, cy, rx, ry = l + tx + bx * sx, t + ty + by * sy, brx * sx, bry * sy
        print(f"geometry from log: s=({sx:.4f},{sy:.4f}) t=({tx:.1f},{ty:.1f})")
    else:
        cx, cy, rx, ry = l + a.cx * W, t + a.cy * H, a.rx * W, a.ry * H
    pts = ellipse_points(cx, cy, rx, ry, a.points)
    before = screenshot(a.serial)
    adb(a.serial, "logcat", "-c")
    play_polyline(a.serial, pts)
    time.sleep(a.wait)
    after = screenshot(a.serial)
    m = overlay_mask(before, after, (l, t, r, b), thresh=60)
    bb = bbox(m)
    loop = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(loop, [np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)], 1)
    loop_grown = cv2.dilate(loop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.12 * max(rx, ry)) + 1,) * 2))
    px = int(m.sum())
    print(f"canvas {(l, t, r, b)}  lasso centre ({cx:.0f},{cy:.0f}) radii ({rx:.0f},{ry:.0f})")
    if bb is None or px == 0:
        print("FAIL no overlay appeared"); return 1
    inside = float((m & loop_grown).sum()) / px
    fill = px / max(1.0, float(loop.sum()))
    print(f"overlay box {bb}  pixels {px}  inside-lasso(+12%) {inside:.2f}  fill-ratio {fill:.2f}")
    print(adb(a.serial, "logcat", "-d", "-s", "QIDK.Smart:I").strip().splitlines()[-3:])
    ok = inside >= 0.85
    print("ok   overlay stays on the circled object" if ok else "FAIL overlay left the lasso (touch mapping?)")
    print("note fill-ratio ~1.0 = overlay is just the circle (not snapped); < 0.85 = follows the object's contour")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometry"); g.add_argument("--serial", default="8f27557f"); g.add_argument("--tol", type=float, default=45.0)
    s = sub.add_parser("lasso"); s.add_argument("--serial", default="8f27557f")
    for k, v in (("cx", 0.5), ("cy", 0.6), ("rx", 0.3), ("ry", 0.2)):
        s.add_argument(f"--{k}", type=float, default=v)
    s.add_argument("--bmp", type=float, nargs=4, metavar=("CX", "CY", "RX", "RY"), help="centre and radii in BITMAP pixels (512x512); overrides --cx/--cy/--rx/--ry")
    s.add_argument("--points", type=int, default=36); s.add_argument("--wait", type=float, default=4.0)
    a = p.parse_args()
    sys.exit(cmd_geometry(a) if a.cmd == "geometry" else cmd_lasso(a))


if __name__ == "__main__":
    main()
