#!/usr/bin/env python3
# ==============================================================================
# build_he_index.py — cut each array's microscope H&E into a JPEG tile pyramid
#                     so the app can crop any window in milliseconds.
#
# WHY THIS EXISTS (author, 2026-10-05: "the H&E images are very heavy we might
# need some indexation")
#   The source TIFs are 236-302 Mpx, LZW, single IFD, NO pyramid, and stripped
#   with RowsPerStrip = 1.  he_image.py can read a row range out of them exactly
#   (verified byte-identical to a full decode), but it costs ~0.7 ms per image
#   row -- 0.5 s for a 320 um window, 1.4 s for 800 um, and it grows with the
#   window.  That is too slow to sit behind a drag-to-zoom.
#
#   A tile pyramid turns every crop into "read 9-16 small JPEGs", independent of
#   how big the window is.
#
# WHY JPEG, AND WHAT THAT COSTS
#   Chosen by the author over a lossless store: ~10x less disk than raw tiles
#   (~25 GB) and no new Python dependency, where a lossless pyramid would need
#   tifffile + imagecodecs installed.  The pixels are therefore NOT bit-exact.
#   That is acceptable because this index exists to DRAW a 620 px panel, and
#   he_image.crop(..., exact=True) still reads the original TIF for anything
#   quantitative.  he_crop_selftest.py measures the agreement rather than
#   assuming it.
#
# RESUMABLE: an array whose manifest already matches its source file's size and
# mtime is skipped, so a killed run can simply be restarted.
#
# INPUT    HiRes_Images/<sample>_fullres.TIF (via the symlink)
# OUTPUT   HiRes_Index/<sample>/manifest.json
#          HiRes_Index/<sample>/L<level>/<tile_y>_<tile_x>.jpg
# Usage    cd ~/Documents/Spatial
#          nohup ../spatial-venv/bin/python -u SpatialRGB/build_he_index.py \
#                > SpatialRGB/build_he_index.log 2>&1 &
#          ../spatial-venv/bin/python SpatialRGB/build_he_index.py KTx_18   # one
# ==============================================================================

import json
import os
import shutil
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import he_image as H                                            # noqa: E402
import spatial_rgb_data as D                                    # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def du(path):
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            n += os.path.getsize(os.path.join(root, f))
    return n


def is_current(sample):
    """True when a finished index already matches the source file on disk."""
    m = f"{H.INDEX_DIR}/{sample}/manifest.json"
    src = H.he_path(sample)
    if not os.path.exists(m) or src is None:
        return False
    try:
        man = json.load(open(m))
    except Exception:                                           # noqa: BLE001
        return False
    st = os.stat(src)
    return (man.get("complete") is True
            and man.get("source_size") == st.st_size
            and man.get("source_mtime") == int(st.st_mtime)
            and man.get("tile") == H.TILE
            and man.get("levels") == H.N_LEVELS)


def build(sample, force=False):
    src = H.he_path(sample)
    if src is None:
        print(f"{sample}: NO microscope TIF — skipped", flush=True)
        return None
    if not force and is_current(sample):
        out = f"{H.INDEX_DIR}/{sample}"
        print(f"{sample}: already indexed ({du(out)/2**30:.2f} GiB) — skipped",
              flush=True)
        return None

    t0 = time.time()
    out = f"{H.INDEX_DIR}/{sample}"
    if os.path.exists(out):                   # a partial run leaves stale tiles
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    im = Image.open(src).convert("RGB")       # one full decode, ~7 s, ~0.9 GB
    W0, H0 = im.size
    t_dec = time.time() - t0
    sizes, n_tiles = [], 0

    for lv in range(H.N_LEVELS):
        if lv == 0:
            lim = im
        else:
            lim = im.resize((max(1, W0 >> lv), max(1, H0 >> lv)), Image.LANCZOS)
        w, h = lim.size
        sizes.append([w, h])
        d = f"{out}/L{lv}"
        os.makedirs(d, exist_ok=True)
        for ty in range((h + H.TILE - 1) // H.TILE):
            for tx in range((w + H.TILE - 1) // H.TILE):
                box = (tx * H.TILE, ty * H.TILE,
                       min((tx + 1) * H.TILE, w), min((ty + 1) * H.TILE, h))
                lim.crop(box).save(f"{d}/{ty}_{tx}.jpg", quality=H.JPEG_Q,
                                   subsampling=0)   # 4:4:4 — keep nuclear detail
                n_tiles += 1
        if lv:
            lim.close()
        print(f"  {sample} L{lv}: {w}x{h}, "
              f"{(h + H.TILE - 1)//H.TILE} x {(w + H.TILE - 1)//H.TILE} tiles",
              flush=True)
    im.close()

    st = os.stat(src)
    man = dict(sample=sample, width=W0, height=H0, tile=H.TILE,
               levels=H.N_LEVELS, level_size=sizes, quality=H.JPEG_Q,
               mpp=H.mpp_of(sample), n_tiles=n_tiles,
               source=os.path.basename(src), source_size=st.st_size,
               source_mtime=int(st.st_mtime),
               seconds=round(time.time() - t0, 1), complete=True)
    json.dump(man, open(f"{out}/manifest.json", "w"), indent=1)
    gib = du(out) / 2**30
    print(f"{sample}: {W0}x{H0}, {n_tiles:,} tiles, {gib:.2f} GiB, "
          f"{time.time()-t0:.1f}s (decode {t_dec:.1f}s)", flush=True)
    return gib


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    force = "--force" in sys.argv
    todo = args or D.samples()
    os.makedirs(H.INDEX_DIR, exist_ok=True)
    t0, total = time.time(), 0.0
    for i, s in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {s}", flush=True)
        g = build(s, force)
        if g:
            total += g
    print(f"\nALL DONE  {len(todo)} arrays, {total:.2f} GiB written, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
