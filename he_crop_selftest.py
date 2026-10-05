#!/usr/bin/env python3
# ==============================================================================
# he_crop_selftest.py — can the H&E panel's window mapping be WRONG and still
#                       pass?  Three checks designed so that it cannot.
#
# WHY THIS EXISTS
#   he_image.py claims that a window in the app's micron frame lands on the same
#   tissue in the microscope H&E, because pxl_*_in_fullres index that image and
#   so tif_px = micron / mpp.  "The crop looks like kidney" is not evidence for
#   that -- it passes for any offset, any scale, and any array.  Each check here
#   has a way to come out wrong.
#
#   A  INDEX vs SOURCE.  The JPEG tile pyramid must agree with a direct read of
#      the original TIF.  Correlation alone is not enough: on a near-blank field
#      +-1 of JPEG noise drags r down to 0.72 while the images are identical, so
#      the test reports the window's CONTRAST too and asks for the best
#      alignment to be (0, 0).  A tile- or level-indexing error shifts the image
#      and moves that peak.
#   B  OFFSET SWEEP against in_tissue.  Across many windows, how much tissue the
#      H&E shows must track how many bins Space Ranger called in_tissue -- and
#      the agreement must PEAK AT ZERO SHIFT.  A constant offset peaks somewhere
#      else; a wrong scale gives no peak at all.  in_tissue is derived from the
#      image, which is the point: this tests GEOMETRY, not biology.  (Same
#      pattern as he_cytassist_overlay.py.)
#   C  LANDMARK + CONTROL.  A window centred on the densest patch of 8 um bins
#      RCTD calls Podo -- chosen from RNA alone, with no reference to the image
#      -- must show a glomerulus, and a window shifted off it must not.  Without
#      the control, "cortex everywhere" would pass by default.
#
# INPUTS   HiRes_Images/<s>_fullres.TIF, HiRes_Index/<s>/ (optional),
#          Processed/<s>/outs/binned_outputs/square_008um/spatial/
#            tissue_positions.parquet (the FULL lattice, for in_tissue),
#          L0_CellType/<s>_celltype_008um.csv (via spatial_rgb_data.labels)
# OUTPUTS  DataOverview/he_crop/he_crop_selftest.json
#          DataOverview/he_crop/he_crop_selftest.png
# Usage    ../spatial-venv/bin/python SpatialRGB/he_crop_selftest.py [KTx_18 ...]
# ==============================================================================

import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Where the data lives (Processed/, L0_CellType/, HiRes_*).  $SPATIAL_BASE if set,
# else the folder that contains SpatialRGB/ -- which is the project root here.
BASE = os.environ.get(
    "SPATIAL_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import he_image as HE                                           # noqa: E402
import spatial_rgb_data as D                                    # noqa: E402

OUT = f"{BASE}/DataOverview/he_crop"
os.makedirs(OUT, exist_ok=True)

WIN_UM = 200.0            # window side for the sweep
N_WIN = 200               # windows sampled
SHIFT_UM = 40.0           # sweep half-range
STEP_UM = 10.0
# THE TISSUE CRITERION IS MEASURED, NOT GUESSED.  The first attempt ("a pixel
# brighter than 215 in every channel is slide") marked 100.0% of every crop as
# tissue and made the whole sweep NaN -- because off-tissue here is NOT blank
# white slide: in KTx_18 it sits at luminance 162 with saturation 80, against 95
# for tissue.  So the cut is on luminance, between those two modes, and
# calibrate() re-measures the separation it achieves on each array and stores it
# in the JSON.  A criterion that stopped separating would show up there instead
# of quietly flattening the sweep.
DARK = 145


def tissue_frac(img):
    """Fraction of the crop that is stained tissue rather than background."""
    if img.size == 0:
        return 0.0
    return float((img.astype(np.float32).mean(-1) < DARK).mean())


def calibrate(him, sample, rng, n=12, win=200.0):
    """Measured separation of the DARK cut on windows that are all-in / all-out."""
    bx, by, it = lattice(sample)
    got = {True: [], False: []}
    tries = 0
    while (len(got[True]) < n or len(got[False]) < n) and tries < 6000:
        tries += 1
        x = rng.uniform(bx.min(), bx.max() - win)
        y = rng.uniform(by.min(), by.max() - win)
        m = ((bx >= x) & (bx < x + win) & (by >= y) & (by < y + win))
        if m.sum() < 50:
            continue
        f = float(it[m].mean())
        want = f > 0.98 if f > 0.98 else (False if f < 0.02 else None)
        if want is None or len(got[want]) >= n:
            continue
        img, info = him.crop(x, y, win, 128)
        if info["coverage"] > 0.999:
            got[want].append(img)
    out = {}
    for k, lab in ((True, "in_tissue"), (False, "out_of_tissue")):
        if got[k]:
            a = np.concatenate([i.reshape(-1, 3) for i in got[k]])
            out[lab] = dict(n_windows=len(got[k]),
                            mean_luminance=round(float(a.mean(1).mean()), 1),
                            frac_below_cut=round(float(
                                (a.astype(np.float32).mean(1) < DARK).mean()), 4))
    if "in_tissue" in out and "out_of_tissue" in out:
        out["separation"] = round(out["in_tissue"]["frac_below_cut"]
                                  - out["out_of_tissue"]["frac_below_cut"], 4)
    out["cut"] = DARK
    return out


def lattice(sample):
    """The FULL 8 um lattice with in_tissue, in microns — not just kept bins."""
    sd = f"{BASE}/Processed/{sample}/outs/binned_outputs/square_008um/spatial"
    mpp = json.load(open(f"{sd}/scalefactors_json.json"))["microns_per_pixel"]
    t = pq.read_table(f"{sd}/tissue_positions.parquet",
                      columns=["in_tissue", "pxl_row_in_fullres",
                               "pxl_col_in_fullres"]).to_pandas()
    return (t["pxl_col_in_fullres"].to_numpy(float) * mpp,
            t["pxl_row_in_fullres"].to_numpy(float) * mpp,
            t["in_tissue"].to_numpy().astype(bool))


def best_shift(a, b, rad=6):
    """Integer (dx, dy) aligning b onto a, and the correlation there."""
    af = a.astype(float).mean(-1); bf = b.astype(float).mean(-1)
    af = af - af.mean(); bf = bf - bf.mean()
    best = (-2.0, 0, 0)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            A = af[max(0, dy):af.shape[0] + min(0, dy),
                   max(0, dx):af.shape[1] + min(0, dx)]
            B = bf[max(0, -dy):bf.shape[0] + min(0, -dy),
                   max(0, -dx):bf.shape[1] + min(0, -dx)]
            if A.size < 100 or B.size < 100:
                continue
            A = A[:B.shape[0], :B.shape[1]]; B = B[:A.shape[0], :A.shape[1]]
            d = np.sqrt((A * A).sum() * (B * B).sum())
            if d <= 0:
                continue
            c = float((A * B).sum() / d)
            if c > best[0]:
                best = (c, dx, dy)
    return best


def check_a(him, rng):
    """Index vs source, at several window sizes, on CONTRASTY windows."""
    if not him.indexed:
        return dict(skipped="no index built for this array")
    rows = []
    for side in (60.0, 120.0, 320.0, 800.0, 2000.0):
        # Pick a window with real structure -- a blank field cannot test
        # alignment (on one, +-1 of JPEG noise puts r at 0.72 while the images
        # are identical).  SEARCH with the fast path and pay for the exact read
        # only once, or a 2000 um window costs minutes of full-width decoding.
        x = y = 0.0
        for _ in range(60):
            x = rng.uniform(0, max(him.width * him.mpp - side, 1))
            y = rng.uniform(0, max(him.height * him.mpp - side, 1))
            probe, ip = him.crop(x, y, side, 128)
            if ip["coverage"] > 0.999 and probe.astype(float).mean(-1).std() > 12:
                break
        a, ia = him.crop(x, y, side, exact=True)
        b, ib = him.crop(x, y, side)
        c, dx, dy = best_shift(a, b)
        rows.append(dict(side_um=side, contrast_sd=round(float(
            a.astype(float).mean(-1).std()), 2), corr_at_best=round(c, 4),
            shift_px=[dx, dy], mean_abs_diff=round(float(np.abs(
                a.astype(int) - b.astype(int)).mean()), 2),
            exact_s=ia["seconds"], index_s=ib["seconds"], level=ib["level"]))
    ok = all(r["shift_px"] == [0, 0] and r["corr_at_best"] > 0.99 for r in rows)
    return dict(windows=rows, all_aligned_and_faithful=bool(ok))


def check_b(him, sample, rng):
    """Offset sweep: H&E tissue fraction vs in_tissue bins must peak at zero."""
    bx, by, it = lattice(sample)
    shifts = np.arange(-SHIFT_UM, SHIFT_UM + 1e-9, STEP_UM)
    pad = SHIFT_UM
    big = WIN_UM + 2 * pad
    out_px = 128
    scale = out_px / big                      # output px per micron of the big crop
    keep_px = int(round(WIN_UM * scale))
    frac_bins, crops = [], []
    tries = 0
    while len(crops) < N_WIN and tries < N_WIN * 20:
        tries += 1
        x = rng.uniform(0, max(him.width * him.mpp - big, 1))
        y = rng.uniform(0, max(him.height * him.mpp - big, 1))
        m = ((bx >= x + pad) & (bx < x + pad + WIN_UM) &
             (by >= y + pad) & (by < y + pad + WIN_UM))
        if m.sum() < 50:
            continue
        f = float(it[m].mean())
        if 0.02 < f < 0.98:                   # only windows that can discriminate
            img, info = him.crop(x, y, big, out_px)
            if info["coverage"] > 0.999:
                crops.append(img)
                frac_bins.append(f)
    frac_bins = np.array(frac_bins)
    grid = np.zeros((len(shifts), len(shifts)))
    for iy, dy in enumerate(shifts):
        for ix, dx in enumerate(shifts):
            o_x = int(round((pad + dx) * scale)); o_y = int(round((pad + dy) * scale))
            v = [tissue_frac(c[o_y:o_y + keep_px, o_x:o_x + keep_px]) for c in crops]
            grid[iy, ix] = np.corrcoef(frac_bins, v)[0, 1] if len(v) > 3 else np.nan
    iy, ix = np.unravel_index(np.nanargmax(grid), grid.shape)
    zi = int(np.argmin(np.abs(shifts)))
    return dict(n_windows=len(crops), window_um=WIN_UM,
                shifts_um=[float(s) for s in shifts],
                grid=[[round(float(v), 4) for v in row] for row in grid],
                r_at_zero=round(float(grid[zi, zi]), 4),
                r_at_peak=round(float(grid[iy, ix]), 4),
                peak_shift_um=[float(shifts[ix]), float(shifts[iy])],
                peak_is_zero=bool(shifts[ix] == 0 and shifts[iy] == 0)), grid, shifts


def check_c(him, sample, rng):
    """Landmark chosen from RNA only, plus a shifted control."""
    a8 = D.load(sample, "square_008um")
    codes, _ = a8.labels()
    pod = codes == D.CELLTYPE_ORDER.index("Podo")
    if pod.sum() < 20:
        return dict(skipped="too few Podo bins"), None, None
    Hh, _, _ = np.histogram2d(a8.bx[pod], a8.by[pod], bins=60)
    xe = np.linspace(a8.bx.min(), a8.bx.max(), 61)
    ye = np.linspace(a8.by.min(), a8.by.max(), 61)
    i, j = np.unravel_index(Hh.argmax(), Hh.shape)
    side = 200.0
    gx, gy = xe[i] - side / 2, ye[j] - side / 2
    glom, gi = him.crop(gx, gy, side)
    ctrl, ci = him.crop(gx + 500.0, gy, side)
    bx, by, it = lattice(sample)
    base = []
    for _ in range(60):
        x = rng.uniform(0, max(him.width * him.mpp - side, 1))
        y = rng.uniform(0, max(him.height * him.mpp - side, 1))
        m = ((bx >= x) & (bx < x + side) & (by >= y) & (by < y + side))
        if m.sum() > 50 and it[m].mean() > 0.9:
            img, info = him.crop(x, y, side)
            if info["coverage"] > 0.999:
                base.append(tissue_frac(img))
    return (dict(window_um=side, podo_bins_in_cell=int(Hh.max()),
                 glom_origin_um=[round(gx, 1), round(gy, 1)],
                 glom_tissue_frac=round(tissue_frac(glom), 4),
                 control_tissue_frac=round(tissue_frac(ctrl), 4),
                 random_tissue_frac_median=round(float(np.median(base)), 4)
                 if base else None,
                 glom_coverage=gi["coverage"], control_coverage=ci["coverage"]),
            glom, ctrl)


if __name__ == "__main__":
    samples = [a for a in sys.argv[1:] if a.startswith("KTx")] or ["KTx_18", "KTx_17"]
    res, figs = {}, {}
    for s in samples:
        print(f"=== {s} ===", flush=True)
        rng = np.random.default_rng(0)
        him = HE.HEImage(s)
        cal = calibrate(him, s, rng)
        print(f"  calibration: tissue {cal.get('in_tissue', {}).get('frac_below_cut')} "
              f"vs background {cal.get('out_of_tissue', {}).get('frac_below_cut')} "
              f"-> separation {cal.get('separation')}", flush=True)
        a = check_a(him, rng)
        print(f"  A index-vs-source: {a.get('all_aligned_and_faithful', a.get('skipped'))}",
              flush=True)
        b, grid, shifts = check_b(him, s, rng)
        print(f"  B offset sweep: n={b['n_windows']} r0={b['r_at_zero']} "
              f"peak={b['r_at_peak']} at {b['peak_shift_um']} µm  "
              f"zero={b['peak_is_zero']}", flush=True)
        c, glom, ctrl = check_c(him, s, rng)
        print(f"  C landmark: {c}", flush=True)
        res[s] = dict(calibration=cal, check_a=a, check_b=b, check_c=c,
                      indexed=him.indexed,
                      size_px=[him.width, him.height], mpp=him.mpp)
        figs[s] = (grid, shifts, glom, ctrl)

    json.dump(res, open(f"{OUT}/he_crop_selftest.json", "w"), indent=1)

    # ---- the figure: a reader checks the claim by eye, not only by r ---------
    n = len(samples)
    fig, ax = plt.subplots(n, 3, figsize=(13.5, 4.6 * n), squeeze=False,
                           facecolor="white")
    for k, s in enumerate(samples):
        grid, shifts, glom, ctrl = figs[s]
        b = res[s]["check_b"]
        im = ax[k][0].imshow(grid, origin="lower", cmap="viridis",
                             extent=[shifts[0], shifts[-1], shifts[0], shifts[-1]])
        ax[k][0].plot(0, 0, "w+", ms=14, mew=2)
        ax[k][0].plot(b["peak_shift_um"][0], b["peak_shift_um"][1], "r.", ms=11)
        fig.colorbar(im, ax=ax[k][0], fraction=.046).set_label("r (H&E vs in_tissue)")
        ax[k][0].set_title(f"{s} — offset sweep, n={b['n_windows']}\n"
                           f"peak {b['r_at_peak']} at {b['peak_shift_um']} µm "
                           f"(white + = zero)", fontsize=9, loc="left")
        ax[k][0].set_xlabel("dx µm"); ax[k][0].set_ylabel("dy µm")
        c = res[s]["check_c"]
        for t, img, lab in ((1, glom, "Podo-densest window (picked from RNA)"),
                            (2, ctrl, "same window + 500 µm (control)")):
            if img is None:
                ax[k][t].axis("off"); continue
            ax[k][t].imshow(img, interpolation="nearest")
            f = (c["glom_tissue_frac"] if t == 1 else c["control_tissue_frac"])
            ax[k][t].set_title(f"{s} — {lab}\ntissue fraction {f:.3f} "
                               f"(array median {c['random_tissue_frac_median']})",
                               fontsize=9, loc="left")
            ax[k][t].set_xticks([]); ax[k][t].set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{OUT}/he_crop_selftest.png", dpi=95)
    print(f"\nwrote {OUT}/he_crop_selftest.json and .png")
