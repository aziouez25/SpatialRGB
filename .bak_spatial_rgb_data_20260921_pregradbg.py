#!/usr/bin/env python3
# ==============================================================================
# spatial_rgb_data.py — the data layer behind the RGB marker browser.
#
# WHY THIS IS SEPARATE FROM THE APP
#   Everything here is testable without a browser, and the app is then only
#   layout and callbacks.  It is also the piece that has to be FAST: the whole
#   point of the tool is that any of ~18,000 genes can be put on any channel and
#   the picture appears at once.
#
# THE ONE ENGINEERING FACT THAT MAKES IT POSSIBLE
#   filtered_feature_bc_matrix.h5 is CSC over BARCODES: the column pointer runs
#   over bins, so pulling one GENE out of it means touching all ~9.3 M nonzeros.
#   Transposing once to GENE-major costs ~1.0 s and 75 MB, after which one gene
#   is a contiguous slice and a lookup is ~0.035 ms (measured, KTx_18 at 8 um).
#   Twenty genes across three channels is therefore under a millisecond, so no
#   disk cache is needed at all -- the transpose is done on first use of an array
#   and kept in memory.
#
# THE FRAME, and why it is not the h5's own row/col
#   Bins are placed the way the whole deck places them: x = pxl_col_in_fullres,
#   y = pxl_row_in_fullres, converted to microns, i.e. the MICROSCOPE IMAGE frame
#   with y downward.  array_row / array_col are the lattice's own indices and are
#   rotated relative to the image on these arrays, so using them would silently
#   turn every picture.  The origin is the minimum over the FULL lattice (all
#   702,244 bins, not just in_tissue), which is the capture square -- the same
#   normalisation the cohort figures use, so a window here and a window there
#   mean the same thing.
#
# NOTHING IS ASSERTED THAT IS NOT COMPUTED: every number the app prints comes
# from these functions, and channel_stats() is the same arithmetic as
# celltype_rgb_overlay.py (bins positive, and the co-location counts).
#
# INPUTS   Processed/<s>/outs/binned_outputs/square_{002,008,016}um/
#            filtered_feature_bc_matrix.h5, spatial/tissue_positions.parquet,
#            spatial/scalefactors_json.json
# Usage:   imported by app.py;  python spatial_rgb_data.py  runs a self-check
# ==============================================================================

import functools
import json
import os
import time

import h5py
import numpy as np
import pyarrow.parquet as pq

BASE = "/home/abb2013/Documents/Spatial"
RES = {"2 µm": "square_002um", "8 µm": "square_008um", "16 µm": "square_016um"}
BIN_UM = {"square_002um": 2.0, "square_008um": 8.0, "square_016um": 16.0}

# ============================ CELL-TYPE ANNOTATION ============================
# The July Layer 0 RCTD run (L0_celltype_8um.R): one label per IN-TISSUE 8 um bin
# for all 28 arrays, doublet mode against the GSE183276 atlas.
#   L0_CellType/<s>_celltype_008um.csv  ->  barcode, cell_type, spot_class
#
# THREE THINGS THAT MUST BE SAID WHEREVER THESE LABELS ARE DRAWN
#   1. IT IS 8 um ONLY.  There is no 2 um or 16 um version of this run, and a bin
#      of another size is not a subdivision of an 8 um bin that could inherit one
#      (16 um contains four, ambiguously).  So the layer is offered at 8 um and
#      refused elsewhere rather than silently mapped.
#   2. ABOUT A THIRD OF BINS HAVE NO ROW.  The run kept counts_MIN 10, so bins
#      under it were never offered to RCTD -- 117,279 of KTx_18's 169,844
#      in-tissue bins are present (69%).  "Not annotated" means NOT TESTED, not
#      "no cells": those bins have a median of 13-14 UMI.
#   3. UNKNOWN IS A CALL, NOT A GAP.  L0 maps spot_class reject and
#      doublet_uncertain to "Unknown" -- 36% of KTx_18's rows, 42% of KTx_17's.
#      A doublet_certain bin keeps first_type (the member fitting better alone).
L0_DIR = f"{BASE}/L0_CellType"
L0_RES_DIR = "square_008um"                 # the only bin size these labels exist at

# Legend order: down the nephron, then vessels and stroma, then immune, then the
# two non-calls.  Colours are IMPORTED from the deck so a type keeps its colour
# between this app and slides 21-23/27-28 -- see rctd_annotation_figure.py, which
# builds the same dict from marker_clusters_figure.COLOURS plus six RCTD-only ones.
CELLTYPE_ORDER = ["PT", "DTL", "ATL", "TAL", "DCT", "CNT", "PC", "IC", "PapE",
                  "Podo", "PEC", "EC", "VSM/P", "FIB",
                  "Mono_Mac", "T_NK", "B_PL", "Neutro", "Unknown"]
CELLTYPE_COLOURS = {
    "PT": "#2a78d6", "DTL": "#77b5fb", "ATL": "#98688d", "TAL": "#ffa332",
    "DCT": "#5c20d2", "CNT": "#224a71", "PC": "#296904", "IC": "#21968b",
    "PapE": "#c998a2", "Podo": "#f11c09", "PEC": "#f28ffb", "EC": "#49cc95",
    "VSM/P": "#a84af1", "FIB": "#a57c13", "Mono_Mac": "#963215",
    "T_NK": "#bedf0d", "B_PL": "#950681", "Neutro": "#eb31a5",
    "Unknown": "#bdbdb6",
}
NOT_ANN_COL = "#ececE8"      # bin exists, RCTD never saw it   (deck's NOT_ANN)
OTHER_COL = "#e4e4df"        # bin annotated, type not selected (deck's TISSUE)


def samples():
    """The arrays on disk, in natural order, that have binned outputs."""
    out = []
    for d in os.listdir(f"{BASE}/Processed"):
        if d.startswith("KTx_") and os.path.isdir(
                f"{BASE}/Processed/{d}/outs/binned_outputs"):
            out.append(d)
    return sorted(out, key=lambda s: (len(s), s))


class Array:
    """One array at one bin size: gene-major counts plus each bin's grid cell."""

    def __init__(self, sample, res_dir):
        t0 = time.time()
        outs = f"{BASE}/Processed/{sample}/outs/binned_outputs/{res_dir}"
        self.sample, self.res_dir = sample, res_dir
        self.bin_um = BIN_UM[res_dir]
        mpp = json.load(open(f"{outs}/spatial/scalefactors_json.json"))["microns_per_pixel"]

        with h5py.File(f"{outs}/filtered_feature_bc_matrix.h5") as f:
            m = f["matrix"]
            self.genes = m["features/name"][:].astype(str)
            bcs = np.array([b.decode() for b in m["barcodes"][:]])
            data = m["data"][:]
            ind = m["indices"][:]
            ptr = m["indptr"][:]
            ngene, nbc = (int(x) for x in m["shape"][:])

        # ---- gene-major transpose (the fact in the banner) ------------------
        bc = np.repeat(np.arange(nbc, dtype=np.int32), np.diff(ptr))
        o = np.argsort(ind, kind="stable")
        gi_sorted = ind[o].astype(np.int32)
        self.gbin = bc[o]
        self.gdat = data[o].astype(np.int32)
        self.gptr = np.searchsorted(gi_sorted, np.arange(ngene + 1)).astype(np.int64)

        # ---- where each bin sits, in the microscope image frame -------------
        pos = pq.read_table(f"{outs}/spatial/tissue_positions.parquet",
                            columns=["barcode", "pxl_row_in_fullres",
                                     "pxl_col_in_fullres"]).to_pandas()
        xu = pos["pxl_col_in_fullres"].to_numpy(float) * mpp
        yu = pos["pxl_row_in_fullres"].to_numpy(float) * mpp
        self.x0, self.y0 = float(xu.min()), float(yu.min())   # the capture square
        self.span = float(max(xu.max() - self.x0, yu.max() - self.y0))
        where = {b: i for i, b in enumerate(pos["barcode"].to_numpy())}
        sel = np.array([where[b] for b in bcs])
        self.bx, self.by = xu[sel], yu[sel]                   # microns, per matrix column
        self.n_bins = nbc
        self.bcs = bcs                    # matrix column order — the L0 join key
        self.name2i = {n: i for i, n in enumerate(self.genes)}
        self._lab = None                  # filled on first labels() call
        self.load_s = round(time.time() - t0, 2)

    # ---------------------------------------------------------------- counts
    def gene_vector(self, name):
        """UMI per bin for one gene, as (bin index, value) — a contiguous slice."""
        i = self.name2i.get(name)
        if i is None:
            return np.empty(0, np.int32), np.empty(0, np.int32)
        a, b = self.gptr[i], self.gptr[i + 1]
        return self.gbin[a:b], self.gdat[a:b]

    def channel(self, names):
        """Summed UMI per bin for a SET of genes — 'combine markers in one colour'."""
        v = np.zeros(self.n_bins, np.float32)
        used, missing = [], []
        for n in names:
            if n not in self.name2i:
                missing.append(n)
                continue
            bi, dv = self.gene_vector(n)
            # a gene's slice hits each bin at most once, so += is exact here;
            # summing ACROSS genes is what stacks the markers of one channel
            v[bi] += dv
            used.append(n)
        return v, used, missing

    # ----------------------------------------------------------------- grids
    def grid(self, v, window=None):
        """Rasterise a per-bin vector onto its own lattice.

        window = (x0, y0, side) in microns; None = the whole capture square.
        Returns (image, extent, n_bins_in_view).
        """
        x0, y0, side = window if window else (self.x0, self.y0, self.span)
        keep = ((self.bx >= x0) & (self.bx < x0 + side) &
                (self.by >= y0) & (self.by < y0 + side))
        n = max(int(round(side / self.bin_um)), 1)
        gi = np.clip(((self.bx[keep] - x0) / self.bin_um).astype(int), 0, n - 1)
        gj = np.clip(((self.by[keep] - y0) / self.bin_um).astype(int), 0, n - 1)
        g = np.zeros((n, n), np.float32)
        np.maximum.at(g, (gj, gi), v[keep])
        occ = np.zeros((n, n), bool)
        occ[gj, gi] = True
        return g, [x0, x0 + side, y0 + side, y0], int(keep.sum()), occ

    # ------------------------------------------------------- cell-type labels
    def labels(self):
        """The 8 um RCTD call per bin, as codes into CELLTYPE_ORDER.

        Returns (codes, n_rows_matched).  code -1 means the bin has NO ROW in the
        L0 table -- it was never offered to RCTD (counts_MIN 10), which is not the
        same as Unknown, and not the same as empty.  Raises at any other bin size:
        these labels exist at 8 um only and are not inherited up or down.
        """
        if self.res_dir != L0_RES_DIR:
            raise ValueError(f"the RCTD labels are 8 um only, not {self.bin_um:.0f} um")
        if self._lab is None:
            import csv
            code = {t: i for i, t in enumerate(CELLTYPE_ORDER)}
            by_bc = {}
            with open(f"{L0_DIR}/{self.sample}_celltype_008um.csv") as f:
                for row in csv.DictReader(f):
                    by_bc[row["barcode"]] = code[row["cell_type"]]
            v = np.full(self.n_bins, -1, np.int16)
            hit = 0
            for i, b in enumerate(self.bcs):
                c = by_bc.get(b)
                if c is not None:
                    v[i] = c
                    hit += 1
            # every row must land on a bin of this matrix, or the join is wrong
            if hit != len(by_bc):
                raise ValueError(f"{self.sample}: {len(by_bc) - hit} L0 rows did not "
                                 f"match a bin barcode — wrong array or wrong run")
            self._lab = (v, hit)
        return self._lab

    def grid_code(self, codes, window=None):
        """Rasterise per-bin CODES (not counts).  -2 = no bin in that cell.

        Plain assignment, not np.maximum.at: the lattice pitch IS the bin size, so
        each cell receives exactly one bin and there is nothing to reduce.  Using
        max here would silently prefer the higher-numbered type on a collision.
        """
        x0, y0, side = window if window else (self.x0, self.y0, self.span)
        keep = ((self.bx >= x0) & (self.bx < x0 + side) &
                (self.by >= y0) & (self.by < y0 + side))
        n = max(int(round(side / self.bin_um)), 1)
        gi = np.clip(((self.bx[keep] - x0) / self.bin_um).astype(int), 0, n - 1)
        gj = np.clip(((self.by[keep] - y0) / self.bin_um).astype(int), 0, n - 1)
        g = np.full((n, n), -2, np.int16)
        g[gj, gi] = codes[keep]
        return g, [x0, x0 + side, y0 + side, y0], int(keep.sum())


@functools.lru_cache(maxsize=3)
def load(sample, res_dir):
    return Array(sample, res_dir)


def compose(arr, chan_genes, vmax, window=None, floor=0.10, mask=None):
    """The picture: three gene SETS -> one additive RGB image.

    chan_genes is [red_list, green_list, blue_list].  Channel brightness is
    summed marker UMI per bin divided by `vmax`, clipped — the same linear rule
    celltype_rgb_overlay.py uses, so the app and the deck figure agree.

    `mask` (per-bin bool, optional) keeps only the bins it selects — used by the
    app's "restrict to the selected cell types".  It ZEROES the counts of the
    bins it drops rather than removing them, so the lattice, the extent and every
    ratio's denominator are unchanged; what the picture then shows is the same
    expression seen through the annotation.
    """
    vecs, used, missing = [], [], []
    for names in chan_genes:
        v, u, m = arr.channel(names)
        if mask is not None:
            v = np.where(mask, v, 0)
        vecs.append(v); used.append(u); missing.append(m)
    grids, occ = [], None
    for v in vecs:
        g, ext, nview, o = arr.grid(v, window)
        grids.append(g)
        occ = o
    base = np.where(occ, floor, 0.02)
    rgb = np.stack([np.clip(g / max(vmax, 1e-9), 0, 1) for g in grids], -1)
    rgb = np.maximum(rgb, base[..., None])
    return rgb, ext, nview, vecs, used, missing


# ============================== GRADUAL COLOUR ================================
# The RGB view answers "which set is here"; it cannot show HOW MUCH, because one
# channel only ever ramps one hue and three of them mix.  A single gene set on a
# perceptually-uniform colourmap answers the other question, and is the only view
# here that can carry a COLOURBAR -- so the numbers are readable off the picture
# rather than inferred from brightness.
#
# vmax means exactly what it means in compose(): UMI per bin at full colour, and
# everything above is clipped.  auto_vmax() offers a percentile instead of a
# guess, and the app always PRINTS the value it used -- a gradient with an
# unstated ceiling is a figure that cannot be compared with any other.
def auto_vmax(v, window_keep=None, mask=None, pct=99.0):
    """A ceiling from the data: the `pct`th percentile of the POSITIVE bins shown.

    Positive-only because the field is mostly zero at these bin sizes -- over all
    bins the 99th percentile of a rare gene is still 0, which would blow the
    scale up to nothing.  Returns None when nothing is positive.
    """
    x = v
    if window_keep is not None:
        x = x[window_keep]
        if mask is not None:
            x = np.where(mask[window_keep], x, 0)
    elif mask is not None:
        x = np.where(mask, x, 0)
    x = x[x > 0]
    if x.size == 0:
        return None
    return float(np.percentile(x, pct))


def compose_gradient(arr, names, vmax, cmap, window=None, mask=None):
    """One gene SET on a continuous colourmap, plus what the app needs to label it.

    Returns (rgb, ext, n_in_view, v, used, missing, shown) where `shown` is the
    per-cell boolean the app greys out: a cell with no bin, or a bin the mask
    dropped.  Those two are drawn in DIFFERENT greys, because "outside the
    selected cell types" and "outside the tissue" are not the same statement.
    """
    v, used, missing = arr.channel(names)
    g, ext, nview, occ = arr.grid(v, window)
    rgb = cmap(np.clip(g / max(vmax, 1e-9), 0, 1))[..., :3]
    shown = occ
    if mask is not None:
        mg, _, _, _ = arr.grid(mask.astype(np.float32), window)
        keep = mg > 0
        rgb[occ & ~keep] = _hex(OTHER_COL)        # bin, but not a selected type
        shown = occ & keep
    rgb[~occ] = 1.0                               # outside the lattice: white
    return rgb, ext, nview, v, used, missing, shown


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


# ============================== CELL-TYPE MAP =================================
def celltype_mask(arr, types):
    """Per-bin bool: is this bin's 8 um RCTD call one of `types`?"""
    codes, _ = arr.labels()
    want = {CELLTYPE_ORDER.index(t) for t in types if t in CELLTYPE_ORDER}
    if not want:
        return np.zeros(arr.n_bins, bool)
    return np.isin(codes, list(want))


def compose_celltypes(arr, types, window=None):
    """The annotation itself: selected types in their deck colours.

    FOUR STATES, four colours, and they are NOT interchangeable:
      selected type  -> its own colour (Unknown included, if selected)
      other type     -> OTHER_COL, a bin RCTD called something else
      no row         -> NOT_ANN_COL, a bin RCTD was never given
      no bin         -> white, outside the lattice
    """
    codes, _ = arr.labels()
    g, ext, nview = arr.grid_code(codes, window)
    rgb = np.ones(g.shape + (3,), np.float32)
    rgb[g >= 0] = _hex(OTHER_COL)
    rgb[g == -1] = _hex(NOT_ANN_COL)
    for t in types:
        if t in CELLTYPE_ORDER:
            rgb[g == CELLTYPE_ORDER.index(t)] = _hex(CELLTYPE_COLOURS[t])
    return rgb, ext, nview


def celltype_stats(arr, types, window=None, vecs=None):
    """Counts per type in view — and, if gene channels are given, the overlap.

    The overlap row is the question the layer exists to answer: of the bins RCTD
    called type T, how many carry any UMI of each channel's gene set.  It is a
    CO-LOCATION count at the bin size, exactly like the RGB view's pairs, not
    evidence that the labelled cell expresses the gene.
    """
    codes, n_rows = arr.labels()
    if window:
        x0, y0, side = window
        keep = ((arr.bx >= x0) & (arr.bx < x0 + side) &
                (arr.by >= y0) & (arr.by < y0 + side))
    else:
        keep = np.ones(arr.n_bins, bool)
    c = codes[keep]
    n_view = int(keep.sum())
    n_ann = int((c >= 0).sum())
    rows = []
    for t in types:
        if t not in CELLTYPE_ORDER:
            continue
        sel = c == CELLTYPE_ORDER.index(t)
        n = int(sel.sum())
        row = dict(type=t, bins=n,
                   pct_view=round(100 * n / n_view, 2) if n_view else 0.0,
                   pct_ann=round(100 * n / n_ann, 2) if n_ann else 0.0,
                   colour=CELLTYPE_COLOURS[t], overlap=[])
        if vecs is not None:
            # always one entry per channel, zeros included: the caller renders this
            # as table cells, and a short row would silently shift the columns
            if n:
                m = np.zeros(arr.n_bins, bool)
                m[np.where(keep)[0][sel]] = True
                row["overlap"] = [int(((v > 0) & m).sum()) for v in vecs]
            else:
                row["overlap"] = [0] * len(vecs)
        rows.append(row)
    return dict(bins_in_view=n_view, annotated=n_ann,
                not_annotated=n_view - n_ann,
                pct_annotated=round(100 * n_ann / n_view, 1) if n_view else 0.0,
                unknown=int((c == CELLTYPE_ORDER.index("Unknown")).sum()),
                rows=rows, n_rows_total=n_rows)


def channel_stats(arr, vecs, window=None):
    """Per channel and per combination — the same arithmetic as the deck figure."""
    if window:
        x0, y0, side = window
        keep = ((arr.bx >= x0) & (arr.bx < x0 + side) &
                (arr.by >= y0) & (arr.by < y0 + side))
    else:
        keep = np.ones(arr.n_bins, bool)
    P = [(v[keep] > 0) for v in vecs]
    npos = np.array(P, int).sum(0)
    per = [dict(umi=float(v[keep].sum()), bins=int(p.sum()),
                pct=round(100 * float(p.mean()), 2)) for v, p in zip(vecs, P)]
    any_ = int((P[0] | P[1] | P[2]).sum())
    return dict(
        bins_in_view=int(keep.sum()), per_channel=per, bins_any=any_,
        exactly_1=int((npos == 1).sum()), exactly_2=int((npos == 2).sum()),
        all_3=int((npos == 3).sum()),
        pairs={"R+G": int((P[0] & P[1]).sum()), "R+B": int((P[0] & P[2]).sum()),
               "G+B": int((P[1] & P[2]).sum())})


if __name__ == "__main__":
    a = load("KTx_18", "square_008um")
    print(f"{a.sample} {a.res_dir}: {a.n_bins:,} bins, {len(a.genes):,} genes, "
          f"loaded in {a.load_s}s, span {a.span:.0f} µm")
    t = time.time()
    rgb, ext, n, vecs, used, miss = compose(
        a, [["NPHS1", "NPHS2", "PODXL"], ["PECAM1", "EGFL7", "CDH5"],
            ["ACTA2", "TAGLN", "RGS5"]], vmax=5.0)
    st = channel_stats(a, vecs)
    print(f"compose whole section {time.time()-t:.2f}s -> {rgb.shape}, {n:,} bins in view")
    for lab, s_ in zip("RGB", st["per_channel"]):
        print(f"  {lab}: {s_['umi']:,.0f} UMI  {s_['bins']:,} bins ({s_['pct']}%)")
    print("  ", {k: st[k] for k in ("bins_any", "exactly_1", "exactly_2", "all_3")},
          st["pairs"])

    # ---- cell-type layer -------------------------------------------------
    t = time.time()
    codes, hit = a.labels()
    print(f"\nlabels: {hit:,} of {a.n_bins:,} bins have an L0 row "
          f"({100*hit/a.n_bins:.1f}%), in {time.time()-t:.2f}s")
    cs = celltype_stats(a, ["Podo", "EC", "PT"], vecs=vecs)
    print(f"  annotated {cs['annotated']:,}  not annotated {cs['not_annotated']:,}  "
          f"Unknown {cs['unknown']:,}")
    for r in cs["rows"]:
        print(f"  {r['type']:>6s}: {r['bins']:7,d} bins  {r['pct_ann']:5.2f}% of "
              f"annotated   overlap R/G/B {r['overlap']}")
    t = time.time()
    img, ext, nv = compose_celltypes(a, ["Podo", "EC"])
    print(f"  celltype map {time.time()-t:.2f}s -> {img.shape}")

    # ---- gradient --------------------------------------------------------
    import matplotlib
    v, _, _ = a.channel(["NPHS2"])
    av = auto_vmax(v)
    t = time.time()
    g, ext, nv, v2, used2, miss2, shown = compose_gradient(
        a, ["NPHS2"], av or 5.0, matplotlib.colormaps["viridis"])
    print(f"  gradient (auto vmax {av}) {time.time()-t:.2f}s -> {g.shape}, "
          f"{int(shown.sum()):,} cells shown")

    # ---- mask: the same genes seen only inside one RCTD type --------------
    m = celltype_mask(a, ["Podo"])
    _, _, _, vm, _, _ = compose(a, [["NPHS2"], [], []], 5.0, mask=m)
    print(f"  NPHS2 UMI: {v.sum():,.0f} everywhere, {vm[0].sum():,.0f} inside "
          f"Podo bins ({100*vm[0].sum()/max(v.sum(),1):.1f}%)")
