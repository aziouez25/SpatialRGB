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
L0_RES_DIR = "square_008um"                 # the bin size these labels are COMPUTED at
L0_BIN_UM = 8.0

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
                                     "pxl_col_in_fullres",
                                     "array_row", "array_col"]).to_pandas()
        xu = pos["pxl_col_in_fullres"].to_numpy(float) * mpp
        yu = pos["pxl_row_in_fullres"].to_numpy(float) * mpp
        self.x0, self.y0 = float(xu.min()), float(yu.min())   # the capture square
        self.span = float(max(xu.max() - self.x0, yu.max() - self.y0))
        where = {b: i for i, b in enumerate(pos["barcode"].to_numpy())}
        sel = np.array([where[b] for b in bcs])
        self.bx, self.by = xu[sel], yu[sel]                   # microns, per matrix column
        # the LATTICE's own indices — not a drawing frame (they are rotated relative
        # to the image here), but the only thing the 2 um / 8 um nesting is defined on
        self.arow = pos["array_row"].to_numpy()[sel].astype(np.int64)
        self.acol = pos["array_col"].to_numpy()[sel].astype(np.int64)
        self.n_bins = nbc
        self.bcs = bcs                    # matrix column order — the L0 join key
        self.name2i = {n: i for i, n in enumerate(self.genes)}
        self._lab = None                  # filled on first labels() call
        self._nest = None                 # filled on first nesting_check()
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

        Returns (codes, n_rows_matched).  code -1 means NO LABEL -- at 8 um that is
        a bin with no row in the L0 table (never offered to RCTD, counts_MIN 10),
        at 2 um a bin whose parent has none.  Neither is "Unknown", and neither is
        "empty".

        AT 2 um THE LABEL IS INHERITED FROM THE 8 um PARENT.  That direction is the
        only one that is well posed: an 8 um bin is exactly 4x4 of these, so each
        2 um bin has ONE parent.  Upwards it is not -- a 16 um bin holds four 8 um
        bins that may disagree -- so 16 um still raises.
        """
        if self.res_dir == L0_RES_DIR:
            if self._lab is None:
                self._lab = (self._labels_direct(), None)
                self._lab = (self._lab[0], int((self._lab[0] >= 0).sum()))
            return self._lab
        if self.res_dir == "square_002um":
            if self._lab is None:
                self._lab = self._labels_inherited()
            return self._lab
        raise ValueError(
            f"the RCTD labels are 8 um, and inherit only downwards to 2 um — a "
            f"{self.bin_um:.0f} um bin holds several 8 um bins that can disagree")

    def _read_l0(self):
        """barcode -> code, straight off the Layer 0 CSV."""
        import csv
        code = {t: i for i, t in enumerate(CELLTYPE_ORDER)}
        by_bc = {}
        with open(f"{L0_DIR}/{self.sample}_celltype_008um.csv") as f:
            for row in csv.DictReader(f):
                by_bc[row["barcode"]] = code[row["cell_type"]]
        return by_bc

    def _labels_direct(self):
        by_bc = self._read_l0()
        v = np.full(self.n_bins, -1, np.int16)
        hit = 0
        for i, b in enumerate(self.bcs):
            c = by_bc.get(b)
            if c is not None:
                v[i] = c
                hit += 1
        if hit != len(by_bc):                 # every row must land on a bin here
            raise ValueError(f"{self.sample}: {len(by_bc) - hit} L0 rows did not "
                             f"match a bin barcode — wrong array or wrong run")
        return v

    def _labels_inherited(self):
        """Each 2 um bin takes the label of the 8 um bin that contains it.

        The join is the repo's existing one (celltype_002um_vs_labels.py):
        parent = (array_row // 4, array_col // 4) on the LATTICE indices.  It is
        not trusted — nesting_check() re-measures it against the two coordinate
        tables every time an array is loaded, and this refuses to return labels if
        the grids do not actually nest.
        """
        nest = self.nesting_check()
        if nest["max_offset_um"] > 0.5:       # half a 2 um bin: generous, and never hit
            raise ValueError(f"{self.sample}: 2 um and 8 um grids do not nest "
                             f"(max centre offset {nest['max_offset_um']:.3f} µm)")
        by_bc = self._read_l0()
        p8 = pq.read_table(
            f"{BASE}/Processed/{self.sample}/outs/binned_outputs/{L0_RES_DIR}/"
            "spatial/tissue_positions.parquet",
            columns=["barcode", "array_row", "array_col"]).to_pandas()
        pr = p8["array_row"].to_numpy().astype(np.int64)
        pc = p8["array_col"].to_numpy().astype(np.int64)
        codes8 = np.array([by_bc.get(b, -1) for b in p8["barcode"].to_numpy()], np.int16)
        keep = codes8 >= 0
        key8 = (pr[keep] << 20) | pc[keep]
        o = np.argsort(key8)
        key8, val8 = key8[o], codes8[keep][o]

        div = int(round(L0_BIN_UM / self.bin_um))          # 4
        key2 = ((self.arow // div) << 20) | (self.acol // div)
        i = np.searchsorted(key8, key2)
        ok = (i < key8.size) & (key8[np.clip(i, 0, key8.size - 1)] == key2)
        v = np.where(ok, val8[np.clip(i, 0, val8.size - 1)], -1).astype(np.int16)
        return v, int(ok.sum())

    def nesting_check(self):
        """Prove the 2 um grid nests in the 8 um one before anything relies on it.

        Every 8 um bin's centre is compared with the MEAN CENTRE of the 2 um bins
        that map to it.  Bins at the far edge of the array hold fewer than 16
        children (the lattice is not a multiple of 4), so they are counted
        separately instead of silently inflating the offset.  Same measurement as
        celltype_002um_vs_labels.py's nesting().
        """
        if self._nest is not None:
            return self._nest
        B = f"{BASE}/Processed/{self.sample}/outs/binned_outputs"
        c2 = pq.read_table(f"{B}/square_002um/spatial/tissue_positions.parquet",
                           columns=["array_row", "array_col", "pxl_row_in_fullres",
                                    "pxl_col_in_fullres"]).to_pandas()
        p8 = pq.read_table(f"{B}/{L0_RES_DIR}/spatial/tissue_positions.parquet",
                           columns=["array_row", "array_col", "pxl_row_in_fullres",
                                    "pxl_col_in_fullres"]).to_pandas()
        mpp = json.load(open(f"{B}/{L0_RES_DIR}/spatial/scalefactors_json.json")
                        )["microns_per_pixel"]
        div = 4
        pr = c2["array_row"].to_numpy().astype(np.int64) // div
        pc = c2["array_col"].to_numpy().astype(np.int64) // div
        key = (pr << 20) | pc
        uk, inv = np.unique(key, return_inverse=True)
        n = np.bincount(inv)
        sx = np.bincount(inv, weights=c2["pxl_col_in_fullres"].to_numpy(float))
        sy = np.bincount(inv, weights=c2["pxl_row_in_fullres"].to_numpy(float))
        k8 = ((p8["array_row"].to_numpy().astype(np.int64) << 20)
              | p8["array_col"].to_numpy().astype(np.int64))
        j = np.searchsorted(uk, k8)
        ok = (j < uk.size) & (uk[np.clip(j, 0, uk.size - 1)] == k8)
        j = j[ok]
        full = n[j] == div * div
        dx = np.abs(sx[j] / n[j] - p8["pxl_col_in_fullres"].to_numpy(float)[ok])
        dy = np.abs(sy[j] / n[j] - p8["pxl_row_in_fullres"].to_numpy(float)[ok])
        d = np.maximum(dx, dy)[full]
        self._nest = dict(
            divisor=div, n_child=div * div, n_parent=int(ok.sum()),
            n_parent_full=int(full.sum()), n_parent_partial=int((~full).sum()),
            max_offset_px=float(d.max()) if d.size else float("nan"),
            max_offset_um=float(d.max() * mpp) if d.size else float("nan"))
        return self._nest

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


# ====================== THE GRADIENTS AND THE BACKGROUND ======================
# Each channel is a CONTINUOUS ramp from the background up to full hue at `vmax`,
# and the three are ADDED, so a bin carrying red and green reads as the mix of two
# gradients rather than as a flat "both present" colour.
#
# WHY THE SCALE IS SELECTABLE
#   These are counts, and small ones: a marker panel puts 1-3 UMI in most bins it
#   touches and 10+ in a few.  On a linear ramp with any vmax low enough to show
#   the 1s, the 10s saturate and every structure reads as the same flat hue --
#   which is exactly the complaint that "the gradient" was meant to fix.  sqrt and
#   log keep the top of the range from eating the bottom.  The ramp is reported
#   next to the picture, because a gradient whose transfer function is unstated
#   cannot be compared with another figure.
def _ramp(x, scale):
    x = np.clip(x, 0.0, 1.0)
    if scale == "sqrt":
        return np.sqrt(x)
    if scale == "log":
        return np.log1p(9.0 * x) / np.log(10.0)      # 0->0, 1->1, lifts the low end
    return x


RAMPS = ("linear", "sqrt", "log")


def safe_ground(vmax, scale, margin=0.90):
    """The highest ground that cannot HIDE a positive bin.

    The blend is max(hue, ground), so a bin whose every channel ramps below the
    ground is painted over and disappears — and it disappears ONLY inside the
    lifted cell type, which is exactly where the eye is being asked to judge
    density.  The faintest thing that can exist is 1 UMI (counts are integers),
    so the ground must stay under ramp(1 / vmax).

    Measured on KTx_18 at 8 um with a linear ramp and vmax 5: a bin needs 1.10 UMI
    to clear a 0.22 ground against 0.43 to clear black, and 14,313 of 19,340
    positive bins (74%) vanish inside the lifted region.  On sqrt the same ground
    needs 0.24 UMI and hides none.  Hence this is enforced, not documented.
    """
    return float(margin * _ramp(np.array(1.0 / max(vmax, 1e-9)), scale))


def hidden_by_ground(vecs, vmax, scale, ground, keep=None):
    """How many positive bins the ground would paint over — reported, not assumed."""
    stack = np.stack([v if keep is None else v[keep] for v in vecs], -1)
    pos = stack.max(1) > 0
    if not pos.any():
        return 0, 0
    r = _ramp(stack / max(vmax, 1e-9), scale).max(1)
    return int((pos & (r < ground)).sum()), int(pos.sum())


def celltype_background(arr, types, base=0.085, sel=0.22):
    """Per-bin grey level: the RCTD map used as the GROUND under the gradients.

    Every in-tissue bin sits at `base` — a near-black that shows the tissue
    outline without competing with a colour — and the bins whose RCTD call is one
    of `types` lift to `sel`, a greyish black.  So choosing a cell type does not
    recolour the picture; it silhouettes where that type is, underneath whatever
    the gene channels are doing.

    Returns None when there is nothing to lift, so the caller keeps the flat
    ground and never pays for the label join.
    """
    if not types:
        return None
    codes, _ = arr.labels()
    want = [CELLTYPE_ORDER.index(t) for t in types if t in CELLTYPE_ORDER]
    if not want:
        return None
    g = np.full(arr.n_bins, base, np.float32)
    g[np.isin(codes, want)] = sel
    return g


def compose(arr, chan_genes, vmax, window=None, floor=0.085, mask=None,
            scale="linear", bg=None, void=0.02):
    """The picture: three gene SETS -> one image of three mixed gradients.

    chan_genes is [red_list, green_list, blue_list]; channel intensity is summed
    marker UMI per bin over `vmax`, put through `scale`, and ADDED to the grey
    ground so the hue survives (taking a maximum instead would let a grey ground
    swallow a faint channel whole).

    `bg` (per-bin grey, from celltype_background) is the RCTD layer.  `mask`
    (per-bin bool) instead keeps only the bins it selects, ZEROING the rest, so
    the lattice, the extent and every denominator are unchanged.

    THE BLEND IS A MAXIMUM, NOT A SUM, and that was measured rather than assumed:
    adding the grey ground to the hue desaturates every strong channel (a red
    tuft over a lifted ground came out pink), while a maximum lets the ground
    show only where no channel reaches it, so the hues stay clean.  It also means
    that with NO type lifted the rule reduces to max(expr, floor) -- exactly what
    this function did before, and what deck slide 12 draws.
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
    if bg is None:
        grey = np.where(occ, floor, void)
    else:
        bgg, _, _, _ = arr.grid(bg, window)          # one bin per cell: exact
        grey = np.where(occ, np.maximum(bgg, floor), void)
    expr = np.stack([_ramp(g / max(vmax, 1e-9), scale) for g in grids], -1)
    rgb = np.maximum(expr, grey[..., None])
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

    # The LEVEL of a channel, not just its extent.  Asked for 2026-09-21 after
    # "there is only one UMI per 8 um bins for NPHS2 for example?" — a fair
    # challenge to a claim made from an array-wide mean.  A single gene's positive
    # bins really do sit at a median of 1 UMI, while the PANEL SUM inside the
    # tissue that expresses it sits at 3, and only one of those numbers is what
    # the channel draws.  Both are cheap, so the table stops hiding the difference.
    def lvl(v, p):
        x = v[keep][p]
        return (dict(median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                     mx=float(x.max())) if x.size else
                dict(median=0.0, p90=0.0, mx=0.0))

    per = [dict(umi=float(v[keep].sum()), bins=int(p.sum()),
                pct=round(100 * float(p.mean()), 2), **lvl(v, p))
           for v, p in zip(vecs, P)]
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
