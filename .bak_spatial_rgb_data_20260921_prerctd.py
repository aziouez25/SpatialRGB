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
        self.name2i = {n: i for i, n in enumerate(self.genes)}
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


@functools.lru_cache(maxsize=3)
def load(sample, res_dir):
    return Array(sample, res_dir)


def compose(arr, chan_genes, vmax, window=None, floor=0.10):
    """The picture: three gene SETS -> one additive RGB image.

    chan_genes is [red_list, green_list, blue_list].  Channel brightness is
    summed marker UMI per bin divided by `vmax`, clipped — the same linear rule
    celltype_rgb_overlay.py uses, so the app and the deck figure agree.
    """
    vecs, used, missing = [], [], []
    for names in chan_genes:
        v, u, m = arr.channel(names)
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
