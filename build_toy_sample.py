#!/usr/bin/env python3
# ==============================================================================
# build_toy_sample.py — cut a ~1 mm block out of one array as a toy sample.
#
# WHY THIS EXISTS (author, 2026-10-05: "can you add a toy sample?")
#   The app-only repository holds no data, so a fresh clone has nothing to show.
#   This writes a small sample, KTx_toy, in the layout the app reads, so the app
#   runs straight after `git clone`.  The author chose a REAL crop of KTx_18
#   (expression, cell-type labels and H&E), not synthetic data.
#
# WHAT IT CUTS
#   A square block of N x N 8 um bins on the LATTICE (array_row / array_col),
#   not a box in the image: the 2 um bins inside it are then exactly the 16
#   children of each 8 um bin, so the app's nesting check holds with no partial
#   parents.  The block is the one with the most Podo-labelled bins among blocks
#   that are >= 50% in tissue and whose H&E lies fully on the scan, i.e. the
#   block richest in glomeruli.  50%, not more: the tissue is biopsy cores about
#   1 mm wide, so no 1 mm square of KTx_18 is more than 69% tissue.
#
# WHAT CHANGES IN THE COPY, AND WHAT DOES NOT
#   - pxl_row/col_in_fullres are shifted by the H&E crop's corner, so they index
#     the cropped image.  Nothing else about a bin moves.
#   - barcodes, array_row/col, counts and labels are the originals, unaltered.
#   - all 18,085 genes are kept, so the gene pickers behave as on a full array.
#   - the H&E is the native-resolution crop, re-tiled like build_he_index.py.
#     There is no full-resolution TIF, so the export button has nothing to read.
#
# INPUT    $SPATIAL_BASE: Processed/<src>/..., L0_CellType/, HiRes_Images/
# OUTPUT   toy/ next to this file (or --out DIR):
#            Processed/KTx_toy/outs/binned_outputs/square_{002,008}um/...
#            L0_CellType/KTx_toy_celltype_008um.csv
#            HiRes_Index/KTx_toy/{manifest.json, L<k>/<ty>_<tx>.jpg}
#            toy_source.json   every number about the cut
# RUN      ../spatial-venv/bin/python SpatialRGB/build_toy_sample.py
#          then  git add -f SpatialRGB/toy   (-f: the project .gitignore hides
#          *.h5, *.parquet, *.csv, *.jpg and Processed/, so new files in toy/
#          are otherwise skipped without a word)
# ==============================================================================
import csv
import json
import os
import shutil
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import he_image as HE                                           # noqa: E402
import spatial_rgb_data as D                                    # noqa: E402

SRC = "KTx_18"
TOY = "KTx_toy"
N8 = 128                    # 8 um bins per side -> 1024 um
STRIDE = 16                 # candidate blocks every 16 bins (128 um)
MIN_TISSUE = 0.50
MARGIN_PX = 24              # H&E kept around the outermost bin centres
OUT = (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv
       else os.path.join(HERE, "toy"))

Image.MAX_IMAGE_PIXELS = None
B = f"{D.BASE}/Processed/{SRC}/outs/binned_outputs"


def positions(res):
    return pq.read_table(f"{B}/{res}/spatial/tissue_positions.parquet").to_pandas()


# ---------------------------------------------------------------- pick the block
p8 = positions("square_008um")
lab = {}
with open(f"{D.BASE}/L0_CellType/{SRC}_celltype_008um.csv") as f:
    for row in csv.DictReader(f):
        lab[row["barcode"]] = row["cell_type"]
r8 = p8["array_row"].to_numpy().astype(int)
c8 = p8["array_col"].to_numpy().astype(int)
nr, nc = r8.max() + 1, c8.max() + 1
tissue = np.zeros((nr, nc), np.int32)
podo = np.zeros((nr, nc), np.int32)
tissue[r8, c8] = p8["in_tissue"].to_numpy()
podo[r8, c8] = [lab.get(b) == "Podo" for b in p8["barcode"]]

he = HE.HEImage(SRC)
tif = he._tiff()
if not (he.path and tif.supported()):
    sys.exit(f"{SRC}: the microscope TIF is missing or not strip-readable")
px = np.full((nr, nc), np.nan); py = np.full((nr, nc), np.nan)
px[r8, c8] = p8["pxl_col_in_fullres"].to_numpy(float)
py[r8, c8] = p8["pxl_row_in_fullres"].to_numpy(float)
half8 = 4.0 / he.mpp                              # half an 8 um bin, in pixels

cand = []
for r0 in range(0, nr - N8 + 1, STRIDE):
    for c0 in range(0, nc - N8 + 1, STRIDE):
        sl = (slice(r0, r0 + N8), slice(c0, c0 + N8))
        frac = tissue[sl].mean()
        x_lo, x_hi = px[sl].min() - half8, px[sl].max() + half8
        y_lo, y_hi = py[sl].min() - half8, py[sl].max() + half8
        on_scan = (x_lo - MARGIN_PX >= 0 and y_lo - MARGIN_PX >= 0 and
                   x_hi + MARGIN_PX <= he.width and y_hi + MARGIN_PX <= he.height)
        if frac >= MIN_TISSUE and on_scan:
            cand.append((int(podo[sl].sum()), float(frac), r0, c0))
if not cand:
    sys.exit("no block passed the tissue / on-scan filter")
cand.sort(reverse=True)
npod, frac, r0, c0 = cand[0]
print(f"{len(cand)} candidate blocks; chosen rows {r0}-{r0+N8-1}, cols {c0}-{c0+N8-1}: "
      f"{npod} Podo bins, {100*frac:.1f}% in tissue (runner-up {cand[1][0]} Podo bins)")

# ------------------------------------------------------------- the H&E corner
sl = (slice(r0, r0 + N8), slice(c0, c0 + N8))
X0 = int(np.floor(px[sl].min() - half8 - MARGIN_PX))
Y0 = int(np.floor(py[sl].min() - half8 - MARGIN_PX))
X1 = int(np.ceil(px[sl].max() + half8 + MARGIN_PX))
Y1 = int(np.ceil(py[sl].max() + half8 + MARGIN_PX))

if os.path.exists(OUT):
    shutil.rmtree(OUT)
summary = dict(source=SRC, toy=TOY, block_8um=dict(row0=r0, col0=c0, n=N8),
               side_um=N8 * 8, podo_bins_8um=npod, in_tissue_frac=round(frac, 4),
               candidates=len(cand), he_crop_px=[X0, Y0, X1, Y1], res={})


# ------------------------------------------------- matrices and bin positions
def cut(res, div):
    """Write the block at one bin size.  div = 8 um bin / this bin size."""
    pos = p8 if res == "square_008um" else positions(res)
    ar = pos["array_row"].to_numpy().astype(int) // div
    ac = pos["array_col"].to_numpy().astype(int) // div
    keep = (ar >= r0) & (ar < r0 + N8) & (ac >= c0) & (ac < c0 + N8)
    sub = pos[keep].copy()
    sub["pxl_col_in_fullres"] = sub["pxl_col_in_fullres"] - X0
    sub["pxl_row_in_fullres"] = sub["pxl_row_in_fullres"] - Y0
    out = f"{OUT}/Processed/{TOY}/outs/binned_outputs/{res}"
    os.makedirs(f"{out}/spatial", exist_ok=True)
    schema = pq.read_schema(f"{B}/{res}/spatial/tissue_positions.parquet")
    pq.write_table(pa.Table.from_pandas(sub, schema=schema, preserve_index=False),
                   f"{out}/spatial/tissue_positions.parquet", compression="zstd")
    shutil.copy(f"{B}/{res}/spatial/scalefactors_json.json", f"{out}/spatial/")

    inside = set(sub["barcode"])
    with h5py.File(f"{B}/{res}/filtered_feature_bc_matrix.h5") as f:
        m = f["matrix"]
        bcs = m["barcodes"][:]
        cols = np.flatnonzero([b.decode() in inside for b in bcs])
        ptr = m["indptr"][:]
        n = (ptr[cols + 1] - ptr[cols]).astype(np.int64)
        new_ptr = np.concatenate([[0], np.cumsum(n)])
        idx = np.repeat(ptr[cols] - new_ptr[:-1], n) + np.arange(new_ptr[-1])
        data = m["data"][:][idx]
        ind = m["indices"][:][idx]
        feats = {k: m[f"features/{k}"][:] for k in ("name", "id", "feature_type",
                                                    "genome")}
        ngene = int(m["shape"][0])
    with h5py.File(f"{out}/filtered_feature_bc_matrix.h5", "w") as g:
        mm = g.create_group("matrix")
        mm.create_dataset("barcodes", data=bcs[cols], compression="gzip")
        mm.create_dataset("data", data=data, compression="gzip")
        mm.create_dataset("indices", data=ind, compression="gzip")
        mm.create_dataset("indptr", data=new_ptr)
        mm.create_dataset("shape", data=np.array([ngene, cols.size], np.int32))
        ff = mm.create_group("features")
        for k, v in feats.items():
            ff.create_dataset(k, data=v, compression="gzip")
        g.attrs["filetype"] = "matrix"
        g.attrs["library_ids"] = np.array([TOY.encode()])
    summary["res"][res] = dict(lattice_bins=int(keep.sum()), matrix_bins=int(cols.size),
                               umi=int(data.sum()), nonzeros=int(data.size))
    print(f"  {res}: {int(keep.sum()):,} lattice bins, {cols.size:,} in the matrix, "
          f"{int(data.sum()):,} UMI")
    return [b.decode() for b in bcs[cols]]


bc8 = cut("square_008um", 1)
cut("square_002um", 4)

# ------------------------------------------------------------ cell-type labels
os.makedirs(f"{OUT}/L0_CellType", exist_ok=True)
inside8 = set(bc8)
n_lab = 0
with open(f"{D.BASE}/L0_CellType/{SRC}_celltype_008um.csv") as f, \
        open(f"{OUT}/L0_CellType/{TOY}_celltype_008um.csv", "w", newline="") as g:
    rd = csv.DictReader(f)
    wr = csv.DictWriter(g, fieldnames=rd.fieldnames, quoting=csv.QUOTE_ALL)
    wr.writeheader()
    for row in rd:
        if row["barcode"] in inside8:
            wr.writerow(row)
            n_lab += 1
summary["labelled_bins_8um"] = n_lab
print(f"  labels: {n_lab:,} of {len(bc8):,} matrix bins have an L0 row")

# ------------------------------------------------------------------ the H&E
im = Image.fromarray(np.ascontiguousarray(tif.rows(Y0, Y1)[:, X0:X1]))
W0, H0 = im.size
assert (W0, H0) == (X1 - X0, Y1 - Y0), "H&E crop left the scan"
idx_dir = f"{OUT}/HiRes_Index/{TOY}"
sizes, n_tiles = [], 0
for lv in range(HE.N_LEVELS):                     # same tiling as build_he_index.py
    lim = im if lv == 0 else im.resize((max(1, W0 >> lv), max(1, H0 >> lv)),
                                       Image.LANCZOS)
    w, h = lim.size
    sizes.append([w, h])
    os.makedirs(f"{idx_dir}/L{lv}", exist_ok=True)
    for ty in range((h + HE.TILE - 1) // HE.TILE):
        for tx in range((w + HE.TILE - 1) // HE.TILE):
            box = (tx * HE.TILE, ty * HE.TILE,
                   min((tx + 1) * HE.TILE, w), min((ty + 1) * HE.TILE, h))
            lim.crop(box).save(f"{idx_dir}/L{lv}/{ty}_{tx}.jpg", quality=HE.JPEG_Q,
                               subsampling=0)
            n_tiles += 1
json.dump(dict(sample=TOY, width=W0, height=H0, tile=HE.TILE, levels=HE.N_LEVELS,
               level_size=sizes, quality=HE.JPEG_Q, mpp=he.mpp, n_tiles=n_tiles,
               source=f"crop of {SRC}", complete=True),
          open(f"{idx_dir}/manifest.json", "w"), indent=1)
summary["he"] = dict(width=W0, height=H0, tiles=n_tiles, mpp=he.mpp)
print(f"  H&E: {W0} x {H0} px, {n_tiles} tiles")

size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs)
summary["bytes"] = size
json.dump(summary, open(f"{OUT}/toy_source.json", "w"), indent=1)
print(f"wrote {OUT}: {size/1e6:.1f} MB")
