#!/usr/bin/env python3
# ==============================================================================
# he_image.py — crop the high-resolution microscope H&E for a window given in
#               the app's micron frame.
#
# WHY THIS EXISTS (author, 2026-10-05)
#   "whenever I zoom to a specific location, [I] will get the same zoom in the
#   high resolution H&E image.  As the H&E images are very heavy we might need
#   some indexation."
#   SpatialRGB can show which genes are where, but not what the tissue looks
#   like, so there is no way to tell a glomerulus from a tubule lumen from an
#   infiltrate without leaving the app.
#
# THE MAPPING IS ONE LINE, AND THAT IS A MEASURED FACT, NOT AN ASSUMPTION
#   pxl_col/row_in_fullres index the MICROSCOPE H&E TIF itself: the hires PNG is
#   exactly fullres x tissue_hires_scalef (KTx_18 24145x12507 -> 6000x3108 at
#   0.24849865), while cytassist_image.tiff is 3200x3000 on every array and so
#   cannot be that frame.  The app's window is already absolute pxl*mpp microns
#   (spatial_rgb_data.py:135-141), so
#        tif_px = micron / mpp
#   with NO offset and NO rotation term.  Six scripts in this repo already do
#   exactly this by hand (nucleus_on_he_and_cytassist.py:233-236 is the idiom);
#   this module is meant to be the first shared one.
#   mpp is PER ARRAY -- 0.4102 (KTx_18) vs 0.5133 (KTx_17).  Never hardcode it.
#
# THE TRAP: THE CAPTURE SQUARE STICKS OUT OF THE SCAN
#   KTx_18's pxl_row minimum is -2368 and 164,816 of its 702,244 lattice bins
#   (23.5%) fall outside the TIF entirely; KTx_17 2.1%.  A crop therefore has to
#   clip AND SAY SO -- every crop returns the fraction of the window that was
#   really inside the scan, so the panel can mark off-scan area instead of
#   silently clamping it onto the edge pixels.
#
# TWO READ PATHS
#   exact  : the source TIFs are LZW+predictor-2, single IFD, NO pyramid, and
#            STRIPPED WITH RowsPerStrip = 1.  That is the best stripped case --
#            one independently decodable strip per image row -- so a row range
#            can be read without touching the rest of the file.  Measured on
#            KTx_18: 2000 rows in 1.02 s and byte-identical to a full decode,
#            against 5.83 s and 1.7 GB RSS for PIL's own crop (which decodes
#            everything).  Cost is ~0.46 ms/row and does not depend on width.
#   index  : a JPEG tile pyramid built once per array by build_he_index.py.
#            ~10-20 ms a crop.  Used when present, and the exact path is the
#            fallback, so the panel works before the index is built and the two
#            can be compared against each other (he_crop_selftest.py does).
#   Lossy JPEG is for DISPLAY ONLY.  Anything quantitative must use exact=True.
#
# INPUTS   HiRes_Images/<sample>_fullres.TIF  (a symlink -- resolve it, the
#            targets are irregularly named: KTx_14-Repeat -> KTx14-repeat.TIF)
#          Processed/<s>/outs/binned_outputs/square_008um/spatial/
#            scalefactors_json.json          (microns_per_pixel)
# OUTPUT   HiRes_Index/<sample>/{manifest.json, L<k>/<ty>_<tx>.jpg}
# Usage    imported by app.py;  python he_image.py  runs a self-check
# ==============================================================================

import io
import json
import os
import struct
import time

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None          # 302 Mpx trips Pillow's 89 Mpx bomb guard

# Where the data lives (Processed/, L0_CellType/, HiRes_*).  $SPATIAL_BASE if set,
# else the folder that contains SpatialRGB/ -- which is the project root here.
BASE = os.environ.get(
    "SPATIAL_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_DIR = f"{BASE}/HiRes_Index"
TILE = 512
N_LEVELS = 5                           # /1 /2 /4 /8 /16
JPEG_Q = 90
OFF_SCAN = 246                         # grey painted where the window left the scan


# ------------------------------------------------------------------ locating
def he_path(sample):
    """The microscope TIF for an array, via the symlink.

    NOT by reconstructing the filename: nucleus_on_he_and_cytassist.he_path()
    parses KTx_(\\d+) out of the sample name, which maps KTx_14-Repeat onto
    KTx14.TIF -- a different array.  The symlink in HiRes_Images is keyed by the
    exact sample name, so it is the only safe route.
    """
    link = f"{BASE}/HiRes_Images/{sample}_fullres.TIF"
    if not os.path.exists(link):                      # follows the link
        return None
    return os.path.realpath(link)


def mpp_of(sample, res_dir="square_008um"):
    """Microns per FULLRES pixel. Identical at 2/8/16 um -- the image is one image."""
    p = (f"{BASE}/Processed/{sample}/outs/binned_outputs/{res_dir}/spatial/"
         "scalefactors_json.json")
    return float(json.load(open(p))["microns_per_pixel"])


# -------------------------------------------------- the exact (strip) reader
class _Tiff:
    """Just enough TIFF to read a row range out of a stripped LZW file.

    Pillow cannot do this: Image.crop() calls load(), which decodes the whole
    image.  So the strips for the wanted rows are copied into a small in-memory
    TIFF that is handed back to Pillow -- libtiff then decodes only those rows.
    Verified byte-identical to a full decode (he_crop_selftest.py check 1).
    """

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            head = f.read(8)
            if head[:2] != b"II" or struct.unpack("<H", head[2:4])[0] != 42:
                raise ValueError(f"{path}: not a little-endian classic TIFF")
            f.seek(struct.unpack("<I", head[4:8])[0])
            n = struct.unpack("<H", f.read(2))[0]
            tags = {}
            for _ in range(n):
                tag, typ, cnt, val = struct.unpack("<HHII", f.read(12))
                tags[tag] = (typ, cnt, val)
            self.tags = tags
            g = lambda t: tags[t][2] if t in tags else None
            self.width = g(256)
            self.height = g(257)
            self.compression = g(259)
            self.photometric = g(262)
            self.samples = g(277) or 1
            self.rows_per_strip = g(278)
            self.predictor = g(317) or 1
            self.offsets = self._array(f, 273)
            self.counts = self._array(f, 279)

    def _array(self, f, tag):
        typ, cnt, val = self.tags[tag]
        fmt = {3: "<%dH" % cnt, 4: "<%dI" % cnt}[typ]
        size = cnt * (2 if typ == 3 else 4)
        if size <= 4:
            return [val]
        f.seek(val)
        return list(struct.unpack(fmt, f.read(size)))

    def supported(self):
        """Only the exact shape these 28 files have; anything else falls back."""
        return (self.compression == 5 and self.predictor == 2
                and self.rows_per_strip == 1 and self.samples == 3
                and self.photometric == 2
                and len(self.offsets) == self.height)

    def rows(self, r0, r1):
        """Decode image rows [r0, r1) at full width -> (h, W, 3) uint8."""
        r0 = max(0, min(int(r0), self.height))
        r1 = max(r0, min(int(r1), self.height))
        if r1 == r0:
            return np.zeros((0, self.width, 3), np.uint8)
        n = r1 - r0
        buf = bytearray(b"II*\x00" + struct.pack("<I", 0))
        offs = []
        with open(self.path, "rb") as f:
            for i in range(r0, r1):
                f.seek(self.offsets[i])
                offs.append(len(buf))
                buf += f.read(self.counts[i])
        bps_at = len(buf); buf += struct.pack("<3H", 8, 8, 8)
        off_at = len(buf); buf += struct.pack("<%dI" % n, *offs)
        cnt_at = len(buf); buf += struct.pack("<%dI" % n,
                                              *self.counts[r0:r1])
        ifd_at = len(buf)
        e = [(256, 4, 1, self.width), (257, 4, 1, n), (258, 3, 3, bps_at),
             (259, 3, 1, 5), (262, 3, 1, 2), (273, 4, n, off_at),
             (277, 3, 1, 3), (278, 4, 1, 1), (279, 4, n, cnt_at),
             (284, 3, 1, 1), (317, 3, 1, 2)]
        buf += struct.pack("<H", len(e))
        for tag, typ, cnt, val in e:
            # a SHORT that fits goes inline, in the low half of the value field
            if typ == 3 and cnt == 1:
                buf += struct.pack("<HHIHH", tag, typ, cnt, val, 0)
            else:
                buf += struct.pack("<HHII", tag, typ, cnt, val)
        buf += struct.pack("<I", 0)
        buf[4:8] = struct.pack("<I", ifd_at)
        return np.asarray(Image.open(io.BytesIO(bytes(buf))).convert("RGB"))


# ----------------------------------------------------------------- the image
class HEImage:
    """One array's H&E, croppable by a micron window in the app's frame."""

    def __init__(self, sample):
        self.sample = sample
        self.path = he_path(sample)
        self.mpp = mpp_of(sample)
        self.manifest = None
        self._tif = None
        m = f"{INDEX_DIR}/{sample}/manifest.json"
        if os.path.exists(m):
            self.manifest = json.load(open(m))
            self.width, self.height = self.manifest["width"], self.manifest["height"]
        elif self.path:
            t = self._tiff()
            self.width, self.height = t.width, t.height
        else:
            self.width = self.height = 0

    def _tiff(self):
        if self._tif is None:
            self._tif = _Tiff(self.path)
        return self._tif

    @property
    def indexed(self):
        return self.manifest is not None

    # ------------------------------------------------------------ the crop
    def crop(self, x_um, y_um, side_um, out_px=620, exact=False):
        """The window as an (out_px, out_px, 3) uint8 image, plus what it cost.

        Returns (rgb, info).  info["coverage"] is the fraction of the requested
        window that was inside the scan; the rest is painted OFF_SCAN grey and
        must be reported, never passed off as pale tissue.
        """
        t0 = time.time()
        if not self.path and not self.indexed:
            raise FileNotFoundError(f"no microscope TIF for {self.sample}")
        px0 = x_um / self.mpp
        py0 = y_um / self.mpp
        pside = side_um / self.mpp

        # how much of the window is actually on the scan
        ix0, iy0 = max(px0, 0.0), max(py0, 0.0)
        ix1 = min(px0 + pside, float(self.width))
        iy1 = min(py0 + pside, float(self.height))
        inside = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        cov = inside / (pside * pside) if pside > 0 else 0.0

        img = np.full((out_px, out_px, 3), OFF_SCAN, np.uint8)
        info = dict(coverage=round(cov, 4), source=None, level=None,
                    px_window=[round(px0, 1), round(py0, 1), round(pside, 1)],
                    mpp=self.mpp, um_per_out_px=round(side_um / out_px, 4))
        if cov <= 0:
            info["source"] = "off-scan"
            info["seconds"] = round(time.time() - t0, 3)
            return img, info

        use_index = self.indexed and not exact
        sub = (self._from_index(ix0, iy0, ix1, iy1, out_px, info) if use_index
               else self._from_tiff(ix0, iy0, ix1, iy1, out_px, info))

        # paste the on-scan part where it belongs inside the output square
        ox0 = int(round((ix0 - px0) / pside * out_px))
        oy0 = int(round((iy0 - py0) / pside * out_px))
        h, w = sub.shape[:2]
        h = min(h, out_px - oy0); w = min(w, out_px - ox0)
        if h > 0 and w > 0:
            img[oy0:oy0 + h, ox0:ox0 + w] = sub[:h, :w]
        info["seconds"] = round(time.time() - t0, 3)
        return img, info

    def _out_size(self, ix0, iy0, ix1, iy1, out_px, pside):
        """Output pixels the on-scan part should occupy (keeps it in register)."""
        return (max(1, int(round((ix1 - ix0) / pside * out_px))),
                max(1, int(round((iy1 - iy0) / pside * out_px))))

    def _from_tiff(self, ix0, iy0, ix1, iy1, out_px, info):
        pside = info["px_window"][2]
        w_out, h_out = self._out_size(ix0, iy0, ix1, iy1, out_px, pside)
        t = self._tiff()
        if not t.supported():                     # not the expected layout
            im = Image.open(self.path).convert("RGB").crop(
                (int(ix0), int(iy0), int(ix1), int(iy1)))
            info["source"] = "tiff-full-decode"
            return np.asarray(im.resize((w_out, h_out), Image.BILINEAR))
        rows = t.rows(int(iy0), int(np.ceil(iy1)))
        sub = rows[:, int(ix0):int(np.ceil(ix1))]
        info["source"] = "tiff-strips"
        info["rows_decoded"] = int(rows.shape[0])
        if sub.size == 0:
            return np.full((h_out, w_out, 3), OFF_SCAN, np.uint8)
        return np.asarray(Image.fromarray(sub).resize((w_out, h_out), Image.BILINEAR))

    def _from_index(self, ix0, iy0, ix1, iy1, out_px, info):
        pside = info["px_window"][2]
        w_out, h_out = self._out_size(ix0, iy0, ix1, iy1, out_px, pside)
        # the coarsest level that still has at least out_px across the window
        lv = int(np.floor(np.log2(max(pside / max(out_px, 1), 1.0))))
        lv = max(0, min(lv, self.manifest["levels"] - 1))
        s = 2 ** lv
        lw, lh = self.manifest["level_size"][lv]
        a0, b0 = ix0 / s, iy0 / s
        a1, b1 = min(ix1 / s, lw), min(iy1 / s, lh)
        tx0, ty0 = int(a0 // TILE), int(b0 // TILE)
        tx1, ty1 = int((np.ceil(a1) - 1) // TILE), int((np.ceil(b1) - 1) // TILE)
        canvas = Image.new("RGB", ((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE),
                           (OFF_SCAN, OFF_SCAN, OFF_SCAN))
        n = 0
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                p = f"{INDEX_DIR}/{self.sample}/L{lv}/{ty}_{tx}.jpg"
                if os.path.exists(p):
                    canvas.paste(Image.open(p), ((tx - tx0) * TILE, (ty - ty0) * TILE))
                    n += 1
        sub = canvas.crop((int(a0 - tx0 * TILE), int(b0 - ty0 * TILE),
                           int(np.ceil(a1)) - tx0 * TILE,
                           int(np.ceil(b1)) - ty0 * TILE))
        info["source"] = "index"
        info["level"] = lv
        info["tiles"] = n
        return np.asarray(sub.resize((w_out, h_out), Image.BILINEAR))

    # ------------------------------------------------- native-resolution crop
    def crop_native(self, x_um, y_um, side_um, max_px=6000):
        """The window at 1 output pixel = 1 scan pixel — the most that exists.

        Read from the ORIGINAL TIF, never the JPEG index, and NOT resampled, so
        these are the scanner's own pixels.  The panel shows ~835 px whatever the
        field; this is what the data actually holds, which for a 120 um field on
        KTx_18 is 293 px square and no more.  Above `max_px` the request is
        downsampled rather than refused, and the info says so -- a whole section
        at native would be 24145 x 12507 (906 MB).
        """
        if not self.path:
            raise FileNotFoundError(f"no microscope TIF for {self.sample}")
        t0 = time.time()
        px0, py0 = x_um / self.mpp, y_um / self.mpp
        pside = side_um / self.mpp
        # A SQUARE box of ceil(side) pixels from a common origin, THEN clamped.
        # Taking floor on the origin and ceil on the far edge independently made
        # the result 733 x 732 for a square window -- off by one in height only,
        # which is the sort of thing that quietly breaks a later re-crop.
        n = int(np.ceil(pside))
        ix0, iy0 = int(np.floor(px0)), int(np.floor(py0))
        ix1, iy1 = ix0 + n, iy0 + n
        ix0, iy0 = max(ix0, 0), max(iy0, 0)             # clamp: non-square only
        ix1, iy1 = min(ix1, self.width), min(iy1, self.height)   # at the edge
        if ix1 <= ix0 or iy1 <= iy0:
            raise ValueError("the window does not overlap the scan")
        t = self._tiff()
        if t.supported():
            img = t.rows(iy0, iy1)[:, ix0:ix1]
        else:
            img = np.asarray(Image.open(self.path).convert("RGB").crop(
                (ix0, iy0, ix1, iy1)))
        native = [int(img.shape[1]), int(img.shape[0])]
        capped = max(native) > max_px
        if capped:                       # too big to hand over whole
            k = max_px / max(native)
            img = np.asarray(Image.fromarray(img).resize(
                (max(1, int(native[0] * k)), max(1, int(native[1] * k))),
                Image.LANCZOS))
        inside = (ix1 - ix0) * (iy1 - iy0)
        info = dict(
            sample=self.sample, mpp=self.mpp,
            window_um=[round(float(x_um), 2), round(float(y_um), 2),
                       round(float(side_um), 2)],
            tif_pixel_box=[ix0, iy0, ix1, iy1],
            native_px=native, returned_px=[int(img.shape[1]), int(img.shape[0])],
            um_per_px=round(self.mpp if not capped else
                            side_um / max(img.shape[1], img.shape[0]), 4),
            downsampled=bool(capped),
            coverage=round(inside / (pside * pside), 4) if pside > 0 else 0.0,
            source="tiff-exact", seconds=round(time.time() - t0, 2))
        return img, info

    # -------------------------------------------------------- whole section
    def whole(self, out_px=620):
        """The entire scan, for the panel before any zoom box exists."""
        return self.crop(0.0, 0.0,
                         max(self.width, self.height) * self.mpp, out_px)


def available(sample):
    return he_path(sample) is not None


# ---------------------------------------------------------------- self-check
if __name__ == "__main__":
    import sys
    s = sys.argv[1] if len(sys.argv) > 1 else "KTx_18"
    h = HEImage(s)
    print(f"{s}: {h.width:,} x {h.height:,} px, mpp {h.mpp:.4f} "
          f"({h.width * h.mpp / 1000:.2f} x {h.height * h.mpp / 1000:.2f} mm), "
          f"indexed={h.indexed}")
    print(f"  file {h.path}")
    t = h._tiff()
    print(f"  compression={t.compression} predictor={t.predictor} "
          f"rows_per_strip={t.rows_per_strip} strips={len(t.offsets):,} "
          f"supported={t.supported()}")
    mid_x = h.width * h.mpp * 0.5
    mid_y = h.height * h.mpp * 0.5
    for side in (120.0, 320.0, 800.0):
        img, info = h.crop(mid_x, mid_y, side)
        print(f"  {side:6.0f} µm window -> {info['source']:16s} "
              f"{info['seconds']:5.2f}s  coverage {info['coverage']:.3f}  "
              f"rows={info.get('rows_decoded', '-')}  mean px {img.mean():.1f}")
    # a window deliberately off the top of the scan: coverage must drop
    img, info = h.crop(mid_x, -400.0, 320.0)
    print(f"  off the top edge  -> coverage {info['coverage']:.3f} "
          f"(must be < 1), source {info['source']}")
