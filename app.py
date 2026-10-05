#!/usr/bin/env python3
# ==============================================================================
# app.py — a marker browser for Visium HD: three expression gradients mixing over
#          the 8 um RCTD cell-type map, used as the ground.
#
# WHAT IT IS FOR (author, 2026-09-21)
#   Deck slide 12 is one fixed picture: three fixed marker panels, one fixed
#   window, one array.  Making the next one means editing a script.  This turns
#   that figure into a control panel -- pick an array, pick genes for red, green
#   and blue (several per colour, summed, which is what a "marker panel" is),
#   and drag a box to zoom.
#
# THE PICTURE (author, same day: "I meant the gradiant of red, green and blue
# depending on the expression.  the mix will be mix of gradient.  the RCTD
# celltype will be the black backround, grayish black if a specific celltype is
# chosen")
#   Each channel is a CONTINUOUS ramp from the ground to full hue at vmax, and the
#   three are mixed, so a bin carrying red and green reads as the mix of two
#   gradients.  Underneath, every in-tissue bin sits at a near-black; choosing one
#   or more RCTD cell types LIFTS those bins to a greyish black, silhouetting
#   where the type is without recolouring anything.
#
#   THE BLEND IS A MAXIMUM, NOT A SUM.  Measured, not assumed: summing the ground
#   into the hue turned a red glomerular tuft pink, while a maximum lets the
#   ground show only where no channel reaches it.  It also means that with no type
#   lifted and a linear ramp the picture is BYTE-IDENTICAL to what this app drew
#   before, and to deck slide 12 -- verified, not asserted.
#
#   THE RAMP IS SELECTABLE because these are small counts: a marker panel puts 1-3
#   UMI in most bins it touches and 10+ in a few, so on a linear ramp low enough
#   to show the 1s, the 10s saturate and the gradient flattens.  sqrt and log keep
#   the top of the range from eating the bottom.  The ramp in force is named next
#   to the picture and drawn in the legend.
#
# WHAT NONE OF IT CHANGES
#     - brightness is still summed marker UMI per bin over a shared vmax;
#     - the co-location counts are the same ones celltype_rgb_overlay.py writes;
#     - A MIXED COLOUR MEANS TWO CHANNELS SHARE A BIN.  At 8 um a bin is about
#       one nucleus wide (6.8-7.1 um here) but is not one cell, so a mixed colour
#       is co-location at the bin size -- never co-expression in a cell.
#     - a channel can be dark because its cells are elsewhere OR because its
#       genes are rare; the per-channel UMI total is always shown.
#     - the same applies to the ground: a gene positive in a bin RCTD called T_NK
#       is co-location at 8 um, not transcription by that T cell.
#
# PERFORMANCE: see spatial_rgb_data.py -- gene-major transpose once per array
#   (~1-2 s), then ~0.035 ms a gene; compose ~0.04 s; the label join ~0.2 s once.
#
# Usage:
#   ../spatial-venv/bin/python -m shiny run --port 8765 SpatialRGB/app.py
# ==============================================================================

import functools
import io as _io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from shiny import App, reactive, render, ui

BASE = "/home/abb2013/Documents/Spatial"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spatial_rgb_data as D                                   # noqa: E402
import he_image as HE                                          # noqa: E402
from celltype_marker_maps import PANELS as DECK_PANELS         # noqa: E402

# The crop resolution has to match the panel it feeds, or the extra screen
# pixels only magnify the ones we fetched.  HE_H is the card height; Shiny sizes
# the figure from the client box, so this is what the viewer actually gets.
HE_H = "900px"
HE_PX = 1200                     # crop pixels requested; >= the displayed square


@functools.lru_cache(maxsize=8)
def he_of(sample):
    """One HEImage per array. Cheap to hold: it stores a manifest, not pixels."""
    return HE.HEImage(sample)

CHAN = [("r", "RED", "#FF4040"), ("g", "GREEN", "#3BD23B"), ("b", "BLUE", "#5B8CFF")]
PRESETS = {k.split("  ")[0]: list(v) for k, v in DECK_PANELS.items()}
NUCLEUS_UM = (6.8, 7.1)
MODES = {"expr": "Expression gradients over cell types",
         "ct": "Cell types alone"}

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("sample", "Array", choices=D.samples(), selected="KTx_18"),
        ui.input_select("res", "Bin size", choices=list(D.RES), selected="8 µm"),
        ui.input_radio_buttons("mode", "View", choices=MODES, selected="expr"),
        ui.hr(),

        ui.panel_conditional(
            "input.mode !== 'ct'",
            *[x for cid, lab, col in CHAN for x in (
                ui.tags.div(ui.tags.b(lab, style=f"color:{col}"),
                            style="margin-top:2px"),
                ui.input_selectize(f"sel_{cid}", None, choices=[], multiple=True,
                                   options={"placeholder": "type a gene name…"}),
                ui.input_select(f"preset_{cid}", None,
                                choices=["— deck panel —"] + sorted(PRESETS)),
            )],
            ui.input_numeric("vmax", "Full hue at (UMI per bin)", 5.0,
                             min=0.5, max=200, step=0.5),
            ui.input_radio_buttons("ramp", "Gradient", choices=list(D.RAMPS),
                                   selected="sqrt", inline=True)),

        ui.hr(),
        # Author, 2026-10-05: "I can't see the cell at this resolution. Can't we
        # make more zoom?"  The panel is already as large as a square can be in
        # that row -- what was missing is a way to REACH a small field, because
        # the section view puts ~6,800 um into 620 px, so a 300 um drag box is
        # 27 px on screen and cannot be placed deliberately.  This zooms the H&E
        # about the centre of the selection without disturbing the other panels.
        ui.input_select("he_fov", ui.tags.b("H&E zoom (field of view)"),
                        choices={"match": "match the selection",
                                 "1000": "1000 µm", "600": "600 µm",
                                 "400": "400 µm", "300": "300 µm",
                                 "200": "200 µm", "120": "120 µm",
                                 "60": "60 µm", "30": "30 µm"},
                        selected="match"),
        ui.output_ui("he_note"),
        ui.download_button("dl_he", "Save this field at full resolution",
                           class_="btn-sm"),
        ui.output_ui("he_where"),
        ui.hr(),
        ui.output_ui("ct_controls"),
        ui.input_action_button("reset", "Reset zoom", class_="btn-sm"),
        ui.output_ui("status"),
        width=345,
    ),
    ui.layout_columns(
        ui.card(ui.card_header("Whole array — drag a box to zoom"),
                ui.output_plot("section", brush=ui.brush_opts(
                    direction="xy", reset_on_new=False, fill="#ffffff22",
                    stroke="#ffffff"), height="620px")),
        ui.card(ui.card_header("Selection"),
                ui.output_plot("zoom", height="620px")),
        col_widths=[6, 6],
    ),
    # The H&E gets a row of its own (author, 2026-10-05: "the H&E image is too
    # small. I need to have a cell resolution").  Sharing the row three ways
    # capped it at a third of the width; on its own row the square is limited by
    # HE_H instead, which is 1.45x more screen pixels for the same field.
    ui.card(ui.card_header("Same window — microscope H&E"),
            ui.output_plot("he", height=HE_H)),
    ui.card(ui.output_ui("stats")),
    title="Visium HD — expression gradients over the 8 µm RCTD map",
    fillable=False,
)


def server(input, output, session):

    @reactive.calc
    def arr():
        return D.load(input.sample(), D.RES[input.res()])

    def labels_ok():
        """8 um has the labels; 2 um inherits them from its parent. 16 um cannot."""
        return D.RES[input.res()] in (D.L0_RES_DIR, "square_002um")

    def inherited():
        return D.RES[input.res()] == "square_002um"

    @reactive.effect
    def _fill_gene_lists():
        g = list(arr().genes)
        with reactive.isolate():                 # keep the picks, do not depend on them
            for cid, _, _ in CHAN:
                ui.update_selectize(f"sel_{cid}", choices=g,
                                    selected=list(input[f"sel_{cid}"]() or ()),
                                    server=True)

    for cid, _, _ in CHAN:
        def _mk(cid=cid):
            @reactive.effect
            @reactive.event(input[f"preset_{cid}"])
            def _apply():
                p = input[f"preset_{cid}"]()
                if p in PRESETS:
                    ui.update_selectize(f"sel_{cid}", choices=list(arr().genes),
                                        selected=PRESETS[p], server=True)
        _mk()

    # Built server-side so the 8 um restriction is stated where it bites, rather
    # than offering a list of types that cannot be honoured at this bin size.
    @render.ui
    def ct_controls():
        if not labels_ok():
            return ui.tags.div(
                ui.tags.b("Cell types: not at 16 µm."),
                ui.tags.div("The Layer 0 RCTD run labels 8 µm bins. A 16 µm bin "
                            "CONTAINS four of them, which can disagree, so there is no "
                            "one label to carry up. 8 µm and 2 µm both work — a 2 µm "
                            "bin has exactly one 8 µm parent.",
                            style="color:#7A4A00"),
                class_="small", style="margin:4px 0")
        note = None
        if inherited():
            n = arr().nesting_check()
            note = ui.tags.div(
                ui.tags.b("2 µm bins, 8 µm shading. "),
                f"Each 2 µm bin takes the label of the 8 µm bin that contains it, so the "
                f"ground is in 8 µm blocks while the colour is per 2 µm bin. The grids "
                f"nest: {n['n_parent_full']:,} parents hold all 16 children and the worst "
                f"centre offset is {n['max_offset_um']:.4f} µm (measured on load).",
                class_="small", style="color:#7A4A00;margin:4px 0")
        return ui.tags.div(
            note if note is not None else "",
            ui.input_selectize("ct", ui.tags.b("Cell types (8 µm RCTD) — the ground"),
                               choices=D.CELLTYPE_ORDER, multiple=True,
                               options={"placeholder": "pick one or more…"}),
            ui.panel_conditional(
                "input.mode !== 'ct'",
                ui.input_slider("ground", "How far the chosen type lifts out of black",
                                min=0.05, max=0.40, value=0.22, step=0.01),
                ui.output_ui("ground_cap"),
                ui.input_checkbox("ct_mask", "Also hide genes outside these types",
                                  False)),
        )

    @reactive.calc
    def picked():
        return [list(input[f"sel_{c}"]() or ()) for c, _, _ in CHAN]

    @reactive.calc
    def ct_picked():
        if not labels_ok():
            return []
        try:
            return list(input.ct() or ())
        except Exception:
            return []

    def _opt(name, default):
        """A control that lives inside the dynamic sidebar may not exist yet."""
        try:
            v = input[name]()
            return default if v is None else v
        except Exception:
            return default

    @reactive.calc
    def ground_level():
        """What the ground is actually drawn at, and whether the slider was clamped.

        The blend is max(hue, ground), so a ground above the faintest representable
        signal paints positive bins over — and only inside the lifted type, which
        is precisely where density is being judged.  So the request is CLAMPED to
        safe_ground() and the clamp is shown, rather than left to be discovered as
        "the shading changes how much expression there seems to be".
        """
        want = float(_opt("ground", 0.22))
        cap = D.safe_ground(input.vmax(), input.ramp())
        return min(want, cap), want, cap

    @reactive.calc
    def ground():
        """Per-bin grey: the RCTD map as the ground under the gradients."""
        if not ct_picked():
            return None
        return D.celltype_background(arr(), ct_picked(), sel=ground_level()[0])

    @render.ui
    def ground_cap():
        if not ct_picked():
            return ui.tags.div()
        lvl, want, cap = ground_level()
        if lvl < want - 1e-9:
            return ui.tags.div(
                ui.tags.small(f"capped at {cap:.2f} — above that the ground paints over "
                              f"1-UMI bins inside the lifted type only, which fakes a "
                              f"change in density"), style="color:#9E2A2B")
        return ui.tags.div(ui.tags.small(f"ceiling {cap:.2f} at this ramp and vmax"),
                           style="color:#666")

    @reactive.calc
    def mask():
        if input.mode() == "ct" or not ct_picked() or not _opt("ct_mask", False):
            return None
        return D.celltype_mask(arr(), ct_picked())

    # There is no supported way to clear a brush from the server, so the window is
    # held here: "Reset zoom" sets it to None and the next drag sets it again.  A
    # SQUARE is taken so a micron stays a micron in both panels.
    win_rv = reactive.value(None)

    @reactive.effect
    @reactive.event(input.section_brush)
    def _take_brush():
        b = input.section_brush()
        if not b:
            return
        side = max(b["xmax"] - b["xmin"], b["ymax"] - b["ymin"])
        win_rv.set((float(b["xmin"]), float(min(b["ymin"], b["ymax"])), float(side)))

    @reactive.effect
    @reactive.event(input.reset, input.sample, input.res)
    def _clear():
        win_rv.set(None)

    # A 2 um bin holds a fraction of an 8 um bin's counts, so a vmax chosen at 8 um
    # renders 2 um almost black.  Rather than hard-code a number per bin size, take
    # the 95th percentile of the POSITIVE bins of whatever is currently picked --
    # computed from the data, printed by the footer, and overridable by hand.
    @reactive.effect
    @reactive.event(input.res)
    def _rescale():
        sets = [x for grp in picked() for x in grp]
        if not sets:
            return
        v, _, _ = arr().channel(sets)
        p95 = D.auto_vmax(v, pct=95.0)
        if p95:
            ui.update_numeric("vmax", value=round(max(p95, 0.5), 1))

    def window():
        return win_rv()

    @reactive.calc
    def he_field():
        """(x0, y0, side) for the H&E panel: the selection, or a zoom inside it.

        Centred on the selection so that zooming in does not wander off the
        thing you picked.  With no selection it centres on the whole scan.
        """
        try:
            him = he_of(input.sample())
        except Exception:                                  # noqa: BLE001
            return None
        w = window()
        if w is None:
            side_full = max(him.width, him.height) * him.mpp
            cx = him.width * him.mpp / 2.0
            cy = him.height * him.mpp / 2.0
            sel = side_full
        else:
            cx, cy = w[0] + w[2] / 2.0, w[1] + w[2] / 2.0
            sel = w[2]
        f = _opt("he_fov", "match")
        side = sel if f == "match" else float(f)
        side = max(min(side, sel if f == "match" else side), 5.0)
        return (cx - side / 2.0, cy - side / 2.0, side)

    def keep_of(a, win):
        """Per-bin bool for what is inside the window (None = the whole array)."""
        if not win:
            return None
        x0, y0, side = win
        return ((a.bx >= x0) & (a.bx < x0 + side) &
                (a.by >= y0) & (a.by < y0 + side))

    def _msg(ax, t):
        ax.text(.5, .5, t, ha="center", va="center", color="#888",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return None

    def scalebar(ax, ext, col="white"):
        side = ext[1] - ext[0]
        bar = side * 0.2
        ax.plot([ext[1] - bar - side * .05, ext[1] - side * .05],
                [ext[2] - side * .05] * 2, "-", color=col, lw=2.4)
        ax.text(ext[1] - bar / 2 - side * .05, ext[2] - side * .075,
                f"{bar:,.0f} µm", ha="center", color=col, fontsize=8.5)

    def draw(fig, ax, win, title):
        a = arr()
        if input.mode() == "ct":
            if not labels_ok():
                return _msg(ax, "cell types need 8 µm bins")
            if not ct_picked():
                return _msg(ax, "pick one or more cell types")
            rgb, ext, n = D.compose_celltypes(a, ct_picked(), win)
            ax.imshow(rgb, extent=ext, interpolation="nearest")
            ax.legend(handles=[Patch(facecolor=D.CELLTYPE_COLOURS[t], label=t)
                               for t in ct_picked()] +
                              [Patch(facecolor=D.OTHER_COL, label="other type"),
                               Patch(facecolor=D.NOT_ANN_COL, label="not annotated")],
                      loc="upper right", fontsize=7, framealpha=.9, borderpad=.4)
            scalebar(ax, ext, col="#333")
            sub = f"{len(ct_picked())} type(s) · 8 µm RCTD (Layer 0)"
        else:
            if not any(picked()):
                return _msg(ax, "pick at least one gene")
            rgb, ext, n, vecs, used, miss = D.compose(
                a, picked(), input.vmax(), win, mask=mask(),
                scale=input.ramp(), bg=ground())
            ax.imshow(rgb, extent=ext, interpolation="nearest")
            scalebar(ax, ext)
            sub = f"{input.ramp()} ramp · full hue at {input.vmax():g} UMI/bin"
            if ct_picked():
                lvl, want, cap = ground_level()
                sub += (" · 8 µm ground: " if inherited() else " · ground: ") \
                       + "/".join(ct_picked()) + f" at {lvl:.2f}"
                if lvl < want - 1e-9:
                    sub += f" (capped from {want:.2f} so it hides nothing)"
                if mask() is not None:
                    sub += " (genes hidden outside)"
        side = ext[1] - ext[0]
        ax.set_title(f"{title} · {side:,.0f} µm across · {a.bin_um:.0f} µm bins · "
                     f"{n:,} bins\n{sub}", fontsize=9, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        return True

    @render.plot
    def section():
        fig, ax = plt.subplots(figsize=(6.6, 6.6))
        fig.subplots_adjust(0.01, 0.01, 0.99, 0.90)
        draw(fig, ax, None, f"{input.sample()} — whole capture square")
        return fig

    @render.plot
    def zoom():
        fig, ax = plt.subplots(figsize=(6.6, 6.6))
        fig.subplots_adjust(0.01, 0.01, 0.99, 0.90)
        w = window()
        if w is None:
            _msg(ax, "drag a box on the left")
        else:
            draw(fig, ax, w, "selection")
            f = he_field()
            if f is not None and f[2] < w[2] * 0.98:
                ax.add_patch(Rectangle((f[0], f[1]), f[2], f[2], fill=False,
                                       ec="#FFD400", lw=1.8, ls="--"))
                ax.text(f[0] + f[2] / 2, f[1] - w[2] * 0.012,
                        f"H&E {f[2]:,.0f} µm", ha="center", va="bottom",
                        color="#FFD400", fontsize=8.5)
        return fig

    # The H&E panel reads the SAME win_rv as the zoom panel, so the two cannot
    # drift apart, and draws on the same extent so a micron is a micron in all
    # three.  pxl_*_in_fullres index the H&E TIF itself, so the window needs no
    # transform beyond / mpp -- see he_image.py's banner for the evidence.
    @render.plot
    def he():
        fig, ax = plt.subplots(figsize=(6.6, 6.6))
        fig.subplots_adjust(0.01, 0.01, 0.99, 0.90)
        s = input.sample()
        try:
            him = he_of(s)
        except Exception as e:                            # noqa: BLE001
            return _msg(ax, f"H&E unavailable: {e}") or fig
        if not him.path and not him.indexed:
            return _msg(ax, f"no microscope H&E for {s}") or fig

        f = he_field()
        if f is None:
            return _msg(ax, "H&E unavailable") or fig
        x0, y0, side = f
        w = window()
        if w is None:
            title = f"{s} — whole scan"
        elif side < w[2] * 0.98:
            title = f"selection, zoomed to {side:,.0f} µm"
        else:
            title = "selection — H&E"
        try:
            img, info = him.crop(x0, y0, side, HE_PX)
        except Exception as e:                            # noqa: BLE001
            return _msg(ax, f"crop failed: {e}") or fig

        ax.imshow(img, extent=[x0, x0 + side, y0 + side, y0],
                  interpolation="nearest")
        scalebar(ax, [x0, x0 + side, y0 + side, y0], col="#222")
        src = {"index": f"index L{info['level']}", "tiff-strips": "TIF (exact)",
               "tiff-full-decode": "TIF (full decode)",
               "off-scan": "outside the scan"}.get(info["source"], info["source"])
        sub = (f"{src} · {info['seconds']:.2f}s · "
               f"{info['um_per_out_px']:.2f} µm per screen pixel")
        if info["coverage"] < 0.999:
            sub += f" · {100 * (1 - info['coverage']):.0f}% outside the scan (grey)"
        # KTx_1 only: a block holding 24.9% of the sample's UMI sits where the
        # H&E is blank slide -- tissue moved between the microscope and CytAssist
        # exposures, so blank H&E under real counts is the DATA, not a bad crop.
        if s == "KTx_1":
            sub += "\nKTx_1: tissue moved between exposures — a block with 25% of "
            sub += "the UMI has no H&E under it. Blank here is real, not misaligned."
        ax.set_title(f"{title} · {side:,.0f} µm across\n{sub}",
                     fontsize=9, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    # Measured: on a full-width 900 px row the square the viewer gets is ~835 px
    # (checked through the websocket with an 1850x900 client box).  Used here so
    # the nucleus figure is what you see, not what we cropped.
    HE_SCREEN_PX = 835

    @render.ui
    def he_note():
        try:
            him = he_of(input.sample())
            f = he_field()
        except Exception:                                  # noqa: BLE001
            return ui.tags.div()
        if f is None:
            return ui.tags.div()
        side = f[2]
        on_screen = side / HE_SCREEN_PX
        nuc = NUCLEUS_UM[0] / on_screen
        bits = [f"{side:,.0f} µm field · a {NUCLEUS_UM[0]}–{NUCLEUS_UM[1]} µm "
                f"nucleus ≈ {nuc:.0f} px on screen"]
        style = "color:#666"
        if on_screen < him.mpp:
            # Past the scan's own sampling: the picture gets bigger but carries no
            # new detail.  Said plainly, because a soft image here would otherwise
            # read as out-of-focus tissue or a broken crop.
            bits.append(f"past the scan's native {him.mpp:.2f} µm/px — enlarged "
                        f"{him.mpp / on_screen:.1f}×, so it is soft by the limit of "
                        f"the image, not by focus")
            style = "color:#7A4A00"
        return ui.tags.div(*[ui.tags.div(ui.tags.small(b)) for b in bits],
                           style=style + ";margin-top:4px")

    @render.ui
    def he_where():
        """The exact window, in both frames, so it can be quoted or re-cropped."""
        try:
            him = he_of(input.sample())
            f = he_field()
        except Exception:                                  # noqa: BLE001
            return ui.tags.div()
        if f is None:
            return ui.tags.div()
        x0, y0, side = f
        px = [int(x0 / him.mpp), int(y0 / him.mpp),
              int((x0 + side) / him.mpp), int((y0 + side) / him.mpp)]
        n = int(round(side / him.mpp))
        return ui.tags.div(
            ui.tags.div(ui.tags.small(ui.tags.b("exact location "), "(app µm, the "
                        "frame every panel uses): ",
                        ui.tags.code(f"x {x0:,.1f}  y {y0:,.1f}  side {side:,.1f}"))),
            ui.tags.div(ui.tags.small("same box in ", ui.tags.code(him.sample),
                        " fullres pixels: ", ui.tags.code(
                            f"[{px[0]}, {px[1]}, {px[2]}, {px[3]}]"))),
            ui.tags.div(ui.tags.small(f"full resolution here is {n} × {n} px "
                                      f"at {him.mpp:.4f} µm/px")),
            style="color:#666;margin-top:6px")

    def _he_name():
        """Readable microns PLUS the exact integer pixel box.

        The rounded microns alone do NOT reproduce the crop -- re-cropping from
        "x4802" when the window was x4802.4 shifts the box and changed the size
        by a pixel in testing.  The pixel box is exact, so the file can always be
        cut again from the TIF.
        """
        f = he_field()
        him = he_of(input.sample())
        b = [int(np.floor(f[0] / him.mpp)), int(np.floor(f[1] / him.mpp))]
        n = int(np.ceil(f[2] / him.mpp))
        return (f"{him.sample}_x{f[0]:.0f}_y{f[1]:.0f}_{f[2]:.0f}um"
                f"_px{b[0]}-{b[1]}-{b[0] + n}-{b[1] + n}"
                f"_{him.mpp:.4f}umpx.png")

    @render.download(filename=_he_name)
    def dl_he():
        """The field at 1 pixel = 1 scan pixel, from the ORIGINAL TIF.

        Not the JPEG index and not the 835 px the panel draws -- this is every
        pixel the scanner recorded for that window, which is the most that
        exists.  The filename carries the window and the micron scale so the
        crop can be placed again later.
        """
        him = he_of(input.sample())
        f = he_field()
        img, info = him.crop_native(f[0], f[1], f[2])
        buf = _io.BytesIO()
        Image.fromarray(img).save(buf, "PNG")
        yield buf.getvalue()

    @render.ui
    def status():
        a = arr()
        bits = [f"{a.n_bins:,} bins · {len(a.genes):,} genes · loaded in {a.load_s}s"]
        if labels_ok():
            try:
                _, hit = a.labels()
                bits.append(f"{hit:,} bins have an RCTD label "
                            f"({100 * hit / a.n_bins:.0f}% of in-tissue)"
                            + (" — inherited from the 8 µm parent" if inherited() else ""))
            except Exception as e:                       # noqa: BLE001
                bits.append(f"labels unavailable: {e}")
        return ui.tags.div(*[ui.tags.div(ui.tags.small(b)) for b in bits],
                           style="color:#666;margin-top:8px")

    # ---- the legend is GENERATED from the same ramp the picture used, so it
    # ---- cannot drift from it: each stop is the colour of a bin at that UMI.
    def ramp_swatch(idx, vmax, scale, grey):
        stops = []
        for k in range(21):
            f = k / 20
            lvl = max(float(D._ramp(np.array(f), scale)), grey)
            c = [int(round(255 * grey))] * 3
            c[idx] = int(round(255 * lvl))
            stops.append(f"rgb({c[0]},{c[1]},{c[2]}) {100 * f:.0f}%")
        return "linear-gradient(to right, " + ", ".join(stops) + ")"

    @render.ui
    def stats():
        a, w, m = arr(), window(), mask()
        scope = (f"the selection ({w[2]:,.0f} µm across)" if w
                 else "the whole capture square")
        blocks, vecs = [], None

        if any(picked()):
            _, _, _, vecs, used, missing = D.compose(
                a, picked(), input.vmax(), w, mask=m, scale=input.ramp(), bg=ground())
            st = D.channel_stats(a, vecs, w)
            grey = _opt("ground", 0.22) if ct_picked() else 0.085
            rows = []
            for i, ((cid, lab, col), names, s_, miss) in enumerate(
                    zip(CHAN, used, st["per_channel"], missing)):
                rows.append(ui.tags.tr(
                    ui.tags.td(ui.tags.b(lab, style=f"color:{col}")),
                    ui.tags.td(ui.tags.div(
                        style=f"width:110px;height:13px;border-radius:2px;"
                              f"background:{ramp_swatch(i, input.vmax(), input.ramp(), grey)}")),
                    ui.tags.td(", ".join(names) or "—"),
                    ui.tags.td(f"{s_['umi']:,.0f}"),
                    ui.tags.td(f"{s_['bins']:,} ({s_['pct']}%)"),
                    ui.tags.td(f"{s_['median']:.0f} / {s_['p90']:.0f} / {s_['mx']:.0f}"),
                    ui.tags.td(ui.tags.span(", ".join(miss), style="color:#9E2A2B")
                               if miss else "")))
            blocks.append(ui.tags.table(
                ui.tags.thead(ui.tags.tr(*[ui.tags.th(h) for h in
                              ("channel", f"0 → {input.vmax():g} UMI", "genes", "UMI",
                               "bins positive", "UMI where + : med / p90 / max",
                               "not on this panel")])),
                ui.tags.tbody(*rows), class_="table table-sm"))
            blocks.append(ui.tags.p(
                ui.tags.b("Where the channels overlap — "),
                f"of {st['bins_any']:,} bins in {scope} carrying any channel, "
                f"{st['exactly_1']:,} carry one, {st['exactly_2']:,} carry two and "
                f"{st['all_3']:,} carry all three. "
                f"By pair: R+G {st['pairs']['R+G']:,}, R+B {st['pairs']['R+B']:,}, "
                f"G+B {st['pairs']['G+B']:,}."))

        if labels_ok() and ct_picked():
            cs = D.celltype_stats(a, ct_picked(), w, vecs)
            hdr = ["cell type", "bins", "% of annotated", "% of all bins in view"]
            if vecs is not None:
                hdr += [f"also {l}+" for _, l, _ in CHAN]
            trs = []
            for r in cs["rows"]:
                tds = [ui.tags.td(ui.tags.span("■ ", style=f"color:{r['colour']}"),
                                  ui.tags.b(r["type"])),
                       ui.tags.td(f"{r['bins']:,}"),
                       ui.tags.td(f"{r['pct_ann']}%"),
                       ui.tags.td(f"{r['pct_view']}%")]
                for o in (r["overlap"] if vecs is not None else []):
                    pc = f" ({100 * o / r['bins']:.0f}%)" if r["bins"] else ""
                    tds.append(ui.tags.td(f"{o:,}{pc}"))
                trs.append(ui.tags.tr(*tds))
            blocks.append(ui.tags.table(
                ui.tags.thead(ui.tags.tr(*[ui.tags.th(h) for h in hdr])),
                ui.tags.tbody(*trs), class_="table table-sm"))
            blocks.append(ui.tags.p(
                ui.tags.b("What the denominators are — "),
                f"of {cs['bins_in_view']:,} in-tissue bins in {scope}, "
                f"{cs['annotated']:,} ({cs['pct_annotated']}%) have an RCTD row and "
                f"{cs['not_annotated']:,} have none. "
                f"{cs['unknown']:,} of the annotated ones are Unknown. "
                "“Not annotated” means the bin was never offered to RCTD (the run "
                "kept counts_MIN 10), NOT that it is empty; “Unknown” is a call RCTD "
                "made — spot_class reject or doublet_uncertain."
                + (" These are 2 µm bins counted under their 8 µm parent's label, so "
                   "each labelled parent contributes 16 of them and the percentages "
                   "match the 8 µm ones by construction — they are not 16 times more "
                   "evidence." if inherited() else ""),
                style="color:#7A4A00"))

        if not blocks:
            return ui.tags.em("Pick genes for at least one channel, or one or more "
                              "cell types. Several genes in one channel are summed — "
                              "that is what a marker panel is.")

        if input.mode() == "ct":
            foot = ("Cell types: the July Layer 0 RCTD run (L0_celltype_8um.R), doublet "
                    "mode against the GSE183276 atlas, one call per 8 µm bin. B_PL is "
                    "the atlas's combined B + plasma class.")
        else:
            foot = (f"Each channel ramps {input.ramp()} from the ground to full hue at "
                    f"{input.vmax():g} UMI per bin, and the three are mixed by taking the "
                    f"brighter of hue and ground per channel — with no type lifted and a "
                    f"linear ramp that is byte-identical to deck slide 12.")
            if ct_picked():
                lvl, want, cap = ground_level()
                foot += (" The ground is the 8 µm RCTD map: every in-tissue bin sits near "
                         "black and " + "/".join(ct_picked()) + f" lifts to {lvl:.2f}")
                foot += (", in 8 µm blocks under 2 µm colour." if inherited() else ".")
                nhid, npos = ((0, 0) if vecs is None else
                              D.hidden_by_ground(vecs, input.vmax(), input.ramp(),
                                                 want, keep_of(a, w)))
                if vecs is None:
                    pass
                elif lvl < want - 1e-9:
                    blocks.append(ui.tags.p(
                        ui.tags.b("The ground was capped, and here is why it matters — "),
                        f"you asked for {want:.2f}; at that level the blend would have "
                        f"painted over {nhid:,} of the {npos:,} positive bins in view "
                        f"({100 * nhid / max(npos, 1):.0f}%), and ONLY inside the lifted "
                        f"type — so the shading would have appeared to thin out the "
                        f"expression it sits under. It is drawn at {lvl:.2f} instead, the "
                        f"highest level a 1-UMI bin still clears on the "
                        f"{input.ramp()} ramp at vmax {input.vmax():g}. Raise vmax or "
                        f"switch to sqrt/log to allow a heavier ground.",
                        style="color:#9E2A2B"))
                else:
                    blocks.append(ui.tags.p(
                        ui.tags.b("The ground hides nothing here — "),
                        f"every one of the {npos:,} positive bins in view clears "
                        f"{lvl:.2f} on the {input.ramp()} ramp. It still LOWERS CONTRAST "
                        f"though: a 1-UMI red bin reads 1.57:1 against black and 1.18:1 "
                        f"against a 0.22 grey, so expression looks fainter over a lifted "
                        f"type even when every bin is still drawn. The counts in the "
                        f"table are unaffected — read density there, not off the picture.",
                        style="color:#7A4A00"))
        blocks.append(ui.tags.p(
            ui.tags.b(f"Read a mixed colour as co-location at {a.bin_um:.0f} µm, not as a "
                      "cell. "),
            f"A bin is {a.bin_um:.0f} µm and a nucleus here is {NUCLEUS_UM[0]}–"
            f"{NUCLEUS_UM[1]} µm, so orange means the red and green sets both put UMI in "
            "the same square — not that one cell expressed both. A dark channel may mean "
            "its cells are elsewhere OR that its genes are rare: check the UMI column "
            "before reading absence.", style="color:#7A4A00"))
        blocks.append(ui.tags.small(
            foot + " SpatialRGB/app.py + spatial_rgb_data.py", style="color:#666"))
        return ui.tags.div(*blocks)


# The app is wrapped so that the page, the WebSocket and the download endpoint
# are all behind one password; see auth.py for why a login page inside the app
# would only guard the first of the three.  protect() raises if no password is
# configured, so there is no way to start this serving openly by accident.
import auth as AUTH                                              # noqa: E402

app = AUTH.protect(App(app_ui, server))
