#!/usr/bin/env python3
# ==============================================================================
# app.py — a marker browser for Visium HD: genes on colour channels, on a
#          gradient, or against the 8 um RCTD cell-type calls.
#
# WHAT IT IS FOR (author, 2026-09-21)
#   Deck slide 12 is one fixed picture: three fixed marker panels, one fixed
#   window, one array.  Making the next one means editing a script.  This turns
#   that figure into a control panel -- pick an array, pick genes for red, green
#   and blue (several per colour, summed, which is what a "marker panel" is),
#   and drag a box to zoom.
#
# THE THREE VIEWS, and why there are three rather than one
#   RGB       three gene sets, additive.  Answers WHICH SET IS HERE.  It cannot
#             show how much: one channel ramps one hue, and three of them mix.
#   GRADIENT  one gene set on a perceptually-uniform colourmap.  Answers HOW
#             MUCH, and is the only view that can carry a COLOURBAR, so the
#             numbers are read off the picture instead of guessed from
#             brightness.  Its ceiling is always printed -- a gradient with an
#             unstated vmax cannot be compared with any other figure.
#   CELL TYPE the July Layer 0 RCTD call per 8 um bin, for the types you pick.
#             Also usable as a MASK over either gene view, which is the question
#             the layer is really for: where does this gene sit RELATIVE TO the
#             cells RCTD named?
#
# WHAT NONE OF THEM CHANGE
#   The arithmetic and the caveats are the deck's, not new ones:
#     - brightness is summed marker UMI per bin over a shared vmax, linear;
#     - the co-location counts are the same ones celltype_rgb_overlay.py writes;
#     - A MIXED COLOUR MEANS TWO CHANNELS SHARE A BIN.  At 8 um a bin is about
#       one nucleus wide (6.8-7.1 um here) but is not one cell, so a mixed colour
#       is co-location at the bin size -- never co-expression in a cell.  The app
#       prints the bin size next to every picture for that reason.
#     - a channel can be dark because its cells are elsewhere OR because its
#       genes are rare; the per-channel UMI total is always shown so the two are
#       distinguishable.
#     - THE SAME APPLIES TO A CELL-TYPE OVERLAP: a gene positive in a bin RCTD
#       called T_NK is co-location at 8 um, not that the T cell expressed it.
#
# PERFORMANCE: see spatial_rgb_data.py -- the matrix is transposed to gene-major
#   once per array (~1-2 s) and then a gene is ~0.035 ms, so the picture keeps up
#   with the controls.  Whole-section compose is ~0.04 s at 8 um; the label join
#   is ~0.2 s once per array and the cell-type map ~0.01 s.
#
# Usage:
#   ../spatial-venv/bin/python -m shiny run --port 8765 SpatialRGB/app.py
#   then open http://127.0.0.1:8765
# ==============================================================================

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from shiny import App, reactive, render, ui

BASE = "/home/abb2013/Documents/Spatial"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spatial_rgb_data as D                                   # noqa: E402
from celltype_marker_maps import PANELS as DECK_PANELS         # noqa: E402

CHAN = [("r", "RED", "#FF4040"), ("g", "GREEN", "#3BD23B"), ("b", "BLUE", "#5B8CFF")]
PRESETS = {k.split("  ")[0]: list(v) for k, v in DECK_PANELS.items()}
NUCLEUS_UM = (6.8, 7.1)

MODES = {"rgb": "Genes · three sets, RGB",
         "grad": "Genes · one set, gradient",
         "ct": "Cell types · 8 µm RCTD"}

# "deck blue" is the ramp rctd_annotation_figure.py draws its own maps with, so a
# gradient here can be put beside a deck figure without a colour change of its own.
DECK_BLUE = LinearSegmentedColormap.from_list(
    "deck blue", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])
CMAPS = ["viridis", "magma", "inferno", "plasma", "cividis", "deck blue"]


def cmap_of(name):
    return DECK_BLUE if name == "deck blue" else matplotlib.colormaps[name]


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("sample", "Array", choices=D.samples(), selected="KTx_18"),
        ui.input_select("res", "Bin size", choices=list(D.RES), selected="8 µm"),
        ui.input_radio_buttons("mode", "View", choices=MODES, selected="rgb"),
        ui.hr(),

        # ---- gene channels (used by rgb and grad) ---------------------------
        ui.panel_conditional(
            "input.mode !== 'ct'",
            *[x for cid, lab, col in CHAN for x in (
                ui.tags.div(ui.tags.b(lab, style=f"color:{col}"),
                            style="margin-top:2px"),
                ui.input_selectize(f"sel_{cid}", None, choices=[], multiple=True,
                                   options={"placeholder": "type a gene name…"}),
                ui.input_select(f"preset_{cid}", None,
                                choices=["— deck panel —"] + sorted(PRESETS)),
            )]),

        # ---- gradient controls ----------------------------------------------
        ui.panel_conditional(
            "input.mode === 'grad'",
            ui.hr(),
            ui.input_radio_buttons("grad_chan", "Map which set",
                                   choices={c: l for c, l, _ in CHAN},
                                   selected="r", inline=True),
            ui.input_select("cmap", "Colour ramp", choices=CMAPS, selected="viridis"),
            ui.input_checkbox("autoscale", "Auto ceiling (99th pct of positive bins)",
                              True)),

        # ---- cell-type layer -------------------------------------------------
        ui.hr(),
        ui.output_ui("ct_controls"),

        ui.hr(),
        ui.input_numeric("vmax", "Full brightness at (UMI per bin)", 5.0,
                         min=0.5, max=200, step=0.5),
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
    ui.card(ui.output_ui("stats")),
    title="Visium HD — markers, gradients and the 8 µm RCTD calls",
    fillable=False,
)


def server(input, output, session):

    @reactive.calc
    def arr():
        return D.load(input.sample(), D.RES[input.res()])

    def is8():
        return D.RES[input.res()] == D.L0_RES_DIR

    # 18k choices cannot go to the browser as HTML; selectize searches server-side
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

    # The cell-type controls are built server-side so the 8 um restriction is
    # stated where it bites, instead of offering a list that cannot be honoured.
    @render.ui
    def ct_controls():
        if not is8():
            return ui.tags.div(
                ui.tags.b("Cell types: 8 µm only."),
                ui.tags.div(f"The Layer 0 RCTD run labels 8 µm bins. A "
                            f"{arr().bin_um:.0f} µm bin is not a subdivision of one, "
                            "so the labels are not carried over. Switch Bin size to "
                            "8 µm to use them.", style="color:#7A4A00"),
                class_="small", style="margin:4px 0")
        return ui.tags.div(
            ui.input_selectize("ct", ui.tags.b("Cell types (8 µm RCTD)"),
                               choices=D.CELLTYPE_ORDER, multiple=True,
                               options={"placeholder": "pick one or more…"}),
            ui.panel_conditional(
                "input.mode !== 'ct'",
                ui.input_checkbox("ct_mask", "Show genes only inside these types",
                                  False)),
        )

    @reactive.calc
    def picked():
        return [list(input[f"sel_{c}"]() or ()) for c, _, _ in CHAN]

    @reactive.calc
    def ct_picked():
        if not is8():
            return []
        try:
            return list(input.ct() or ())
        except Exception:
            return []

    @reactive.calc
    def mask():
        """Per-bin bool, or None. Only in the gene views, only when asked."""
        if input.mode() == "ct" or not is8() or not ct_picked():
            return None
        try:
            if not input.ct_mask():
                return None
        except Exception:
            return None
        return D.celltype_mask(arr(), ct_picked())

    # The window is held here rather than read straight off the brush: there is no
    # supported way to clear a brush from the server, so "Reset zoom" sets this to
    # None and the next drag sets it again.  A SQUARE is taken so the zoom keeps the
    # lattice's aspect and a micron stays a micron in both panels.
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

    def window():
        return win_rv()

    def keep_of(a, win):
        if not win:
            return None
        x0, y0, side = win
        return ((a.bx >= x0) & (a.bx < x0 + side) &
                (a.by >= y0) & (a.by < y0 + side))

    def grad_vmax(a, win):
        """The ceiling actually used by the gradient — computed, never assumed."""
        names = picked()["rgb".index(input.grad_chan())]
        if input.autoscale() and names:
            v, _, _ = a.channel(names)
            av = D.auto_vmax(v, keep_of(a, win), mask())
            if av:
                return av, True
        return input.vmax(), False

    # ------------------------------------------------------------------ draw
    def contrast_on(rgb):
        """Black or white, whichever is readable on that background colour."""
        r, g, b = rgb[:3]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "#222" if lum > 0.55 else "white"

    def scalebar(ax, ext, light=True, col=None):
        side = ext[1] - ext[0]
        bar = side * 0.2
        col = col or ("white" if light else "#333")
        ax.plot([ext[1] - bar - side * .05, ext[1] - side * .05],
                [ext[2] - side * .05] * 2, "-", color=col, lw=2.4)
        ax.text(ext[1] - bar / 2 - side * .05, ext[2] - side * .075,
                f"{bar:,.0f} µm", ha="center", color=col, fontsize=8.5)

    def draw(fig, ax, win, title):
        a, m = arr(), mask()
        mode = input.mode()

        if mode == "ct":
            if not is8():
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
            scalebar(ax, ext, light=False)
            sub = f"{len(ct_picked())} type(s)"

        elif mode == "grad":
            names = picked()["rgb".index(input.grad_chan())]
            if not names:
                return _msg(ax, f"pick genes for {input.grad_chan().upper()}")
            vm, auto = grad_vmax(a, win)
            rgb, ext, n, v, used, miss, shown = D.compose_gradient(
                a, names, vm, cmap_of(input.cmap()), win, m)
            ax.imshow(rgb, extent=ext, interpolation="nearest")
            sm = plt.cm.ScalarMappable(cmap=cmap_of(input.cmap()),
                                       norm=plt.Normalize(0, vm))
            cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.015)
            cb.set_label(f"UMI per {a.bin_um:.0f} µm bin"
                         f"{'  (auto)' if auto else ''}", fontsize=8)
            cb.ax.tick_params(labelsize=7)
            scalebar(ax, ext, col=contrast_on(cmap_of(input.cmap())(0.0)))
            sub = f"{'+'.join(used) if used else '—'} · ceiling {vm:,.3g}"

        else:
            if not any(picked()):
                return _msg(ax, "pick at least one gene")
            rgb, ext, n, vecs, used, miss = D.compose(
                a, picked(), input.vmax(), win, mask=m)
            ax.imshow(rgb, extent=ext, interpolation="nearest")
            scalebar(ax, ext, light=True)
            sub = f"vmax {input.vmax():g}"

        if m is not None:
            sub += " · inside " + "/".join(ct_picked())
        side = ext[1] - ext[0]
        ax.set_title(f"{title} · {side:,.0f} µm across · {a.bin_um:.0f} µm bins · "
                     f"{n:,} bins\n{sub}", fontsize=9, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        return True

    def _msg(ax, t):
        ax.text(.5, .5, t, ha="center", va="center", color="#888",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return None

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
        return fig

    @render.ui
    def status():
        a = arr()
        bits = [f"{a.n_bins:,} bins · {len(a.genes):,} genes · loaded in {a.load_s}s"]
        if is8():
            try:
                _, hit = a.labels()
                bits.append(f"{hit:,} bins have an RCTD label "
                            f"({100 * hit / a.n_bins:.0f}% of in-tissue)")
            except Exception as e:                       # noqa: BLE001
                bits.append(f"labels unavailable: {e}")
        return ui.tags.div(*[ui.tags.div(ui.tags.small(b)) for b in bits],
                           style="color:#666;margin-top:8px")

    # ----------------------------------------------------------------- stats
    @render.ui
    def stats():
        a, w, m = arr(), window(), mask()
        mode = input.mode()
        scope = (f"the selection ({w[2]:,.0f} µm across)" if w
                 else "the whole capture square")
        blocks = []

        gene_sets = picked()
        vecs = None
        if any(gene_sets):
            _, _, _, vecs, used, missing = D.compose(a, gene_sets, input.vmax(), w,
                                                     mask=m)
            st = D.channel_stats(a, vecs, w)
            rows = []
            for (cid, lab, col), names, s_, miss in zip(CHAN, used,
                                                        st["per_channel"], missing):
                rows.append(ui.tags.tr(
                    ui.tags.td(ui.tags.b(lab, style=f"color:{col}")),
                    ui.tags.td(", ".join(names) or "—"),
                    ui.tags.td(f"{s_['umi']:,.0f}"),
                    ui.tags.td(f"{s_['bins']:,} ({s_['pct']}%)"),
                    ui.tags.td(ui.tags.span(", ".join(miss), style="color:#9E2A2B")
                               if miss else "")))
            blocks.append(ui.tags.table(
                ui.tags.thead(ui.tags.tr(*[ui.tags.th(h) for h in
                              ("channel", "genes", "UMI", "bins positive",
                               "not on this panel")])),
                ui.tags.tbody(*rows), class_="table table-sm"))
            if mode == "rgb":
                blocks.append(ui.tags.p(
                    ui.tags.b("Where the channels overlap — "),
                    f"of {st['bins_any']:,} bins in {scope} carrying any channel, "
                    f"{st['exactly_1']:,} carry one, {st['exactly_2']:,} carry two and "
                    f"{st['all_3']:,} carry all three. "
                    f"By pair: R+G {st['pairs']['R+G']:,}, R+B {st['pairs']['R+B']:,}, "
                    f"G+B {st['pairs']['G+B']:,}."))

        # ---- the cell-type block -------------------------------------------
        if is8() and ct_picked():
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
                "made — spot_class reject or doublet_uncertain.",
                style="color:#7A4A00"))
            if vecs is not None:
                blocks.append(ui.tags.p(
                    ui.tags.b("Read the “also” columns as co-location, not expression. "),
                    f"A bin is {a.bin_um:.0f} µm and a nucleus here is "
                    f"{NUCLEUS_UM[0]}–{NUCLEUS_UM[1]} µm, so “Podo bins that are also "
                    "RED+” counts squares where both things landed — not that the "
                    "podocyte transcribed the red genes.", style="color:#7A4A00"))

        if not blocks:
            return ui.tags.em("Pick genes for at least one channel, or one or more "
                              "cell types. Several genes in one channel are summed — "
                              "that is what a marker panel is.")

        # ---- the footer: what the picture's numbers mean ---------------------
        if mode == "grad":
            vm, auto = grad_vmax(a, w)
            how = ("the 99th percentile of the positive bins in view, recomputed as "
                   "you zoom" if auto else "fixed, from the box in the sidebar")
            foot = (f"Gradient: {input.cmap()}, 0 to {vm:,.3g} UMI per bin ({how}), "
                    f"linear and clipped above.")
        elif mode == "ct":
            foot = ("Cell types: the July Layer 0 RCTD run (L0_celltype_8um.R), "
                    "doublet mode against the GSE183276 atlas, one call per 8 µm bin. "
                    "B_PL is the atlas's combined B + plasma class.")
        else:
            foot = (f"Brightness is summed UMI per bin ÷ {input.vmax():g}, linear and "
                    f"clipped — the same rule as deck slide 12.")
        if m is not None:
            foot += (" Genes are shown only in bins RCTD called "
                     + "/".join(ct_picked()) + "; the denominators above are unchanged.")
        blocks.append(ui.tags.small(
            foot + " SpatialRGB/app.py + spatial_rgb_data.py", style="color:#666"))
        return ui.tags.div(*blocks)


app = App(app_ui, server)
