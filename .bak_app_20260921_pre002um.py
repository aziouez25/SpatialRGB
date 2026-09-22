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

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    ui.card(ui.output_ui("stats")),
    title="Visium HD — expression gradients over the 8 µm RCTD map",
    fillable=False,
)


def server(input, output, session):

    @reactive.calc
    def arr():
        return D.load(input.sample(), D.RES[input.res()])

    def is8():
        return D.RES[input.res()] == D.L0_RES_DIR

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
        if not is8():
            return ui.tags.div(
                ui.tags.b("Cell types: 8 µm only."),
                ui.tags.div(f"The Layer 0 RCTD run labels 8 µm bins. A "
                            f"{arr().bin_um:.0f} µm bin is not a subdivision of one, "
                            "so the labels are not carried over. Switch Bin size to "
                            "8 µm to use them.", style="color:#7A4A00"),
                class_="small", style="margin:4px 0")
        return ui.tags.div(
            ui.input_selectize("ct", ui.tags.b("Cell types (8 µm RCTD) — the ground"),
                               choices=D.CELLTYPE_ORDER, multiple=True,
                               options={"placeholder": "pick one or more…"}),
            ui.panel_conditional(
                "input.mode !== 'ct'",
                ui.input_slider("ground", "How far the chosen type lifts out of black",
                                min=0.10, max=0.40, value=0.22, step=0.01),
                ui.input_checkbox("ct_mask", "Also hide genes outside these types",
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

    def _opt(name, default):
        """A control that lives inside the dynamic sidebar may not exist yet."""
        try:
            v = input[name]()
            return default if v is None else v
        except Exception:
            return default

    @reactive.calc
    def ground():
        """Per-bin grey: the RCTD map as the ground under the gradients."""
        if not ct_picked():
            return None
        return D.celltype_background(arr(), ct_picked(), sel=_opt("ground", 0.22))

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

    def window():
        return win_rv()

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
                sub += " · ground: " + "/".join(ct_picked())
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
                    ui.tags.td(ui.tags.span(", ".join(miss), style="color:#9E2A2B")
                               if miss else "")))
            blocks.append(ui.tags.table(
                ui.tags.thead(ui.tags.tr(*[ui.tags.th(h) for h in
                              ("channel", f"0 → {input.vmax():g} UMI", "genes", "UMI",
                               "bins positive", "not on this panel")])),
                ui.tags.tbody(*rows), class_="table table-sm"))
            blocks.append(ui.tags.p(
                ui.tags.b("Where the channels overlap — "),
                f"of {st['bins_any']:,} bins in {scope} carrying any channel, "
                f"{st['exactly_1']:,} carry one, {st['exactly_2']:,} carry two and "
                f"{st['all_3']:,} carry all three. "
                f"By pair: R+G {st['pairs']['R+G']:,}, R+B {st['pairs']['R+B']:,}, "
                f"G+B {st['pairs']['G+B']:,}."))

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
                foot += (" The ground is the 8 µm RCTD map: every in-tissue bin sits near "
                         "black and " + "/".join(ct_picked()) + " lifts to grey.")
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


app = App(app_ui, server)
