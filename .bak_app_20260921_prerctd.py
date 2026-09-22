#!/usr/bin/env python3
# ==============================================================================
# app.py — an RGB marker browser for Visium HD: put any genes on any channel.
#
# WHAT IT IS FOR (author, 2026-09-21)
#   Deck slide 12 is one fixed picture: three fixed marker panels, one fixed
#   window, one array.  Making the next one means editing a script.  This turns
#   that figure into a control panel -- pick an array, pick genes for red, green
#   and blue (several per colour, summed, which is what a "marker panel" is),
#   and drag a box to zoom.
#
# WHAT IT DOES NOT CHANGE
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
#
# PERFORMANCE: see spatial_rgb_data.py -- the matrix is transposed to gene-major
#   once per array (~1-2 s) and then a gene is ~0.035 ms, so the picture keeps up
#   with the controls.  Whole-section compose is ~0.04 s at 8 um.
#
# Usage:
#   ../spatial-venv/bin/python -m shiny run --port 8000 SpatialRGB/app.py
#   then open http://127.0.0.1:8000
# ==============================================================================

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shiny import App, reactive, render, ui

BASE = "/home/abb2013/Documents/Spatial"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spatial_rgb_data as D                                   # noqa: E402
from celltype_marker_maps import PANELS as DECK_PANELS         # noqa: E402

CHAN = [("r", "RED", "#FF4040"), ("g", "GREEN", "#3BD23B"), ("b", "BLUE", "#5B8CFF")]
PRESETS = {k.split("  ")[0]: list(v) for k, v in DECK_PANELS.items()}
NUCLEUS_UM = (6.8, 7.1)

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("sample", "Array", choices=D.samples(), selected="KTx_18"),
        ui.input_select("res", "Bin size", choices=list(D.RES), selected="8 µm"),
        ui.hr(),
        *[x for cid, lab, col in CHAN for x in (
            ui.tags.div(ui.tags.b(lab, style=f"color:{col}"),
                        style="margin-top:2px"),
            ui.input_selectize(f"sel_{cid}", None, choices=[], multiple=True,
                               options={"placeholder": "type a gene name…"}),
            ui.input_select(f"preset_{cid}", None,
                            choices=["— deck panel —"] + sorted(PRESETS)),
        )],
        ui.hr(),
        ui.input_numeric("vmax", "Full brightness at (UMI per bin)", 5.0,
                         min=0.5, max=200, step=0.5),
        ui.input_action_button("reset", "Reset zoom", class_="btn-sm"),
        ui.output_ui("status"),
        width=330,
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
    title="Visium HD — three marker sets, one picture",
    fillable=False,
)


def server(input, output, session):

    @reactive.calc
    def arr():
        return D.load(input.sample(), D.RES[input.res()])

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

    @reactive.calc
    def picked():
        return [list(input[f"sel_{c}"]() or ()) for c, _, _ in CHAN]

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

    def draw(ax, win, title):
        a = arr()
        sets = picked()
        if not any(sets):
            ax.text(.5, .5, "pick at least one gene", ha="center", va="center",
                    color="#888", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            return None
        rgb, ext, n, vecs, used, missing = D.compose(a, sets, input.vmax(), win)
        ax.imshow(rgb, extent=ext, interpolation="nearest")
        side = ext[1] - ext[0]
        bar = side * 0.2
        ax.plot([ext[1] - bar - side * .05, ext[1] - side * .05],
                [ext[2] - side * .05] * 2, "-", color="white", lw=2.4)
        ax.text(ext[1] - bar / 2 - side * .05, ext[2] - side * .075,
                f"{bar:,.0f} µm", ha="center", color="white", fontsize=8.5)
        ax.set_title(f"{title} · {side:,.0f} µm across · {a.bin_um:.0f} µm bins · "
                     f"{n:,} bins", fontsize=9.5, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        return vecs

    @render.plot
    def section():
        fig, ax = plt.subplots(figsize=(6.6, 6.6))
        fig.subplots_adjust(0.01, 0.01, 0.99, 0.95)
        draw(ax, None, f"{input.sample()} — whole capture square")
        return fig

    @render.plot
    def zoom():
        fig, ax = plt.subplots(figsize=(6.6, 6.6))
        fig.subplots_adjust(0.01, 0.01, 0.99, 0.95)
        w = window()
        if w is None:
            ax.text(.5, .5, "drag a box on the left", ha="center", va="center",
                    color="#888", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            draw(ax, w, "selection")
        return fig

    @render.ui
    def status():
        a = arr()
        return ui.tags.div(
            ui.tags.small(f"{a.n_bins:,} bins · {len(a.genes):,} genes · "
                          f"loaded in {a.load_s}s"),
            style="color:#666;margin-top:8px")

    @render.ui
    def stats():
        a = arr()
        sets = picked()
        if not any(sets):
            return ui.tags.em("Pick genes for at least one channel. "
                              "Several genes in one channel are summed — that is "
                              "what a marker panel is.")
        w = window()
        _, _, _, vecs, used, missing = D.compose(a, sets, input.vmax(), w)
        st = D.channel_stats(a, vecs, w)
        rows = []
        for (cid, lab, col), names, s_, miss in zip(CHAN, used, st["per_channel"],
                                                    missing):
            rows.append(ui.tags.tr(
                ui.tags.td(ui.tags.b(lab, style=f"color:{col}")),
                ui.tags.td(", ".join(names) or "—"),
                ui.tags.td(f"{s_['umi']:,.0f}"),
                ui.tags.td(f"{s_['bins']:,} ({s_['pct']}%)"),
                ui.tags.td(ui.tags.span(", ".join(miss), style="color:#9E2A2B")
                           if miss else "")))
        scope = (f"the selection ({w[2]:,.0f} µm across)" if w else
                 "the whole capture square")
        return ui.tags.div(
            ui.tags.table(
                ui.tags.thead(ui.tags.tr(*[ui.tags.th(h) for h in
                              ("channel", "genes", "UMI", "bins positive",
                               "not on this panel")])),
                ui.tags.tbody(*rows), class_="table table-sm"),
            ui.tags.p(ui.tags.b("Where the channels overlap — "),
                      f"of {st['bins_any']:,} bins in {scope} carrying any channel, "
                      f"{st['exactly_1']:,} carry one, {st['exactly_2']:,} carry two and "
                      f"{st['all_3']:,} carry all three. "
                      f"By pair: R+G {st['pairs']['R+G']:,}, R+B {st['pairs']['R+B']:,}, "
                      f"G+B {st['pairs']['G+B']:,}."),
            ui.tags.p(ui.tags.b("Read a mixed colour as co-location at "
                                f"{a.bin_um:.0f} µm, not as a cell. "),
                      f"A bin is {a.bin_um:.0f} µm and a nucleus here is "
                      f"{NUCLEUS_UM[0]}–{NUCLEUS_UM[1]} µm, so yellow means the red and "
                      "green gene sets both put UMI in the same square — not that one "
                      "cell expressed both, and not that the two cells touch. "
                      "A dark channel may mean its cells are elsewhere OR that its genes "
                      "are rare: check the UMI column before reading absence.",
                      style="color:#7A4A00"),
            ui.tags.small(f"Brightness is summed UMI per bin ÷ {input.vmax():g}, "
                          f"linear and clipped — the same rule as deck slide 12. "
                          f"SpatialRGB/app.py + spatial_rgb_data.py",
                          style="color:#666"))


app = App(app_ui, server)
