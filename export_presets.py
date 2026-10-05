#!/usr/bin/env python3
# ==============================================================================
# export_presets.py — snapshot the deck's marker panels into presets.json.
#
# WHY THIS EXISTS (author, 2026-10-05: "create a github repo with only the app")
#   The app offers the deck's marker panels as per-channel presets.  It used to
#   import them from celltype_marker_maps.py at the project root, which does not
#   exist in a repository that holds only SpatialRGB/.  So the panels are copied
#   into presets.json, and app.py reads that file instead.
#
#   celltype_marker_maps.PANELS stays the source; presets.json is a copy.  After
#   changing a panel there, re-run this and commit the new presets.json.
#
# INPUT    ../celltype_marker_maps.py  (PANELS)  -- needs the full project
# OUTPUT   presets.json  next to this file
# RUN      ../spatial-venv/bin/python SpatialRGB/export_presets.py
#          --check   exit 1 if presets.json no longer matches PANELS
# ==============================================================================
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "presets.json")
sys.path.insert(0, os.path.dirname(HERE))

try:
    from celltype_marker_maps import PANELS
except ImportError:
    sys.exit("celltype_marker_maps.py not found one level up: this script only "
             "runs inside the full project, not in the app-only repository.")

panels = {k: list(v) for k, v in PANELS.items()}

if "--check" in sys.argv:
    have = json.load(open(OUT))["panels"] if os.path.exists(OUT) else None
    if have != panels:
        sys.exit(f"{OUT} is out of date: re-run without --check")
    print(f"presets.json matches PANELS ({len(panels)} panels)")
else:
    with open(OUT, "w") as f:
        json.dump({"source": "celltype_marker_maps.PANELS", "panels": panels},
                  f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT}: {len(panels)} panels, "
          f"{sum(len(v) for v in panels.values())} genes")
