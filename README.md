# SpatialRGB

A Shiny for Python app for looking at Visium HD arrays: any genes on the red,
green and blue channels, over the 8 µm RCTD cell-type calls, with the same
window shown in the microscope H&E.

This repository holds the code only. The data is not here and has to be
provided separately.

## Data it expects

Set `SPATIAL_BASE` to a folder laid out like this (if unset, the app uses the
folder that contains `SpatialRGB/`):

    Processed/<sample>/outs/binned_outputs/square_{002,008}um/
        filtered_feature_bc_matrix.h5
        spatial/tissue_positions.parquet
        spatial/scalefactors_json.json
    L0_CellType/<sample>_celltype_008um.csv
    HiRes_Index/<sample>/            H&E tile pyramid (build_he_index.py)
    HiRes_Images/<sample>_fullres.TIF   full-resolution H&E, for the export

## Toy sample

`toy/` holds one small sample, `KTx_toy`, so the app runs straight after a
clone: a 1024 µm block cut from one real array (KTx_18), with its counts at
2 µm and 8 µm, its cell-type labels and the matching H&E, about 17 MB. When
`SPATIAL_BASE` is not set and there is no `Processed/` folder beside the
app, the app uses `toy/`. `toy/toy_source.json` records how it was cut, and
`build_toy_sample.py` rebuilds it (that needs the full data).

The full-resolution H&E export does not work on the toy sample: it has the
tile index but not the original image.

## Install and run

    python -m venv venv && venv/bin/pip install -r requirements.txt
    venv/bin/python auth.py --set          # choose the login password, once
    venv/bin/python -m shiny run --port 8765 app.py          # toy sample
    SPATIAL_BASE=/path/to/data venv/bin/python -m shiny run --port 8765 app.py

The app refuses to start without a password. The password is sent with HTTP
basic auth, so use it only over HTTPS or through an SSH tunnel.

## Files

| File | What it does |
|---|---|
| `app.py` | the interface and its callbacks |
| `spatial_rgb_data.py` | loads arrays and composes the pictures |
| `he_image.py` | crops the microscope H&E for a window |
| `build_he_index.py` | builds the H&E tile pyramid, once per array |
| `he_crop_selftest.py` | checks the H&E crop lands on the right tissue |
| `auth.py` | the login password |
| `presets.json` | marker panels offered as presets |
| `build_toy_sample.py` | cuts the toy sample; runs only with the full data |
| `toy/` | the toy sample |
| `export_presets.py` | rewrites `presets.json`; runs only inside the full project |

## Where changes are made

The app is developed inside a larger project repository, in its `SpatialRGB/`
folder, and published here from it:

    git subtree push --prefix=SpatialRGB github master
