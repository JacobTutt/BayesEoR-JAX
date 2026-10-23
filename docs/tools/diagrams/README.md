# Documentation diagrams

The Home page uses `workflow.tex`, adapted from the supplied full workflow
illustration. `component_workflow.tex` contains the EoR-only, foreground-only
and combined forward-model variants, ending at **Modelled Data** rather than
inference. The Modelled Data panel uses the right-hand, noiseless waterfall
from `icon_likelihood.pdf`. `inference_workflow.tex` contains the two-row
inference-only workflow, with modelled data and observation in a shared input
panel. All diagrams share `style.tex` and the thumbnails in `plots/`.
The two-row model and inference diagrams share a fixed canvas width so their
panels and text remain the same size when displayed at the same width. The
inference rows use a tighter gap because they need no intervening operator labels.

The diagrams are schematic illustrations, not numerical outputs of a common
model realisation. The cosmological cube thumbnails originate from supplied
21cmSPACE simulation visualisations. Original manuscript files are not modified.

## Rebuild

With TeX (TikZ and newtx), Poppler and CairoSVG available, run from the repository:

```bash
uv run --no-project --with cairosvg python docs/tools/diagrams/build.py
```

On macOS with Homebrew Cairo, set `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`
if CairoSVG cannot find the library. Ensure `pdflatex` is on your PATH.

The build renders light and dark PNG pairs into `docs/assets/diagrams/`.
Dark mode uses pale labels and arrows on dark panels, preserving the original
sky/cube images. `dark_svg.py` recolours the vector plot furniture in the small
PDF thumbnails without changing their embedded raster data. Intermediate files
are temporary; the generated PNGs are committed, so ordinary documentation
builds need neither TeX nor CairoSVG.

The Markdown image fragments `#only-light` and `#only-dark` use Material's
built-in palette switching. Each visible image links to its own full-resolution
variant. Inspect both palettes after edits, especially text contrast and
thumbnail transparency.
