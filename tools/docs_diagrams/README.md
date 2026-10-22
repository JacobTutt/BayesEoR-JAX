# Documentation diagrams

The Home page uses the supplied full workflow illustration in
`docs/assets/diagrams/workflow.png`. The component-only versions reuse its
TikZ styling and thumbnail assets, remove the other sky component and all
inference boxes, and retain the left-hand waterfall under **Modelled Data**.
They are schematic illustrations, not numerical outputs of a common model
realisation. The cosmological cube thumbnails originate from the supplied
21cmSPACE simulation visualisations.

`component_workflow.tex` contains the EoR-only, foreground-only and combined
forward-model variants; `style.tex` contains the shared styles and cube drawing.
The combined version appears on the Model overview and retains both sky
components while ending at Modelled Data. The foreground variant starts in image space
and therefore has no transverse or line-of-sight Fourier stages.

## Rebuild

From this directory, with a TeX distribution including TikZ and newtx installed,
and Poppler providing `pdftoppm`:

```bash
diagram_build_dir=$(mktemp -d)
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory="$diagram_build_dir" -jobname=eor-workflow \
  '\input{component_workflow.tex}'
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory="$diagram_build_dir" -jobname=foreground-workflow \
  '\def\ForegroundWorkflow{1}\input{component_workflow.tex}'
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory="$diagram_build_dir" -jobname=joint-model-workflow \
  '\def\JointWorkflow{1}\input{component_workflow.tex}'
pdftoppm -singlefile -png -r 300 "$diagram_build_dir/eor-workflow.pdf" \
  ../../docs/assets/diagrams/eor-workflow
pdftoppm -singlefile -png -r 300 "$diagram_build_dir/foreground-workflow.pdf" \
  ../../docs/assets/diagrams/foreground-workflow
pdftoppm -singlefile -png -r 300 "$diagram_build_dir/joint-model-workflow.pdf" \
  ../../docs/assets/diagrams/joint-model-workflow
```

Inspect all three PNGs after editing. Generated PNGs are committed so ordinary
documentation builds require neither TeX nor Poppler. The original manuscript
figures and source files are not modified by this workflow.
