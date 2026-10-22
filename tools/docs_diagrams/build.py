"""Render the documentation flowcharts in light and dark palettes.

Requires pdflatex, pdftocairo, pdftoppm and CairoSVG. These are development-only
dependencies; normal documentation builds use the committed PNGs.
"""

from pathlib import Path
import subprocess
import tempfile

import cairosvg

from dark_svg import convert


SOURCE = Path(__file__).resolve().parent
OUTPUT = SOURCE.parents[1] / 'docs/assets/diagrams'
VARIANTS = {
    'workflow': ('workflow.tex', ''),
    'eor-workflow': ('component_workflow.tex', ''),
    'foreground-workflow': ('component_workflow.tex', r'\def\ForegroundWorkflow{1}'),
    'joint-model-workflow': ('component_workflow.tex', r'\def\JointWorkflow{1}'),
}


def build():
    """Compile both palettes, preserving the original sky and cube thumbnails."""
    with tempfile.TemporaryDirectory(prefix='bayeseor-diagrams-') as directory:
        work = Path(directory)
        dark_plots = work / 'dark-plots'
        dark_plots.mkdir()
        # Recolour only vector plot furniture; leave embedded waterfall data alone.
        for icon in (SOURCE / 'plots').glob('*.pdf'):
            svg = work / f'{icon.stem}.svg'
            subprocess.run(['pdftocairo', '-svg', str(icon), str(svg)], check=True)
            convert(svg, svg)
            cairosvg.svg2pdf(url=str(svg), write_to=str(dark_plots / icon.name))
        for name, (source, variant) in VARIANTS.items():
            for dark in (False, True):
                job = name + ('-dark' if dark else '')
                palette = rf'\def\DarkWorkflow{{1}}\def\DarkPlots{{{dark_plots}}}' if dark else ''
                subprocess.run([
                    'pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                    f'-output-directory={work}', f'-jobname={job}',
                    palette + variant + rf'\input{{{source}}}',
                ], cwd=SOURCE, check=True, stdout=subprocess.DEVNULL)
                subprocess.run([
                    'pdftoppm', '-singlefile', '-png', '-r', '300',
                    str(work / f'{job}.pdf'), str(OUTPUT / job),
                ], check=True)
                print(f'Rendered {job}', flush=True)


if __name__ == '__main__':
    build()
