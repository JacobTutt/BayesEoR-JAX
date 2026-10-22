# Building and publishing the documentation

The site uses MkDocs Material, mkdocstrings for the Python API, MathJax for equations,
and mike for versioned documentation.

## Local preview

From the repository root, using Python 3.12 or newer:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs serve
```

Open <http://127.0.0.1:8000>. Changes to the guides and configuration are rebuilt
automatically. To check the complete site:

```bash
python -m mkdocs build --strict
```

The generated site is written to `site/`, which is excluded from Git.
The API reference reads source docstrings without importing the package, so building
the documentation does not require a GPU or the native MultiNest library.

## Editing

- Edit guides in `docs/` and navigation in `mkdocs.yml`.
- Edit API descriptions in the corresponding Python docstrings, using NumPy style.
- Add API entries to the appropriate page under `docs/api/`.
- Use Markdown mathematics for equations; MathJax renders them in the browser.

The Home and Model pages use committed workflow images in `docs/assets/diagrams/`.
Editable diagram sources and regeneration commands are in
[tools/docs_diagrams](https://github.com/JacobTutt/BayesEoR-JAX/tree/main/tools/docs_diagrams).
Their generation is separate from the documentation build and does not require
changes to the original manuscript figures. Light/dark pairs use Material's
`#only-light` and `#only-dark` image fragments to follow the active site palette.

The header icon and Home wordmark use transparent light/dark SVG pairs in
`docs/assets/branding/`, taken from the supplied logo source artwork. The `-light`
files use dark lettering and outlines; the `-dark` files use white lettering and
outlines, retaining the original colours. `branding.css` follows Material's
active `data-md-color-scheme` so the correct artwork also appears when the user
changes the theme manually. The small `partials/logo.html` override applies the
same pairing to the header and mobile navigation.

The **Documentation** GitHub Actions workflow runs a strict build for relevant
pushes and pull requests. Documentation dependencies are installed separately from
the normal CPU and HPC environments.

## Enable hosting when ready

Publishing is manual: ordinary pushes and pull requests only build the site.

1. On GitHub, open **Actions → Documentation → Run workflow** on `main`.
2. Enable **Publish the dev documentation to gh-pages**.
3. Once the workflow succeeds, open **Settings → Pages** and select
   **Deploy from a branch**, branch `gh-pages`, folder `/ (root)`.

The site will be available at <https://jacobtutt.github.io/BayesEoR-JAX/>.
The workflow publishes the `dev` version and makes it the default.
It does not change repository hosting settings.

## Versioned releases

When release documentation is needed, mike can publish additional versions from
the corresponding source checkout:

```bash
mike deploy --push --update-aliases VERSION latest
mike set-default --push latest
```

Replace `VERSION` with the release number. At that point, change the default
version in `mkdocs.yml` and the publishing workflow from `dev` to `latest`.
Existing versions are retained on `gh-pages`.
