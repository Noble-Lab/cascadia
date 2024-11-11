from importlib.metadata import version

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Cascadia'
copyright = '2024, Justin Sanders'
author = 'Justin Sanders'
# release = version("cascadia")
# version = ".".join(release.split(".")[:2])
version = '0.0.1'

# -- General configuration

extensions = ['myst_parser']

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

# The format for each file suffix:
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
