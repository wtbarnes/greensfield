# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import datetime
import os

from packaging.version import Version

# The full version, including alpha/beta/rc tags
from greensfield import __version__

# -- Project information -----------------------------------------------------

_version_ = Version(__version__)
# NOTE: Avoid "post" appearing in version string in rendered docs
if _version_.is_postrelease:
    version = release = f'{_version_.major}.{_version_.minor}.{_version_.micro}'
# NOTE: Avoid long githashes in rendered Sphinx docs
elif _version_.is_devrelease:
    version = release = f'{_version_.major}.{_version_.minor}.dev{_version_.dev}'
else:
    version = release = str(_version_)
is_development = _version_.is_devrelease

project = "greensfield"
author = "Will Barnes"
copyright = f"{datetime.datetime.utcnow().year}, {author}"  # noqa: A001

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# Treat everything in single ` as a Python reference.
default_role = 'py:obj'

# Enable nitpicky mode, which forces links to be non-broken
nitpicky = True
# This is not used. See docs/nitpick-exceptions file for the actual listing.
nitpick_ignore = []
for line in open('nitpick-exceptions'):
    if line.strip() == "" or line.startswith("#"):
        continue
    dtype, target = line.split(None, 1)
    target = target.strip()
    nitpick_ignore.append((dtype, target))

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        (None, "http://www.astropy.org/astropy-data/intersphinx/python3.inv"),
    ),
    "numpy": (
        "https://numpy.org/doc/stable/",
        (None, "http://www.astropy.org/astropy-data/intersphinx/numpy.inv"),
    ),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "sunpy": ("https://docs.sunpy.org/en/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
    "streamtracer": ("https://docs.sunpy.org/projects/streamtracer/en/stable/", None)
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "show_nav_level": 2,
    "logo": {
        "text": f"greensfield {version}",
    },
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/wtbarnes/greensfield",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/greensfield",
            "icon": "fa-brands fa-python",
        },
    ],
}
html_context = {
    "github_user": "wtbarnes",
    "github_repo": "greensfield",
    "github_version": "main",
    "doc_path": "docs",
}
# Sidebar removal
html_sidebars = {
    "bibliography*": [],
    "development*": [],
}
# Render inheritance diagrams in SVG
graphviz_output_format = "svg"
graphviz_dot_args = [
    '-Nfontsize=10',
    '-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif',
    '-Efontsize=10',
    '-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif',
    '-Gfontsize=10',
    '-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif'
]

# -- Other options ----------------------------------------------------------
bibtex_bibfiles = ['references.bib']

# -- Sphinx gallery -----------------------------------------------------------
sphinx_gallery_conf = {
    'backreferences_dir': os.path.join('generated', 'modules'),
    'filename_pattern': '^((?!skip_).)*$',
    'examples_dirs': os.path.join('..', 'examples'),
    'within_subsection_order': 'ExampleTitleSortKey',
    'gallery_dirs': os.path.join('generated', 'gallery'),
    'matplotlib_animations': True,
    'abort_on_example_error': False,
    'plot_gallery': 'True',
    'remove_config_comments': True,
    'doc_module': ('greensfield',),
    'only_warn_on_example_error': True,
}
