# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'InfeRes'
copyright = '2023, Critical-Infrastructure-Systems-Lab'
author = 'InfeRes Development Team'

release = '0.1'
version = '0.1.0'

# -- General configuration

# Add any Sphinx extension module names here
extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc -- causes warnings with Napoleon however
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]
# Use Google docstrings
napoleon_google_docstring = True

# Suppress certain warnings
suppress_warnings = ['autosectionlabel.*']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments
pygments_style = 'sphinx'
highlight_language = 'csharp'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "show_prev_next": True,
    "icon_links": [
#        {"name": "Web", "url": "https://inferes-test.readthedocs.io/en/latest/", "icon": "fas fa-home"},
        {
            "name": "GitHub",
            "url": "https://github.com/ssmahto/InfeRes_test",
            "icon": "fab fa-github-square",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "search-field"],
    "search_bar_text": "Search",
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "header_links_before_dropdown": 7,
}
#html_sidebars = {
#    "**": ["sidebar-nav-bs"],
#    "index": [],
#    "overview": [],
#    "whatsnew": [],
#    "contributing": [],
#    "code_of_conduct": [],
#    "style_guide": [],
#}

# Suppress certain warnings
suppress_warnings = ['autosectionlabel.*']

# -- Options for EPUB output
epub_show_urls = 'footnote'
