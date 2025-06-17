# Configuration file for the Sphinx documentation builder.

import os
import sys

# Debug: Print current working directory and paths
print(f"Current working directory: {os.getcwd()}")
print(f"Config file location: {__file__}")

# Add the scripts directory to the Python path
# Since we're in docs/source, scripts is ../../scripts
scripts_path = os.path.abspath('../../scripts')
sys.path.insert(0, scripts_path)
print(f"Added to Python path: {scripts_path}")

# Also add the project root to Python path
project_root = os.path.abspath('../..')
sys.path.insert(0, project_root)
print(f"Added project root to Python path: {project_root}")

# Debug: List files in the scripts directory
if os.path.exists(scripts_path):
    print("Files in scripts directory:")
    for item in os.listdir(scripts_path):
        if item.endswith('.py'):
            print(f"  - {item}")
else:
    print(f"WARNING: Scripts directory not found at {scripts_path}")

# -- Project information -----------------------------------------------------
project = 'Data Quality Assessment'
copyright = '2025, Akash Suresh Kumar'
author = 'Akash Suresh Kumar'
release = '3.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'autoapi.extension',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # For Google/NumPy style docstrings
    'sphinx.ext.intersphinx',  # For linking to other documentation
    'sphinx.ext.todo',  # For TODO notes
]

# AutoAPI configuration
autoapi_type = 'python'
autoapi_dirs = ['../../scripts']  # Point to scripts directory
autoapi_root = 'api'
autoapi_add_toctree_entry = True  # CHANGED: Set to True to automatically add to toctree
autoapi_generate_api_docs = True
autoapi_keep_files = True  # Keep generated files for debugging

# Exclude problematic directories and files
autoapi_ignore = [
    '*/iudx/*',  # Exclude the iudx virtual environment/package directory
    '*/Lib/*',   # Exclude any Lib directories
    '*/site-packages/*',  # Exclude site-packages
    '*/__pycache__/*',  # Exclude Python cache
    '*/.*',      # Exclude hidden files
    '*/venv/*',  # Exclude virtual environments
    '*/env/*',   # Exclude virtual environments
    '*/.git/*',  # Exclude git files
]

# Debug AutoAPI
print(f"AutoAPI will scan: {os.path.abspath('../../scripts')}")
print("AutoAPI will ignore patterns:", autoapi_ignore)

# AutoAPI options for comprehensive documentation
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
]

# AutoAPI template directory (optional, for custom templates)
autoapi_template_dir = '_templates/autoapi'

# Napoleon settings for different docstring styles
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True  # CHANGED: Include __init__ docstrings
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True  # CHANGED: Better formatting for examples
napoleon_use_admonition_for_notes = True    # CHANGED: Better formatting for notes
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# Make sure autodoc can import modules
autodoc_mock_imports = []

# Templates path
templates_path = ['_templates']

# List of patterns to exclude when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': False,  # CHANGED: Don't collapse navigation
    'sticky_navigation': True,
    'navigation_depth': 6,  # CHANGED: Deeper navigation
    'includehidden': True,
    'titles_only': False
}

# GitHub Pages configuration
html_baseurl = 'https://akash-suresh-kumar.github.io/data-quality-assessment/'

# Master document
master_doc = 'index'

# Show todos in documentation
todo_include_todos = True

# Add any Sphinx extension configuration here
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

print("Configuration loaded successfully!")