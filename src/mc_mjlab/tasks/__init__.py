"""RL tasks built on the mc_rtc residual action terms.

Discovery mirrors mjlab's own ``tasks`` package: every sub-package here is
imported on import of this one, and each registers its task ids as a side
effect, so adding a task means adding a directory and nothing else. mjlab
reaches this module through the ``mjlab.tasks`` entry point declared in
``pyproject.toml``, which is what lets mjlab's ``train``/``play`` scripts drive
this repo without it shipping copies of them.

Note ``import_packages`` only imports sub-*packages*, so a task has to be a
directory with an ``__init__.py``; a bare module beside this one is never
imported and would silently never register.
"""

from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)
