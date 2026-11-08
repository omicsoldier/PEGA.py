"""
pega.runners
============
Scripts PEGA executes inside another conda environment.

Nothing here is imported by ``pega_env``: each module runs under the
interpreter of the environment its tool needs, launched through
``conda run``. They live outside ``pega.predictors`` on purpose — the
registry imports every module in that package to discover predictors,
and these would fail to import under ``pega_env``.
"""
