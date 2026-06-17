"""Bundled data files (encoder centroids, etc).

This subpackage exists to ship JSON / .npy data with the wheel via
``[tool.setuptools.package-data]``. The :mod:`lens_eval.encoders` module
reads the files directly; nothing here is part of the public API.
"""
