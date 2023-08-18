Welcome to InfeRes: A Python package for inferring reservoir water extent, level and storage volume
====================================================================================================

.. image:: https://img.shields.io/pypi/l/sciris.svg
 :target: https://github.com/ssmahto/InfeRes_test/blob/main/LICENCE

What is InfeRes?
----------------

``InfeRes`` is a python module that is designed to help automatic extraction of reservoir characteristics (water surface area, level, and storage-volume) time-series by taking leverage
of the `Google Earth Engine Landsat data collection <https://developers.google.com/earth-engine/datasets/catalog/landsat/>`_, and
high resolition `DEM (30m) <https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1/>`_.
It built on top of `GDAL <https://gdal.org/>`_, `Scikit-Learn <https://scikit-learn.org/>`_, `NumPy <https://numpy.org/>`_ and `Matplotlib <https://matplotlib.org/>`_,
and other popular python packages. ``InfeRes`` is developed with a novel algorithm which helps inferring reservoir characteristics even from the partially cloudy images.
``InfeRes`` can be applied to monitor water surface area in any reservoir or waterbody; whereas, storage-volume can be obtained for the reservoirs build only after 2000 (limitation of DEM acquisition).

For more information, see the full `documentation <https://inferes-test.readthedocs.io/en/latest/>`_, or `GitHub <https://github.com/ssmahto/InfeRes_test>`_.


Currently ``InfeRes`` is only tested on Python 3.8.
