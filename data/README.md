# Collector cache

The `automation/map-snapshot` branch stores one atomic JSON cache file per
unique Scholar profile under `data/authors/`. The default branch contains only
this documentation and the trusted pipeline code.

Cache records retain the last good normalized source observation. Registry role
changes never delete caches; excluded and former relationships remain available
to the lossless dataset pipeline.
