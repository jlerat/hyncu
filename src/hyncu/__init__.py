
try:
    import cf_units
    CF_UNITS_VERSION = cf_units.__version__
    HAS_CF_UNITS = True
except ImportError:
    HAS_CF_UNITS = False

