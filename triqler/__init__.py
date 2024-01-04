

__version__ = "0.0.0"
try:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version(__name__)
    except PackageNotFoundError:
        pass
except ImportError:
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        __version__ = get_distribution(__name__).version
    except DistributionNotFound:
        pass

__copyright__ = """Copyright (c) 2018-2024 Matthew The, Patrick Truong. All rights reserved.
Written by:
- Matthew The (matthew.the@scilifelab.se)
- Patrick Truong (patrick.truong@scilifelab.se)
in the School of Engineering Sciences in Chemistry, Biotechnology and Health
at the Royal Institute of Technology in Stockholm."""