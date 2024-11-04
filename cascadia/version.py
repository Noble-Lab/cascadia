"""Package version information."""

def _get_version():
    """
    Retrieve the package version.

    Using setuptools-scm dynamically sets the package version depending on the
    latest git release and commits since that point. This function returns the
    current version number.

    Returns
    -------
        The package version number. If not version is found, returns `None`.
    """
    try:
        # Fast, but only works in Python 3.8+.
        from importlib.metadata import version, PackageNotFoundError

        try:
            return version("cascadia")
        except PackageNotFoundError:
            return None
    except ImportError:
        # Slow, but works for all Python 3+.
        from pkg_resources import get_distribution, DistributionNotFound

        try:
            return get_distribution("cascadia").version
        except DistributionNotFound:
            return None