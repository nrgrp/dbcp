from .problem import BiconvexProblem

__all__ = ["BiconvexProblem", "__version__"]

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("dbcp")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
