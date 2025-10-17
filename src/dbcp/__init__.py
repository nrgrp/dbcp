from .problem import BiconvexProblem, BiconvexRelaxProblem

__all__ = ["BiconvexProblem", "BiconvexRelaxProblem", "__version__"]

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("dbcp")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
