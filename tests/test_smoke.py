"""Smoke tests to verify project scaffold is functional."""

import slfd


def test_version_exists():
    """Package exposes a version string."""
    assert hasattr(slfd, "__version__")
    assert isinstance(slfd.__version__, str)
    assert slfd.__version__ == "0.1.0"


def test_import_succeeds():
    """Package can be imported without errors."""
    import slfd  # noqa: F811

    assert slfd is not None
