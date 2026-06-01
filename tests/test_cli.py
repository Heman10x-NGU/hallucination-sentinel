"""Tests for the CLI module."""

import pytest


class TestCLIImports:
    """Test that CLI module imports correctly."""

    def test_import_main(self):
        from hallucination_sentinel.cli import main
        assert callable(main)

    def test_import_check_command(self):
        from hallucination_sentinel.cli import check
        assert callable(check)

    def test_import_calibrate_command(self):
        from hallucination_sentinel.cli import calibrate
        assert callable(calibrate)


class TestCLIRun:
    """Test CLI invocation (stubs only, no API calls)."""

    def test_main_exists(self):
        """main() should be a callable."""
        from hallucination_sentinel.cli import main
        assert main is not None
