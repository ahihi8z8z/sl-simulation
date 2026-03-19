"""Unit tests for Step 0: Project skeleton — verify imports and CLI help."""

import subprocess
import sys


class TestProjectSkeleton:
    def test_module_import(self):
        """python -m serverless_sim should be importable."""
        import serverless_sim
        assert hasattr(serverless_sim, "__name__")

    def test_cli_help_runs(self):
        """python -m serverless_sim --help should print usage and exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "serverless_sim", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "serverless" in result.stdout.lower()

    def test_subcommands_present(self):
        """Help output should list simulate, train, infer commands."""
        result = subprocess.run(
            [sys.executable, "-m", "serverless_sim", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "simulate" in result.stdout
        assert "train" in result.stdout
        assert "infer" in result.stdout

    def test_simulate_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "serverless_sim", "simulate", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--sim-config" in result.stdout

    def test_no_command_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "serverless_sim"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
