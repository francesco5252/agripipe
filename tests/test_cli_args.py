"""Test per i nuovi argomenti CLI di AgriPipe Pro."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from agripipe.cli import app


@pytest.fixture
def runner():
    return CliRunner()


@patch("agripipe.cli.AgriCleaner")
@patch("agripipe.cli.load_raw")
@patch("torch.save")
def test_cli_run_fuzzy_flag(mock_save, mock_load, mock_cleaner, runner, tmp_path):
    """Verifica che il flag --fuzzy venga passato al loader."""
    input_file = tmp_path / "test.xlsx"
    input_file.touch()
    output_file = tmp_path / "out.pt"

    # Mocking
    mock_cleaner.from_preset.return_value.clean.return_value = MagicMock()

    result = runner.invoke(
        app,
        [
            "run",
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--preset",
            "ulivo_ligure",
            "--fuzzy",
        ],
    )

    assert result.exit_code == 0
    # Verifica che load_raw sia stato chiamato con fuzzy=True
    mock_load.assert_called_once()
    assert mock_load.call_args[1]["fuzzy"] is True


@patch("agripipe.cli.AgriCleaner")
@patch("agripipe.cli.batch_load_raw")
@patch("torch.save")
def test_cli_run_input_dir(mock_save, mock_batch_load, mock_cleaner, runner, tmp_path):
    """Verifica che --input-dir chiami batch_load_raw."""
    input_dir = tmp_path / "data_dir"
    input_dir.mkdir()
    output_file = tmp_path / "out.pt"

    mock_cleaner.from_preset.return_value.clean.return_value = MagicMock()

    result = runner.invoke(
        app,
        [
            "run",
            "--input-dir",
            str(input_dir),
            "--output",
            str(output_file),
            "--preset",
            "ulivo_ligure",
        ],
    )

    assert result.exit_code == 0
    mock_batch_load.assert_called_once_with(Path(str(input_dir)), fuzzy=False)


@patch("agripipe.cli.AgriCleaner")
@patch("agripipe.cli.load_raw")
@patch("torch.save")
def test_cli_run_auto_units(mock_save, mock_load, mock_cleaner, runner, tmp_path):
    """Verifica che --auto-units modifichi la config del cleaner."""
    input_file = tmp_path / "test.xlsx"
    input_file.touch()
    output_file = tmp_path / "out.pt"

    cleaner_instance = mock_cleaner.from_preset.return_value
    cleaner_instance.clean.return_value = MagicMock()

    result = runner.invoke(
        app,
        [
            "run",
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--preset",
            "ulivo_ligure",
            "--auto-units",
        ],
    )

    assert result.exit_code == 0
    assert cleaner_instance.config.auto_unit_conversion is True


def test_cli_run_missing_config(runner, tmp_path):
    """Verifica errore su configurazione mancante."""
    p = tmp_path / "dummy.xlsx"
    p.touch()
    result = runner.invoke(app, ["run", "--input", str(p), "--output", "out.pt"])
    assert result.exit_code != 0
    assert "Configurazione mancante" in (result.stdout + result.stderr)


def test_cli_run_missing_input(runner, tmp_path):
    """Verifica errore su input mancante."""
    # Serve un config valido per superare il primo check
    cfg = tmp_path / "c.yaml"
    cfg.write_text("numeric_columns: []")
    result = runner.invoke(app, ["run", "--config", str(cfg), "--output", "out.pt"])
    assert result.exit_code != 0
    assert "Input mancante" in (result.stdout + result.stderr)
