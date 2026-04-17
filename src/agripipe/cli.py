"""CLI: `agripipe run`, `agripipe generate`, `agripipe report`."""

from __future__ import annotations

from pathlib import Path

import torch
import typer

from agripipe.cleaner import AgriCleaner
from agripipe.dataset import AgriDataset
from agripipe.loader import load_raw
from agripipe.report import generate_report
from agripipe.synth import SynthConfig, generate_dirty_excel
from agripipe.utils.logging_setup import get_logger

app = typer.Typer(help="AgriPipe: Excel agronomici → tensor PyTorch.")
logger = get_logger(__name__)


@app.command()
def run(
    input: Path = typer.Option(..., "--input", "-i", exists=True, help="File .xlsx di input."),
    output: Path = typer.Option(..., "--output", "-o", help="Path .pt di output."),
    config: Path = typer.Option(
        Path("configs/default.yaml"), "--config", "-c", help="YAML di configurazione."
    ),
    target: str = typer.Option("yield", "--target", "-t", help="Colonna target."),
    report: Path | None = typer.Option(
        None, "--report", "-r", help="Path HTML del report (opzionale)."
    ),
) -> None:
    """Esegue l'intera pipeline: load → clean → tensorize → save."""
    logger.info("=== AgriPipe run ===")

    df_raw = load_raw(input)
    cleaner = AgriCleaner.from_yaml(config)
    df_clean = cleaner.clean(df_raw)

    ds = AgriDataset(
        df_clean,
        numeric_columns=cleaner.config.numeric_columns,
        categorical_columns=cleaner.config.categorical_columns,
        target=target if target in df_clean.columns else None,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"features": ds.features, "target": ds.target, "feature_names": ds.feature_names},
        output,
    )
    logger.info("Tensor salvati in %s (shape=%s)", output, tuple(ds.features.shape))

    if report:
        generate_report(df_raw, df_clean, report)


@app.command()
def generate(
    output: Path = typer.Option(Path("data/sample/synthetic_dirty.xlsx"), "--output", "-o"),
    rows: int = typer.Option(500, "--rows", "-n", help="Numero righe base."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
) -> None:
    """Genera un Excel sintetico con anomalie realistiche per test."""
    cfg = SynthConfig(n_rows=rows, seed=seed)
    generate_dirty_excel(output, cfg)
    typer.echo(f"✓ Generato: {output}")


@app.command()
def report(
    input: Path = typer.Option(..., "--input", "-i", exists=True, help="Excel grezzo."),
    output: Path = typer.Option(Path("out/report.html"), "--output", "-o"),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
) -> None:
    """Genera solo il report HTML di qualità (senza salvare tensor)."""
    df_raw = load_raw(input)
    df_clean = AgriCleaner.from_yaml(config).clean(df_raw)
    generate_report(df_raw, df_clean, output)
    typer.echo(f"✓ Report: {output}")


@app.command()
def version() -> None:
    """Mostra la versione."""
    from agripipe import __version__

    typer.echo(f"agripipe {__version__}")


if __name__ == "__main__":
    app()
