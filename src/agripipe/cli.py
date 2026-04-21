"""CLI: `agripipe run`, `agripipe generate`, `agripipe report`."""

from __future__ import annotations

from pathlib import Path

import torch
import typer

from agripipe.cleaner import AgriCleaner
from agripipe.dataset import AgriDataset
from agripipe.export import export_ml_bundle
from agripipe.loader import batch_load_raw, load_raw
from agripipe.report import generate_report
from agripipe.synth import SynthConfig, generate_dirty_excel
from agripipe.utils.logging_setup import get_logger

app = typer.Typer(help="AgriPipe Pro: Excel agronomici -> ML Bundles & Reports.")
logger = get_logger(__name__)


@app.command()
def run(
    input: Path | None = typer.Option(
        None, "--input", "-i", exists=True, help="File .xlsx di input."
    ),
    input_dir: Path | None = typer.Option(
        None, "--input-dir", "-d", exists=True, file_okay=False, help="Cartella con file Excel/CSV."
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Path .pt di output."),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="YAML di configurazione (opzionale se si usa --preset)."
    ),
    preset: str | None = typer.Option(
        None, "--preset", "-p", help="Usa un preset regionale (es: ulivo_ligure)."
    ),
    target: str = typer.Option("yield", "--target", "-t", help="Colonna target."),
    report: Path | None = typer.Option(
        None, "--report", "-r", help="Path HTML del report (opzionale)."
    ),
    export_ml: Path | None = typer.Option(
        None, "--export-ml", "-e", help="Esporta bundle ML completo (.zip) in questa cartella."
    ),
    fuzzy: bool = typer.Option(False, "--fuzzy", help="Abilita fuzzy matching dei nomi colonna."),
    auto_units: bool = typer.Option(
        False, "--auto-units", help="Abilita conversione automatica unità SI (F, inch, lb/acre)."
    ),
) -> None:
    """Esegue l'intera pipeline: load -> clean -> tensorize -> save/export."""
    try:
        logger.info("=== AgriPipe run ===")

        if preset:
            logger.info("Caricamento preset: %s", preset)
            cleaner = AgriCleaner.from_preset(preset)
            typer.secho(f"✓ Preset regionale: {preset}", fg=typer.colors.CYAN)
        elif config:
            if not config.exists():
                typer.secho(
                    f"❌ Configurazione non trovata: {config}", fg=typer.colors.RED, err=True
                )
                raise typer.Exit(code=1)
            cleaner = AgriCleaner.from_yaml(config)
        else:
            typer.secho(
                "❌ Configurazione mancante: fornire --config <path> o --preset <nome>.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Applichiamo auto_units alla config del cleaner se richiesto via CLI
        if auto_units:
            cleaner.config.auto_unit_conversion = True

        if input_dir:
            df_raw = batch_load_raw(input_dir, fuzzy=fuzzy)
        elif input:
            df_raw = load_raw(input, fuzzy=fuzzy)
        else:
            typer.secho(
                "❌ Input mancante: fornire --input <file> o --input-dir <cartella>.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        df_clean = cleaner.clean(df_raw)

        if target not in df_clean.columns:
            logger.warning("Colonna target '%s' non trovata. Generazione solo features.", target)
            target_col = None
        else:
            target_col = target

        ds = AgriDataset(
            df_clean,
            numeric_columns=cleaner.config.numeric_columns,
            categorical_columns=cleaner.config.categorical_columns,
            target=target_col,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"features": ds.features, "target": ds.target, "feature_names": ds.feature_names},
            output,
        )
        typer.secho(
            f"✓ Tensor salvati in {output} (shape={tuple(ds.features.shape)})",
            fg=typer.colors.GREEN,
        )

        if export_ml:
            # Recuperiamo il preset dict per il metadata
            preset_dict = cleaner.knowledge.get("regional_presets", {}).get(preset or "", {})
            paths = export_ml_bundle(df_clean, cleaner, preset_dict, export_ml, target=target_col)
            typer.secho(f"✓ Bundle ML esportato in {paths['zip']}", fg=typer.colors.GREEN)

        if report:
            generate_report(df_raw, df_clean, report)
            typer.secho(f"✓ Report salvato in {report}", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"❌ Errore durante l'esecuzione: {str(e)}", fg=typer.colors.RED, err=True)
        if logger.getEffectiveLevel() <= 10:  # DEBUG
            raise e
        raise typer.Exit(code=1)


@app.command()
def check(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
) -> None:
    """Valida la sintassi del file di configurazione YAML."""
    try:
        AgriCleaner.from_yaml(config)
        typer.secho(f"✓ Configurazione {config} valida.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"❌ Configurazione non valida: {str(e)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


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
    config: Path | None = typer.Option(None, "--config", "-c"),
    preset: str | None = typer.Option(None, "--preset", "-p"),
) -> None:
    """Genera solo il report HTML di qualità (senza salvare tensor)."""
    if preset:
        cleaner = AgriCleaner.from_preset(preset)
    elif config:
        cleaner = AgriCleaner.from_yaml(config)
    else:
        typer.secho("❌ Fornire --config o --preset.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    df_raw = load_raw(input)
    df_clean = cleaner.clean(df_raw)
    generate_report(df_raw, df_clean, output)
    typer.echo(f"✓ Report: {output}")


@app.command()
def list_presets(
    region: str | None = typer.Option(None, "--region", "-r", help="Filtra per regione."),
) -> None:
    """Elenca tutti i preset dell'Atlante Agronomico Italiano."""
    try:
        # Carichiamo un cleaner temporaneo per accedere alla knowledge
        cleaner = AgriCleaner(AgriCleaner.from_preset("vite_nebbiolo_barolo").config)
        presets = cleaner.knowledge.get("regional_presets", {})

        if not presets:
            typer.secho("❌ Nessun preset trovato nell'Atlante.", fg=typer.colors.RED)
            return

        typer.secho("\n🇮🇹  ATLANTE AGRONOMICO ITALIANO - AgriPipe\n", fg=typer.colors.GREEN, bold=True)
        
        # Raggruppiamo per regione
        by_region = {}
        for name, data in presets.items():
            reg = data.get("region", "Altro")
            if region and region.lower() not in reg.lower():
                continue
            if reg not in by_region:
                by_region[reg] = []
            by_region[reg].append((name, data.get("crop_display", "N/A"), data.get("zona", "Generica")))

        for reg in sorted(by_region.keys()):
            typer.secho(f"📍 {reg.upper()}", fg=typer.colors.CYAN, bold=True)
            for name, crop, zona in sorted(by_region[reg]):
                typer.echo(f"  • {name:30} | {crop:30} | {zona}")
            typer.echo("")

    except Exception as e:
        typer.secho(f"❌ Errore nel caricamento dell'Atlante: {str(e)}", fg=typer.colors.RED)


@app.command()
def version() -> None:
    """Mostra la versione."""
    from agripipe import __version__

    typer.echo(f"agripipe {__version__}")


if __name__ == "__main__":
    app()
