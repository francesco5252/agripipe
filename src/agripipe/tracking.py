"""Modulo di MLOps: Tracciamento degli artefatti e delle basi agronomiche."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_export_run(
    config_dict: dict,
    diag_dict: dict,
    file_name: str,
    target: str | None,
    split_ratios: tuple[float, float, float] | None,
    baseline_metrics: dict[str, float],
) -> None:
    """Registra i parametri di pulizia e i baseline metrics su MLflow."""
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow non installato. Salto il tracking.")
        return

    mlflow.set_experiment("agripipe-data-refinery")
    with mlflow.start_run():
        mlflow.set_tag("file_name", file_name)
        if target:
            mlflow.set_tag("target", target)

        # Traccia parametri configurazione (solo primitive)
        for k, v in config_dict.items():
            if isinstance(v, (int, float, str, bool)):
                mlflow.log_param(f"cfg_{k}", v)

        # Traccia metriche pulizia
        for k, v in diag_dict.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"obs_{k}", float(v))

        if split_ratios:
            mlflow.log_param("split_train", split_ratios[0])
            mlflow.log_param("split_val", split_ratios[1])
            mlflow.log_param("split_test", split_ratios[2])

        # Traccia Baseline model
        for k, v in baseline_metrics.items():
            mlflow.log_metric(f"model_{k}", v)

    logger.info("Run MLflow registrato correttamente per %s", file_name)
