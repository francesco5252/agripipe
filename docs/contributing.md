# Sviluppo e Contributi

Se vuoi contribuire ad AgriPipe o eseguire test in locale, segui queste istruzioni.

## Setup Ambiente

1. Crea un virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # o venv\Scripts\activate su Windows
   ```

2. Installa le dipendenze di sviluppo:
   ```bash
   pip install -e ".[dev]"
   ```

3. Installa i pre-commit hook:
   ```bash
   pre-commit install
   ```

## Test

Esegui la suite di test completa con copertura:
```bash
pytest
```

## Documentazione

Per visualizzare la documentazione localmente mentre la modifichi:
```bash
mkdocs serve
```
La documentazione sarà disponibile all'indirizzo `http://127.0.0.1:8000`.
