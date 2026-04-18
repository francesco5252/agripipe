# 🌱 AgriPipe

Benvenuto nella documentazione ufficiale di **AgriPipe**.

AgriPipe è una libreria Python progettata per automatizzare il processo di trasformazione di dati agronomici grezzi (spesso salvati in file Excel "sporchi") in tensori PyTorch pronti per l'addestramento di modelli di Machine Learning.

## Caratteristiche Principali

- **Validazione Automatica**: Controllo rigoroso dei tipi e dei range tramite Pydantic.
- **Pulizia Dati Intelligente**: Gestione automatica di outlier (IQR, Z-Score) e valori mancanti.
- **Tensorizzazione**: Conversione immediata in `torch.Tensor` con scaling e encoding persistibili.
- **Interfaccia CLI**: Gestisci l'intera pipeline senza scrivere codice Python.

## Installazione

Per installare AgriPipe in modalità sviluppo:

```bash
git clone https://github.com/yourusername/agripipe.git
cd agripipe
pip install -e ".[dev]"
```

## Prossimi Passi

- Consulta il [Quickstart](getting-started.md) per eseguire la tua prima pipeline.
- Scopri come personalizzare la pulizia dei dati nella sezione [Configurazione](configuration.md).
