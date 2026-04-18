# 📝 Handoff per Claude Code: Transizione a Pure ML-Ops Pipeline

Claude, l'architettura di AgriPipe è stata semplificata radicalmente per focalizzarsi esclusivamente sulla qualità del dato e sulla tensorizzazione per PyTorch. **Tutta la logica di interpretazione agronomica e sostenibilità è stata rimossa.**

### 🚀 Cambiamenti Chiave:
1.  **Moduli Rimossi**: `indices.py` e `sustainability.py` (e relativi test) sono stati eliminati.
2.  **Cleaner Semplificato**: `AgriCleaner` non applica più regole agronomiche. Ora esegue solo pulizia statistica:
    *   Coercizione tipi.
    *   Rimozione Outlier (IQR/Z-Score).
    *   Imputazione (Mean, Median, Time-series interpolation).
    *   Limiti Fisici (solo per evitare dati numericamente assurdi).
3.  **Metadata Puro**: Il file `metadata.json` ora contiene solo info tecniche sul dataset e statistiche di pulizia, senza interpretazioni "green/red".
4.  **UI Core**: I componenti UI sono stati ripuliti dai badge di sostenibilità. La nuova UI deve concentrarsi su:
    *   Visualizzazione distribuzioni (prima/dopo).
    *   Shape dei tensor.
    *   Download dei bundle ML.

### 🛠 Task per te (Claude):
- **UI Integration**: Aggiorna `app.py` per riflettere questo approccio tecnico. Mostra metriche di integrità del dato (es. "% dati imputati") invece di badge di sostenibilità.
- **Robustezza**: Assicurati che l'app Streamlit gestisca correttamente il nuovo `export_ml_bundle` (che restituisce Path a .pt, .json, .zip).

**Stato attuale**: 36 test passati, ambiente pulito, visione 100% ML-Ops.
