# 🎯 MISSIONE: AgriPipe Pure ML-Ops Pipeline

Claude, questo progetto è stato rifondato per essere una **Pipeline ETL professionale e rigida**.
Dimentica l'agronomia interpretativa. Il focus è: **Dato Sporco -> Raffineria -> Tensor PyTorch.**

## 🏗️ Struttura Integrale del Progetto
Il workflow è diviso in tre fasi obbligatorie:
1. **LOADER**: Caricamento dati grezzi (Excel/CSV) con errori realistici. Rigore sullo schema.
2. **CLEANER**: Pulizia automatica delle anomalie e gestione dati mancanti (Statistica pura).
3. **TENSORIZER**: Conversione nel formato tensoriale rigido richiesto da PyTorch (Scaling + Metadata).

---

## 🛠️ STATO ATTUALE (Gemini CLI ha già fatto questo)
- ✅ Rimossi tutti i moduli "sustainability" e "indices".
- ✅ Semplificato `AgriCleaner`: ora fa solo pulizia statistica e fisica di base.
- ✅ Evoluto `export.py`: ora genera un bundle completo (.pt + .json + .zip).
- ✅ Aggiornato `README.md` con la visione tecnica corretta.
- ✅ Creato `data/sample/pro_demo.xlsx` per testare la pipeline con dati complessi.

---

## 🚀 I TUOI TASK PRIORITARI (Claude Code)
Analizza il progetto e procedi con:
1. **Aggiornamento UI (`app.py`)**: 
   - Elimina i badge di sostenibilità.
   - Crea una dashboard che mostri le metriche di **Integrità del Dato** (es. % NaN riempiti, outlier rimossi, shape del tensor finale).
   - **Quantum-Level**: Mostra quante "**Anomalie Peer**" sono state rilevate (confronto tra campi diversi).
   - **Quantum-Level**: Visualizza un avviso di "**Target Bias**" se la resa è troppo uniforme (rischio per il training ML).
   - **Apex-Level**: Mostra un avviso se Gemini rileva "**Colonne Sbilanciate**" (Skewed).
   - **Apex-Level**: Visualizza il conteggio degli "**Outlier Stagionali**" catturati.
   - Visualizza i plot prima/dopo usando il componente `render_before_after_plots`.
   - **Data Science**: Mostra una **Heatmap di Correlazione** (i dati sono in `correlations` nel JSON) e un sommario delle statistiche descrittive (mean, std, min, max).
   - **Production Master**: Mostra quali colonne sono state rimosse perché ridondanti (varianza zero o duplicate).
   - **Production Master**: Segnala se è stata applicata una **Trasformazione Logaritmica** per curare lo skewness.
   - **Batch Power**: Permetti all'utente di selezionare una **Cartella** invece di un singolo file. Spiega che Gemini unirà e deduplicherà tutto automaticamente.
   4. **Flusso E2E**: Garantisci che il passaggio tra Fase 1, 2 e 3 sia fluido nell'app Streamlit.
   **OPERA FINITA**: Il motore tecnico è al 100% della sua potenza. Passaggio a Claude per il capolavoro finale. Ogni fase è ora ai massimi standard industriali.
