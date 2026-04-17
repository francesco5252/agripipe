# 📝 Diario di Sviluppo: AgriPipe (Agritech Data Pipeline)

Questo documento traccia l'evoluzione di **AgriPipe**, un tool automatico progettato per ottimizzare i flussi di lavoro in un'azienda Agritech (X Farm), trasformando dati agricoli grezzi in tensori pronti per il Machine Learning.

---

## 🎯 Obiettivo del Progetto
Creare un "ponte" tra l'agronomia di campo e l'intelligenza artificiale, eliminando la pulizia manuale dei dati e garantendo la coerenza biologica dei dataset.

---

## 🛠 Stato del Progetto (Punti di Forza Attuali)

### 1. Caricamento e Validazione (Fase 1)
- **Input Excel/CSV**: Gestione sicura di file "sporchi" con errori realistici.
- **Schema Rigido**: Utilizzo di validazione automatica (Pydantic) per evitare crash del sistema.

### 2. Il "Cervello Agronomico" (Fase 2 - Core Intelligence)
*Il vero vantaggio competitivo del progetto: non solo matematica, ma biologia.*
- **Dizionario Italiano DOC**: Regole specifiche per eccellenze italiane (Vite DOCG, Grano Duro, Agrumi, Mele Trentine, Kiwi, Zafferano).
- **Controllo Sostenibilità**: Rilevamento sprechi di concime (Azoto) e inefficienza idrica (irrigazione durante la pioggia).
- **Sentinella dei Rischi**: Allerta automatica per Gelate Tardive, Stress Idrico, Stress da Calore e Malattie (Regola dei "Tre 10" per la Peronospora).
- **Salute del Suolo**: Monitoraggio della sostanza organica per prevenire la desertificazione.

### 3. Feature Engineering & Indici (Fase 3)
*Trasformazione di dati grezzi in informazioni preziose per l'IA.*
- **GDD (Gradi Giorno)**: Calcolo automatico del calore accumulato per prevedere fioritura e raccolta.
- **Indice di Huglin**: Specifico per la qualità vitivinicola italiana.
- **Drought Score**: Punteggio di siccità cumulata sugli ultimi 7 giorni.
- **N-Efficiency**: Misura della sostenibilità economica e ambientale della concimazione.

### 4. Visualizzazione e Reporting
- **Report Narrativo**: Generazione di file HTML standalone (facili da condividere).
- **Grafici Automatici**: Boxplot e Distribuzioni che mostrano il "Prima vs Dopo" della pulizia.

---

## 🚀 Log Sviluppi Futuri (Da questo momento in poi)

> *In questa sezione segneremo ogni nuova modifica focalizzata su UI/UX, Esportabilità e Perfezionamento.*

### [2024-04-17] - Focus: Visualizzazione Avanzata e UX Dashboard
- **Obiettivo**: Rendere immediata la percezione del valore aggiunto dalla pulizia agronomica.
- **Risultato**: 
    - Integrazione di **Matplotlib e Seaborn** direttamente nella UI Streamlit.
    - Creazione di una sezione "Analisi Visiva della Qualità" con grafici a confronto (Grezzo vs Ottimizzato).
    - Utilizzo di **Boxplot** per evidenziare la rimozione di outlier territoriali.
    - Utilizzo di **KDE Plots** per mostrare il miglioramento della distribuzione dei dati.
    - Implementazione di sezioni espandibili per non sovraccaricare l'utente (UX pulita).
- **Prossimo Step**: Chiusura del cerchio con un esempio di modello IA (Machine Learning) che utilizza i tensori prodotti.

---

## 💡 Note Agronomiche per il README
*Questi punti evidenziano la tua competenza in Agricoltura Sostenibile:*
- Il software applica la **Direttiva Nitrati** controllando i kg/ha di Azoto.
- La gestione del **Fabbisogno in Freddo** permette di adattarsi ai cambiamenti climatici.
- La coerenza **Pioggia-Umidità** identifica sensori IoT guasti senza bisogno di sopralluoghi.
