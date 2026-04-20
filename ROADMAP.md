# 🗺️ AgriPipe — Roadmap

> AgriPipe nasce come ponte affidabile fra Excel agronomico reale e tensor PyTorch validati. La direzione nei prossimi mesi è ridurre l'attrito all'adozione in contesti agritech di produzione: meno lavoro manuale richiesto, più controlli automatici, più formati supportati.
>
> Questo documento è un **living document**: evolve in base al feedback degli operatori agritech reali, ai limiti che emergono nell'uso, e alle opportunità che si aprono lungo il percorso. Gli orizzonti sono scritti a **livelli decrescenti di specificità** — più vicini nel tempo, più concreti; più lontani, più tematici.      

---

## 🎯 Adesso — Orizzonte 0-3 mesi

> Focus: far girare AgriPipe su un Excel agronomico reale trovato in rete senza chiedere all'utente di rinominare colonne o convertire unità a mano.

### Fuzzy matching dei nomi colonna
- **Cosa**: riconoscere automaticamente varianti come `Temperatura_C`, `Temp °C`, `temperature`, `t_celsius` come la colonna canonica `temp`, via string similarity + dizionario italiano-inglese agronomico.
- **Perché per l'agritech**: chi riceve dati dai clienti non ha controllo sui nomi colonna. Forzare la rinomina manuale prima di ogni esecuzione è friction che in produzione uccide l'adozione.
- **Sforzo**: M

### Batch loading da cartella
- **Cosa**: `agripipe run --input-dir ./daily_exports/` processa tutti gli Excel di una cartella producendo un bundle consolidato e un report di qualità aggregato.
- **Perché per l'agritech**: gli operatori ricevono un Excel al giorno per azienda cliente. Chiamare la CLI una volta per file non è un workflow credibile in produzione — serve il batch.
- **Sforzo**: S

### Conversione automatica unità SI
- **Cosa**: rilevare e convertire Fahrenheit → Celsius, inch → mm, lb/acre → kg/ha in base a suffissi nel nome colonna o a range numerici fuori norma.
- **Perché per l'agritech**: dati da sensori americani o macchinari misti sono frequenti. Sbagliare un'unità è un bug silenzioso che scopri solo in fase di inferenza — il tipo di bug più costoso da rintracciare in un modello ML.
- **Sforzo**: M

---

## 🌱 Prossimo — Orizzonte 3-12 mesi

> Focus: alzare la qualità del dato consegnato al modello ML e ridurre il tempo che separa "bundle pronto" da "modello validato".
>
> Gli item concreti di questo orizzonte **non sono fissati**: verranno selezionati mese per mese in base ai segnali raccolti (feedback di operatori agritech reali, limiti emersi nell'uso, opportunità di contributo esterno). Le tre direzioni sotto sono i **temi** prioritari oggi — le singole feature matureranno dalla conversazione con gli utenti.

### Qualità statistica e imputazione avanzata

L'attuale pipeline usa imputazioni classiche (media, mediana, ffill, interpolazione temporale). In presenza di missingness elevata o struttura multivariata complessa, questi metodi distorcono le correlazioni fra variabili e penalizzano il modello ML finale. Tecniche più sofisticate — KNN, MICE, imputazione basata su modelli ausiliari — sono la direzione naturale di miglioramento quando i dati lo richiedono.

### Feedback anticipato all'utente

Oggi l'utente scopre la qualità del proprio dato solo dopo aver lanciato la pipeline, leggendo i diagnostics a valle. Un profiling pre-pipeline (distribuzioni, anomalie sospette, missingness per colonna) gli permetterebbe di calibrare il Cleaner con cognizione di causa **prima** di investire il tempo di processing — riducendo il numero di tentativi in cui si "scopre" che il dataset ha un problema a metà della pipeline.

### Chiusura del loop dato → modello

AgriPipe produce tensor pronti per PyTorch, ma lascia l'utente da solo a dimostrare che i suoi dati sono effettivamente predittivi. Un percorso opzionale di baseline modeling (modelli statistici o alberi decisionali, non reti neurali) permetterebbe di validare il segnale del dataset prima di investire tempo in architetture ML custom. Se un baseline semplice non predice nulla, il problema non è nel modello — è nei dati.

---

## 🔭 Visione — Orizzonte 12+ mesi

> Focus: trasformare AgriPipe da tool single-tenant a piattaforma leggera multi-cliente per operatori agritech che gestiscono portafogli agronomici.

### Pipeline multi-azienda con consolidation semantica

Un operatore che segue 50 aziende agricole oggi dovrebbe gestire 50 pipeline separate. La versione multi-tenant riconoscerebbe dataset compatibili fra aziende simili (stessa coltura, stesso regime irriguo) e proporrebbe modelli federati — ogni azienda beneficia dell'esperienza aggregata senza condividere dati grezzi.

### Marketplace di preset agronomici community-driven

Oggi i preset (`ulivo_pugliese`, `vite_piemontese`) vivono nel codice. Un marketplace permetterebbe a tecnici agronomi di contribuire preset per colture, regioni o tecniche specifiche (biologico certificato, idroponica, serre climatizzate). Il valore cresce con il network effect.

### Active learning sui dati di campo

L'agronomo oggi usa AgriPipe ma il feedback sulla qualità del modello resta fuori dal tool. Un sistema di active learning raccoglierebbe i casi dove il modello sbaglia, li segnalerebbe per ri-etichettatura in campo, e userebbe quei dati per affinare i modelli successivi. Chiudere il loop dato → modello → decisione → feedback.  

---

## 🚫 Cosa NON è in roadmap (e perché)

Alcune richieste ragionevoli che abbiamo deciso consapevolmente di NON sviluppare:

- **Interfaccia grafica no-code completa (stile CMS)** — la UI Streamlit è e resterà l'MVP. Chi vuole un prodotto SaaS costruisce sopra AgriPipe via CLI/API, non dentro Streamlit.
- **Modelli agronomici interpretativi** (indici di sostenibilità, scorecard green/yellow/red) — scelta di design: AgriPipe produce dati, non giudizi. Separare data prep da interpretazione mantiene la pipeline scientificamente verificabile.
- **Integrazione diretta con ERP agricoli proprietari** — ogni ERP ha la sua API. L'integrazione è scope di business, non di tool open-source. Chi la vuole scrive un connettore sopra la CLI.
- **Unit test su ogni funzione privata** — si testa il comportamento osservabile (input → output), non i dettagli interni. Over-testing rallenta il refactor.

---

## 🤝 Come contribuire

Le task concrete pronte da prendere in mano sono le [good first issues](https://github.com/francesco5252/agripipe/labels/good-first-issue) — scope contenuto, nessun pre-requisito di dominio agronomico.

Per proporre una nuova feature, [apri una issue](https://github.com/francesco5252/agripipe/issues/new?template=feature.md) con label `feature` rispettando il template standard.

Per il setup di sviluppo in locale, vedi [`docs/contributing.md`](docs/contributing.md).
