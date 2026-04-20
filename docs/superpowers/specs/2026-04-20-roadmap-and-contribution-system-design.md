# AgriPipe — Roadmap & Contribution System Design

**Data**: 2026-04-20
**Status**: Approved, ready for implementation

---

## 1. Obiettivo

Costruire un sistema documentale a 3 touchpoint (ROADMAP narrativo + GitHub Issues operative + sezione di raccordo nel README) che sostenga tre job-to-be-done simultaneamente:

1. **Segnale professionale su LinkedIn** — decision-maker agritech leggono ROADMAP.md e percepiscono visione di prodotto + maturità tecnica.
2. **Incentivo ai contributor OSS** — potenziali contributor trovano task concrete e dimensionate su GitHub Issues.
3. **Memoria operativa** — l'autore e i curiosi di settore hanno un contesto continuativo di dove sta andando il progetto.

---

## 2. Non-obiettivi (esclusioni esplicite)

Questi punti sono **fuori scope** per evitare scope creep:

- **Non** creare `VISION.md`, `CONTRIBUTING.md`, o `TODO.md` separati. Un solo file narrativo + Issues.
- **Non** spostare la guida di dev setup da `docs/contributing.md` — già adeguata, solo da linkare.
- **Non** creare un task tracker interno (tipo TODO list): le Issues GitHub sono l'unico task tracker.
- **Non** impegnarsi su deadline precise negli orizzonti: orizzonte = intenzione di priorità, non commitment.

---

## 3. Architettura dei touchpoint

```
┌─────────────────────────────────────────────────────────────┐
│  README.md / README.en.md                                   │
│  └── Sezione nuova "🗺 Roadmap & contributi" (4-5 righe)    │
│      ├── Link → ROADMAP.md                                  │
│      └── Link → GitHub good-first-issues                    │
├─────────────────────────────────────────────────────────────┤
│  ROADMAP.md (nuovo, ~150 righe max)                         │
│  Narrativa strategica a 3 orizzonti, 0 task granulari       │
├─────────────────────────────────────────────────────────────┤
│  GitHub Issues                                              │
│  - 5 label standard                                         │
│  - Template issue standard (Cosa / Perché / Verifica)       │
│  - 4-6 issue iniziali derivate da Orizzonte 1               │
└─────────────────────────────────────────────────────────────┘
```

Responsabilità:
- **README**: discovery (il lettore capisce che esiste una roadmap, in 3 secondi)
- **ROADMAP.md**: convincere / ispirare (visione leggibile in 2 minuti)
- **Issues**: agire (un contributor può prendere in mano una task in 5 minuti)

---

## 4. Struttura di `ROADMAP.md`

### 4.1 Header
- Titolo `🗺 AgriPipe — Roadmap`
- Una frase di posizionamento (cos'è il progetto, dove sta andando in breve)

### 4.2 Orizzonti (3 sezioni parallele)

Ogni orizzonte ha:
- Titolo: `## 📍 Adesso — Orizzonte 0-3 mesi` / `🌱 Prossimo — 3-12 mesi` / `🔭 Visione — 12+ mesi`
- Una frase opinionata che dichiara il focus dell'orizzonte
- 2-4 item (per 1° e 2° orizzonte), 2-3 direzioni più narrative (per il 3°)

### 4.3 Struttura item (orizzonte 1 e 2)

```markdown
### [Titolo item, tipo prodotto non tipo task]
- **Cosa**: 1-2 frasi concrete
- **Perché per l'agritech**: 1-3 frasi di valore di business
- **Sforzo**: S / M / L
- **Issue**: #N (solo per orizzonte 1)
```

**Convenzioni di stile** (critiche per il job-to-be-done #1):
- Il "Perché" è **opinionato**, non descrittivo: contiene almeno una frase citabile fuori contesto
- L'item non si chiama "Aggiungere X" ma "X" (è un capability, non una task)
- Stima effort come taglia maglietta (S = giorni, M = settimana, L = >= mese)

### 4.4 Orizzonte 3 — Visione

Paragrafi di 3-5 righe per direzione, **senza stima di sforzo**. È ambizione, non pianificazione.

### 4.5 Sezione "🚫 Cosa NON è in roadmap"

Estensione della sezione "Limiti noti" del README. Elenca 3-5 cose che qualcuno potrebbe ragionevolmente chiedere ma che abbiamo deciso di non fare, con 1 frase di spiegazione ciascuna.

### 4.6 Sezione "🤝 Come contribuire"

Un paragrafo:
- Link alle good-first-issues
- Come aprire una proposta (nuova issue con label `feature`)
- Link a `docs/contributing.md` per il dev setup

### 4.7 Regola dimensionale

Il file intero sta **sotto 200 righe**. Oltre, è backlog — va su Issues.

---

## 5. Contenuto iniziale del ROADMAP (orizzonte 1)

Derivato dalla sezione "Limiti noti" del README, riformulato come value-driven:

1. **Fuzzy matching dei nomi colonna** — sforzo M
2. **Batch loading da cartella** — sforzo S
3. **Conversione automatica unità SI** — sforzo M

L'Orizzonte 2 e 3 verranno compilati con tre item ciascuno durante l'esecuzione, ispirati a:
- Orizzonte 2 (possibili): imputazione ML-based configurabile, dashboard di profiling pre-pipeline, integrazioni con formati sensor-native (JSON SDI-12, LoRaWAN payload)
- Orizzonte 3 (possibili): pipeline multi-azienda con consolidation semantica, modelli pre-addestrati come bundle add-on, active learning sui dati di campo

---

## 6. GitHub Issues — schema

### 6.1 Labels (5 totali)
- `good-first-issue` — scope piccolo, no pre-requisiti di dominio
- `feature` — nuova funzionalità
- `research` — richiede spike prima di implementazione
- `bug` — difetto confermato
- `docs` — documentazione

### 6.2 Template issue (in `.github/ISSUE_TEMPLATE/feature.md`)

```markdown
---
name: Feature proposal
about: Propose a new capability for AgriPipe
labels: feature
---

## Cosa
<!-- 1-3 frasi su cosa implementare -->

## Perché
<!-- 1-3 frasi sul valore di business, non tecnico -->

## Come verificare che è fatto
- [ ] criterio 1
- [ ] criterio 2
```

### 6.3 Issue iniziali (4-6)

Tre issue derivate dai 3 item dell'Orizzonte 1 (sopra), più 1-2 `good-first-issue` piccole:
- Aggiungere un preset regionale mancante (es. `pomodoro_siciliano`) — good-first-issue
- Aggiungere un test E2E su un Excel reale con nomi colonna non-canonici — good-first-issue

---

## 7. Integrazione README

In `README.md`, subito prima della sezione "Licenza", aggiungere:

```markdown
## 🗺 Roadmap & contributi

Dove sta andando AgriPipe: [`ROADMAP.md`](ROADMAP.md) — visione a 3 orizzonti.

Vuoi contribuire? Le task pronte da prendere in mano sono le
[good first issues](https://github.com/francesco5252/agripipe/labels/good-first-issue).
Per setup di sviluppo vedi [`docs/contributing.md`](docs/contributing.md).
```

Stesso blocco, tradotto, in `README.en.md`.

---

## 8. Success criteria

Il sistema è riuscito se, a un mese dall'implementazione:

- [ ] Un decision-maker agritech che atterra dalla repo trova la visione in < 2 minuti di lettura
- [ ] Un contributor curioso arriva a una issue "good first issue" chiara in < 3 click
- [ ] Il ROADMAP contiene almeno 2 frasi **citabili fuori contesto** (per post LinkedIn)
- [ ] Zero duplicazione: ogni informazione vive in un solo file
- [ ] Il sistema è mantenibile in ~15 minuti/mese (aggiornamento orizzonti + chiusura issue)

---

## 9. Deliverable elenco finale

1. `ROADMAP.md` nuovo (~150 righe), 3 orizzonti compilati, sezione "NON in roadmap", sezione "Come contribuire"
2. Sezione `🗺 Roadmap & contributi` aggiunta a `README.md` e `README.en.md`
3. `.github/ISSUE_TEMPLATE/feature.md` con il template standard
4. 5 label GitHub creati con descrizione
5. 5 issue iniziali create (3 feature + 2 good-first-issue), linkate dal ROADMAP
6. Commit con messaggio conventional `docs:` / `chore:` appropriato per ogni deliverable

---

## 10. Considerazione extra (differita)

L'utente ha esplicitamente deferito la valutazione del punto "voce editoriale per LinkedIn" (frasi citabili sistematiche in ogni item) a dopo l'implementazione di questo spec. Quando saremo in esecuzione, il ROADMAP andrà scritto applicando questo principio anche se non è formalmente in scope qui — è il modo in cui la voce viene calibrata nella scrittura, non una feature separata.
