"""Sustainability Score Card: 4 badge sintetici di sostenibilità agronomica.

Partendo da una diagnostica popolata da ``AgriCleaner``, calcola quattro
indicatori con semantica a semaforo (verde / arancio / rosso).

Soglie di riferimento:
    - **Azoto**: percentuale di righe con violazione Direttiva Nitrati 91/676/CEE.
      Verde <3% | Arancio 3-8% | Rosso >8%.
    - **Peronospora**: conteggio assoluto di eventi "Regola dei Tre 10"
      (temp >10°C, rain >10mm). Verde 0-2 | Arancio 3-4 | Rosso >=5.
    - **Irrigazione**: percentuale di eventi irrigui inefficienti su suolo saturo.
      Verde <5% | Arancio 5-15% | Rosso >15%.
    - **Suolo**: percentuale di letture con sostanza organica <1.5%
      (riferimento FAO per suoli mediterranei degradati).
      Verde <10% | Arancio 10-30% | Rosso >30%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agripipe.cleaner import CleanerDiagnostics

Level = Literal["green", "orange", "red"]

_LEVEL_ICON: dict[str, str] = {
    "green": "\U0001f7e2",  # pallino verde
    "orange": "\U0001f7e0",  # pallino arancio
    "red": "\U0001f534",  # pallino rosso
}


@dataclass(frozen=True)
class Badge:
    """Indicatore sintetico di sostenibilità agronomica.

    Attributes:
        name: Identificatore tecnico (``"azoto"``, ``"peronospora"``,
            ``"irrigazione"``, ``"suolo"``).
        level: Livello di conformità (``"green"``, ``"orange"``, ``"red"``).
        icon: Pallino colorato corrispondente al livello.
        headline: Descrizione sintetica del risultato.
        value: Metrica numerica (percentuale 0-100 o conteggio assoluto).
        tip: Raccomandazione agronomica operativa.
    """

    name: str
    level: Level
    icon: str
    headline: str
    value: float
    tip: str


def _pct(count: int, total: int) -> float:
    return (count / total * 100.0) if total > 0 else 0.0


def _azoto_badge(violations: int, total: int) -> Badge:
    pct = _pct(violations, total)
    if pct < 3.0:
        level: Level = "green"
        headline = "Concimazione azotata conforme"
        tip = "Mantenere il regime attuale di applicazione"
    elif pct <= 8.0:
        level = "orange"
        headline = "Non-conformità marginale"
        tip = "Verificare tempistica applicazioni rispetto a pioggia e umidità del suolo"
    else:
        level = "red"
        headline = "Non-conformità sistemica Direttiva Nitrati"
        tip = "Ridurre dosi; calibrare su analisi fogliare e frazionare gli apporti"
    return Badge(
        name="azoto",
        level=level,
        icon=_LEVEL_ICON[level],
        headline=headline,
        value=round(pct, 2),
        tip=tip,
    )


def _peronospora_badge(events: int) -> Badge:
    if events <= 2:
        level: Level = "green"
        headline = "Nessun evento critico rilevato"
        tip = "Mantenere monitoraggio meteorologico"
    elif events <= 4:
        level = "orange"
        headline = "Soglia di intervento raggiunta"
        tip = "Pianificare trattamento preventivo (rame o fosfiti) entro 48h"
    else:
        level = "red"
        headline = "Infezione probabile in corso"
        tip = "Trattamento curativo immediato; valutazione fitosanitaria in campo"
    return Badge(
        name="peronospora",
        level=level,
        icon=_LEVEL_ICON[level],
        headline=headline,
        value=float(events),
        tip=tip,
    )


def _irrigazione_badge(inefficient: int, total: int) -> Badge:
    pct = _pct(inefficient, total)
    if pct < 5.0:
        level: Level = "green"
        headline = "Uso idrico efficiente"
        tip = "Gestione conforme ai benchmark di precision irrigation"
    elif pct <= 15.0:
        level = "orange"
        headline = "Inefficienza idrica rilevabile"
        tip = "Ricalibrare sensori umidità e soglie di attivazione"
    else:
        level = "red"
        headline = "Spreco idrico sistemico"
        tip = "Audit impianto e revisione del piano irriguo"
    return Badge(
        name="irrigazione",
        level=level,
        icon=_LEVEL_ICON[level],
        headline=headline,
        value=round(pct, 2),
        tip=tip,
    )


def _suolo_badge(low: int, total: int) -> Badge:
    pct = _pct(low, total)
    if pct < 10.0:
        level: Level = "green"
        headline = "Sostanza organica in range"
        tip = "Mantenere pratiche di gestione attuali"
    elif pct <= 30.0:
        level = "orange"
        headline = "Degrado parziale sostanza organica"
        tip = "Introdurre cover crop e apporti di compost"
    else:
        level = "red"
        headline = "Degrado sistemico sostanza organica"
        tip = "Piano rigenerativo: sovescio, letame, rotazione pluriennale"
    return Badge(
        name="suolo",
        level=level,
        icon=_LEVEL_ICON[level],
        headline=headline,
        value=round(pct, 2),
        tip=tip,
    )


def compute_scorecard(
    diagnostics: CleanerDiagnostics,
    total_rows: int,
) -> list[Badge]:
    """Calcola i 4 badge di sostenibilità da una diagnostica del cleaner.

    Args:
        diagnostics: Oggetto popolato da ``AgriCleaner.clean``.
        total_rows: Numero totale di righe del dataset analizzato.
            Usato come denominatore per i badge percentuali (azoto,
            irrigazione, suolo). Ignorato per peronospora.

    Returns:
        Lista ordinata dei 4 badge:
        ``[azoto, peronospora, irrigazione, suolo]``.
    """
    return [
        _azoto_badge(diagnostics.nitrogen_violations, total_rows),
        _peronospora_badge(diagnostics.peronospora_events),
        _irrigazione_badge(diagnostics.irrigation_inefficient, total_rows),
        _suolo_badge(diagnostics.soil_organic_low, total_rows),
    ]


def overall_message(badges: list[Badge]) -> str:
    """Sintetizza lo stato complessivo dai 4 badge.

    Args:
        badges: Lista di badge prodotta da ``compute_scorecard``.

    Returns:
        Messaggio testuale aggregato (es. "Criticita rilevata").
    """
    reds = sum(1 for b in badges if b.level == "red")
    oranges = sum(1 for b in badges if b.level == "orange")
    if reds >= 2:
        return "Criticità sistemiche: revisione protocolli necessaria"
    if reds == 1:
        return "Criticità rilevata: intervento raccomandato"
    if oranges >= 1:
        return "Gestione accettabile con aree di miglioramento"
    return "Gestione conforme agli standard di sostenibilità"
