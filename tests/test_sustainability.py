"""Test della Sustainability Score Card.

Verifica:
- forma della dataclass Badge (frozen, icona coerente al livello);
- soglie di classificazione dei 4 badge (azoto, peronospora, irrigazione, suolo);
- logica di aggregazione `overall_message`.

Le soglie sono giustificate agronomicamente nel modulo `sustainability`.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agripipe.cleaner import CleanerDiagnostics
from agripipe.sustainability import Badge, compute_scorecard, overall_message

# ---------- Badge dataclass ----------


def test_badge_is_frozen_dataclass():
    b = Badge(
        name="azoto",
        level="green",
        icon="\U0001f7e2",  # pallino verde
        headline="headline",
        value=0.0,
        tip="tip",
    )
    with pytest.raises(FrozenInstanceError):
        b.level = "red"  # type: ignore[misc]


@pytest.mark.parametrize(
    "level, expected_icon",
    [
        ("green", "\U0001f7e2"),
        ("orange", "\U0001f7e0"),
        ("red", "\U0001f534"),
    ],
)
def test_badge_icon_matches_level(level, expected_icon):
    diag = CleanerDiagnostics()
    # Forza il livello target tramite i conteggi.
    if level == "green":
        diag.nitrogen_violations = 0
    elif level == "orange":
        diag.nitrogen_violations = 5  # 5% => orange (3<=pct<=8)
    else:
        diag.nitrogen_violations = 20  # 20% => red (>8)
    badges = compute_scorecard(diag, total_rows=100)
    azoto = badges[0]
    assert azoto.level == level
    assert azoto.icon == expected_icon


# ---------- compute_scorecard contract ----------


def test_compute_scorecard_returns_four_badges():
    diag = CleanerDiagnostics()
    badges = compute_scorecard(diag, total_rows=100)
    assert len(badges) == 4
    assert [b.name for b in badges] == [
        "azoto",
        "peronospora",
        "irrigazione",
        "suolo",
    ]


# ---------- Azoto thresholds (3 / 8 percent) ----------


def test_azoto_green_under_3_percent():
    diag = CleanerDiagnostics(nitrogen_violations=2)
    badge = compute_scorecard(diag, total_rows=100)[0]
    assert badge.level == "green"


def test_azoto_orange_between_3_and_8_percent():
    diag = CleanerDiagnostics(nitrogen_violations=5)
    badge = compute_scorecard(diag, total_rows=100)[0]
    assert badge.level == "orange"


def test_azoto_red_over_8_percent():
    diag = CleanerDiagnostics(nitrogen_violations=12)
    badge = compute_scorecard(diag, total_rows=100)[0]
    assert badge.level == "red"


# ---------- Peronospora thresholds (absolute count: 0-2 / 3-4 / 5+) ----------


@pytest.mark.parametrize(
    "events, expected_level",
    [
        (0, "green"),
        (2, "green"),
        (3, "orange"),
        (4, "orange"),
        (5, "red"),
        (10, "red"),
    ],
)
def test_peronospora_thresholds(events, expected_level):
    diag = CleanerDiagnostics(peronospora_events=events)
    badge = compute_scorecard(diag, total_rows=100)[1]
    assert badge.level == expected_level


# ---------- Irrigazione thresholds (5 / 15 percent) ----------


@pytest.mark.parametrize(
    "inefficient, expected_level",
    [
        (3, "green"),  # 3%
        (10, "orange"),  # 10%
        (20, "red"),  # 20%
    ],
)
def test_irrigazione_thresholds(inefficient, expected_level):
    diag = CleanerDiagnostics(irrigation_inefficient=inefficient)
    badge = compute_scorecard(diag, total_rows=100)[2]
    assert badge.level == expected_level


# ---------- Suolo thresholds (10 / 30 percent) ----------


@pytest.mark.parametrize(
    "low, expected_level",
    [
        (5, "green"),  # 5%
        (20, "orange"),  # 20%
        (50, "red"),  # 50%
    ],
)
def test_suolo_thresholds(low, expected_level):
    diag = CleanerDiagnostics(soil_organic_low=low)
    badge = compute_scorecard(diag, total_rows=100)[3]
    assert badge.level == expected_level


# ---------- overall_message aggregation ----------


def test_overall_message_all_green():
    diag = CleanerDiagnostics()  # tutti zero => tutti green
    badges = compute_scorecard(diag, total_rows=100)
    assert overall_message(badges) == ("Gestione conforme agli standard di sostenibilità")


def test_overall_message_multiple_red():
    diag = CleanerDiagnostics(
        nitrogen_violations=20,  # red
        peronospora_events=10,  # red
        irrigation_inefficient=3,  # green
        soil_organic_low=5,  # green
    )
    badges = compute_scorecard(diag, total_rows=100)
    assert overall_message(badges) == ("Criticità sistemiche: revisione protocolli necessaria")
