"""Smoke test: tutti i preset territoriali sono validi e caricabili."""

from pathlib import Path

import yaml


def test_all_presets_have_required_fields():
    data = yaml.safe_load(Path("configs/agri_knowledge.yaml").read_text(encoding="utf-8"))
    presets = data["regional_presets"]
    assert len(presets) >= 12
    for key, p in presets.items():
        assert "region" in p, f"{key}: missing region"
        assert "crop" in p, f"{key}: missing crop"
        assert "crop_display" in p, f"{key}: missing crop_display"
        assert "max_yield" in p, f"{key}: missing max_yield"
        assert "ideal_ph" in p, f"{key}: missing ideal_ph"


def test_presets_cover_at_least_ten_regions():
    data = yaml.safe_load(Path("configs/agri_knowledge.yaml").read_text(encoding="utf-8"))
    regions = {p["region"] for p in data["regional_presets"].values()}
    assert len(regions) >= 10, f"Solo {len(regions)} regioni: {regions}"


def test_crops_biological_rules_exist():
    data = yaml.safe_load(Path("configs/agri_knowledge.yaml").read_text(encoding="utf-8"))
    crops = data.get("crops", {})
    for crop_key in ["olive", "durum_wheat", "soft_wheat", "wine_grape_docg",
                     "rice", "apple", "tomato"]:
        assert crop_key in crops, f"Missing biological rules for {crop_key}"
        assert "t_base" in crops[crop_key]
