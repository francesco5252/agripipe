"""Test del modulo theme (palette + CSS)."""

from agripipe.ui import theme


def test_palette_has_all_required_colors():
    required = {"sage", "forest", "earth", "water", "cream", "card",
                "leaf", "wheat", "pomegranate", "text", "text_muted"}
    assert required.issubset(theme.PALETTE.keys())


def test_palette_values_are_valid_hex():
    import re
    hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
    for name, value in theme.PALETTE.items():
        assert hex_re.match(value), f"{name}={value} is not a valid hex color"


def test_badge_color_map_exists():
    assert theme.BADGE_COLORS["green"] == theme.PALETTE["leaf"]
    assert theme.BADGE_COLORS["yellow"] == theme.PALETTE["wheat"]
    assert theme.BADGE_COLORS["red"] == theme.PALETTE["pomegranate"]


def test_build_stylesheet_contains_all_colors():
    css = theme.build_stylesheet()
    for hex_value in theme.PALETTE.values():
        assert hex_value in css, f"Missing {hex_value} in CSS"
