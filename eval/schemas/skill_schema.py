"""Schema for skill configuration (parsed from skill.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DimensionConfig:
    """Configuration for a single evaluation dimension."""
    id: str
    key: str
    label: str
    label_en: str
    definition: str
    reasoning_fields: str
    veto: bool = False
    veto_condition: str = ""
    d8_note_condition: str = ""


@dataclass
class VetoRule:
    """A veto rule configuration."""
    rule: str
    description: str
    condition: str
    forced_preference: str = ""


@dataclass
class SkillConfig:
    """Configuration parsed from a skill's skill.yaml file."""

    name: str
    description: str
    type: str
    dimensions: list[DimensionConfig] = field(default_factory=list)
    score_range: dict = field(default_factory=dict)
    veto_rules: list[VetoRule] = field(default_factory=list)
    priority_chain: list[str] = field(default_factory=list)

    def get_dimension_keys(self) -> list[str]:
        return [d.key for d in self.dimensions]

    def get_veto_dimensions(self) -> list[DimensionConfig]:
        return [d for d in self.dimensions if d.veto]
