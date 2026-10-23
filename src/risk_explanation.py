"""Explain proximity with explicit demo rules; no hazard model or network calls."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from src.proximity import classify_distances, validate_thresholds
from src.settings import PROJECT_ROOT

RULES_PATH = PROJECT_ROOT / "config" / "proximity_rules.json"


@dataclass(frozen=True)
class ProximityExplanation:
    classification: str
    label: str
    priority: str
    distance_m: float | None
    red_band_m: float
    yellow_width_m: float
    outer_band_m: float
    criterion: str
    reason: str
    potential_exposure: str
    pending_factors: tuple[str, ...]
    actions: tuple[str, ...]
    provider_id: str
    rule_version: str
    notice: str
    is_demonstration: bool = True

    def as_record(self) -> dict:
        return asdict(self)


class ExplanationProvider(Protocol):
    """A future API adapter can satisfy this contract without changing the map."""

    def explain(
        self, distance_m: float | None, red_band_m: float, outer_band_m: float,
        expected_class: str | None = None,
    ) -> ProximityExplanation: ...

    def descriptor(self) -> dict: ...


class DemoRuleProvider:
    """Versioned local explanation rules; numerical distances remain real inputs."""

    def __init__(self, path: str | Path = RULES_PATH):
        self._ruleset = json.loads(Path(path).read_text(encoding="utf-8"))
        if self._ruleset.get("schema_version") != 1 or self._ruleset.get("mode") != "demo_rules":
            raise ValueError("Se requiere un archivo de reglas de demostración schema_version=1.")
        for key in ("provider_id", "rule_version", "title", "source", "notice", "pending_factors"):
            if not self._ruleset.get(key):
                raise ValueError(f"Falta {key} en las reglas de demostración.")
        for color in ("red", "yellow", "green", "unknown"):
            rule = self._ruleset.get("rules", {}).get(color, {})
            if not all(rule.get(key) for key in ("label", "priority", "reason", "potential_exposure", "actions")):
                raise ValueError(f"Regla incompleta para {color}.")

    def descriptor(self) -> dict:
        return {
            key: self._ruleset[key]
            for key in ("schema_version", "provider_id", "rule_version", "mode", "title", "source", "notice")
        }

    def explain(self, distance_m, red_band_m, outer_band_m, expected_class=None) -> ProximityExplanation:
        red, outer = validate_thresholds(red_band_m, outer_band_m)
        try:
            distance = float(distance_m) if not isinstance(distance_m, bool) else float("nan")
        except (TypeError, ValueError):
            distance = float("nan")
        distance = distance if math.isfinite(distance) and distance >= 0 else None
        color = str(classify_distances([distance], red, outer)[0]) if distance is not None else "unknown"
        if isinstance(expected_class, str) and expected_class in {"red", "yellow", "green"} and expected_class != color:
            raise ValueError("La clasificación y la distancia no coinciden con los límites del semáforo.")
        if color == "red":
            criterion = f"Distancia medida: {distance:.2f} m. Regla activa: distancia ≤ {red:g} m."
        elif color == "yellow":
            criterion = f"Distancia medida: {distance:.2f} m. Regla activa: más de {red:g} m y hasta {outer:g} m."
        elif color == "green":
            criterion = f"Distancia medida: {distance:.2f} m. Regla activa: distancia > {outer:g} m."
        else:
            criterion = "Sin una distancia válida no se activa ningún intervalo de color."
        rule = self._ruleset["rules"][color]
        return ProximityExplanation(
            classification=color, label=rule["label"], priority=rule["priority"], distance_m=distance,
            red_band_m=red, yellow_width_m=outer - red, outer_band_m=outer, criterion=criterion,
            reason=rule["reason"], potential_exposure=rule["potential_exposure"],
            pending_factors=tuple(self._ruleset["pending_factors"]), actions=tuple(rule["actions"]),
            provider_id=self._ruleset["provider_id"], rule_version=self._ruleset["rule_version"],
            notice=self._ruleset["notice"],
        )


def with_explanations(table, provider: ExplanationProvider | None = None):
    """Export the same active rules shown by the map, without mutating cached data."""
    provider = provider or DemoRuleProvider()
    result = table.copy()
    records = [
        provider.explain(row["distance_m"], row["red_band_m"], row["band_m"], row.get("proximity_class"))
        for _, row in result.iterrows()
    ]
    for field, attribute in {
        "demo_priority": "priority", "demo_criterion": "criterion", "demo_reason": "reason",
        "demo_potential_exposure": "potential_exposure", "demo_rule_version": "rule_version",
        "demo_notice": "notice",
    }.items():
        result[field] = [getattr(record, attribute) for record in records]
    return result
