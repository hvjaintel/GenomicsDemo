"""Efficiency / TCO maths for the booth panel.

EVERY number produced here is derived from the assumptions in config.yaml, not
measured. The only exception is `seconds_per_genome`, which prefers a real
measured runtime when one exists in this session. Each result carries a flag
saying which it was, and the UI must label it accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass
class EfficiencyResult:
    weekly_samples: int
    seconds_per_genome: float
    runtime_is_measured: bool
    genomes_per_day: float
    days_to_clear_weekly: float
    servers_needed: int
    energy_kwh_per_genome: float
    cost_per_genome: float
    annual_energy_kwh: float
    annual_energy_cost: float
    utilisation_pct: float
    acoustic_dba: float
    currency: str

    @property
    def runtime_source(self) -> str:
        return (
            "measured on this machine in this session"
            if self.runtime_is_measured
            else "configured estimate (no live run yet)"
        )


def compute(
    cfg: Config,
    weekly_samples: int,
    measured_seconds_per_genome: float | None = None,
) -> EfficiencyResult:
    tco = cfg.tco
    a = tco.get("assumptions", {})

    hours_available = float(a.get("hours_per_day_available", 20))
    watts_load = float(a.get("server_power_watts_load", 1000))
    watts_idle = float(a.get("server_power_watts_idle", 400))
    cost_kwh = float(a.get("electricity_cost_per_kwh", 0.16))
    currency = str(a.get("currency", "USD"))

    if measured_seconds_per_genome and measured_seconds_per_genome > 0:
        seconds_per_genome = float(measured_seconds_per_genome)
        measured = True
    else:
        wgs = next((s for s in cfg.samples if s.id == "wgs"), None)
        seconds_per_genome = float(
            (wgs.runtime_estimate_amx_on_s if wgs else None) or 3600
        )
        measured = False

    genomes_per_day = (hours_available * 3600.0) / seconds_per_genome
    weekly_capacity = genomes_per_day * 7.0
    days_to_clear = (
        weekly_samples / genomes_per_day if genomes_per_day > 0 else float("inf")
    )
    servers_needed = max(1, -(-int(weekly_samples) // max(1, int(weekly_capacity))))
    utilisation = (
        min(100.0, 100.0 * weekly_samples / weekly_capacity) if weekly_capacity else 0.0
    )

    hours_per_genome = seconds_per_genome / 3600.0
    energy_kwh_per_genome = watts_load * hours_per_genome / 1000.0
    cost_per_genome = energy_kwh_per_genome * cost_kwh

    # Loaded for as long as the work takes; idle for the rest of the year.
    annual_genomes = weekly_samples * 52.0
    loaded_hours = annual_genomes * hours_per_genome
    idle_hours = max(0.0, 8760.0 - loaded_hours)
    annual_kwh = (loaded_hours * watts_load + idle_hours * watts_idle) / 1000.0

    return EfficiencyResult(
        weekly_samples=int(weekly_samples),
        seconds_per_genome=seconds_per_genome,
        runtime_is_measured=measured,
        genomes_per_day=genomes_per_day,
        days_to_clear_weekly=days_to_clear,
        servers_needed=servers_needed,
        energy_kwh_per_genome=energy_kwh_per_genome,
        cost_per_genome=cost_per_genome,
        annual_energy_kwh=annual_kwh,
        annual_energy_cost=annual_kwh * cost_kwh,
        utilisation_pct=utilisation,
        acoustic_dba=float(cfg.acoustics.get("static_dba", 0)),
        currency=currency,
    )
