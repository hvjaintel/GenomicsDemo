"""Tests for the guarantees this demo makes.

These are not decorative. Each one locks down a claim the booth makes to a live
audience, and would catch a change that made the demo dishonest.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import pytest

from app.config import Config, ConfigError, get_config
from app.parsing import LogParser, isa_is_consistent, parse_vcf
from app.runner import DeepVariantRunner, RunResult, RunSpec, speedup
from app.tco import compute as tco_compute


@pytest.fixture(scope="module")
def cfg() -> Config:
    return get_config()


@pytest.fixture
def spec() -> RunSpec:
    return RunSpec(
        sample_id="chr20",
        bam=Path("/data/chr20/sample.bam"),
        reference=Path("/data/reference/ref.fasta"),
        regions="chr20",
        num_shards=192,
        engine_image="google/deepvariant:1.10.0",
    )


# ---------------------------------------------------------------------------
# The central claim: AMX ON and AMX OFF differ ONLY by the ISA ceiling.
# ---------------------------------------------------------------------------


def test_amx_toggle_changes_only_the_isa(cfg, spec, tmp_path):
    runner = DeepVariantRunner(cfg)
    on = runner.build_command(spec, amx_on=True, out_dir=tmp_path)
    off = runner.build_command(spec, amx_on=False, out_dir=tmp_path)

    def strip_volatile(cmd: list[str]) -> list[str]:
        # The container name encodes the AMX state but has no effect on speed.
        out, skip = [], False
        for token in cmd:
            if skip:
                skip = False
                continue
            if token == "--name":
                skip = True
                continue
            if token.startswith(("ONEDNN_MAX_CPU_ISA=", "DNNL_MAX_CPU_ISA=")):
                continue
            out.append(token)
        return out

    assert strip_volatile(on) == strip_volatile(off), (
        "AMX ON and AMX OFF commands differ by more than the ISA ceiling — "
        "any measured speedup would not be attributable to AMX"
    )


def test_amx_toggle_actually_sets_the_expected_isa(cfg, spec, tmp_path):
    runner = DeepVariantRunner(cfg)
    on = runner.build_command(spec, amx_on=True, out_dir=tmp_path)
    off = runner.build_command(spec, amx_on=False, out_dir=tmp_path)

    assert "ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX" in on
    assert "ONEDNN_MAX_CPU_ISA=AVX512_CORE" in off
    assert "ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX" not in off


def test_shard_count_and_inputs_identical_across_legs(cfg, spec, tmp_path):
    runner = DeepVariantRunner(cfg)
    on = " ".join(runner.build_command(spec, True, tmp_path))
    off = " ".join(runner.build_command(spec, False, tmp_path))
    for token in ("--num_shards=192", "--regions=chr20", "--ref=/ref/ref.fasta",
                  "--reads=/input/sample.bam"):
        assert token in on and token in off


def test_config_rejects_a_meaningless_toggle():
    raw = get_config().raw
    broken = {**raw, "amx": {**raw["amx"]}}
    broken["amx"]["disabled"] = {**broken["amx"]["disabled"],
                                 "onednn_max_cpu_isa": "AVX512_CORE_AMX"}
    with pytest.raises(ConfigError, match="dishonest"):
        Config(broken, Path("test"))


# ---------------------------------------------------------------------------
# Speedup is refused whenever the comparison would be invalid.
# ---------------------------------------------------------------------------


def _result(amx_on: bool, seconds: float, fingerprint: str = "same", ok: bool = True) -> RunResult:
    return RunResult(
        run_id="t",
        amx_on=amx_on,
        requested_isa="X",
        started_at=0.0,
        finished_at=seconds,
        exit_code=0 if ok else 1,
        error=None if ok else "boom",
        fingerprint=fingerprint,
    )


def test_speedup_is_computed_when_valid():
    assert speedup(_result(True, 100), _result(False, 250)) == pytest.approx(2.5)


def test_speedup_refused_when_a_leg_failed():
    assert speedup(_result(True, 100), _result(False, 250, ok=False)) is None
    assert speedup(_result(True, 100, ok=False), _result(False, 250)) is None


def test_speedup_refused_when_parameters_differed():
    assert speedup(_result(True, 100, "a"), _result(False, 250, "b")) is None


def test_fingerprint_ignores_amx_but_catches_real_differences(spec):
    other = RunSpec(**{**spec.__dict__, "num_shards": 96})
    assert spec.fingerprint() != other.fingerprint()
    assert spec.fingerprint() == RunSpec(**spec.__dict__).fingerprint()


# ---------------------------------------------------------------------------
# ISA verification — the toggle must be provable, not just asserted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requested,reported,expected",
    [
        ("AVX512_CORE_AMX", "Intel AVX-512 with AMX and bfloat16", True),
        ("AVX512_CORE", "Intel AVX-512", True),
        ("AVX512_CORE_AMX", "Intel AVX-512", False),        # asked for AMX, didn't get it
        ("AVX512_CORE", "Intel AVX-512 with AMX", False),   # AMX leaked into the baseline
        ("AVX512_CORE_AMX", None, None),                    # nothing reported -> unknown
    ],
)
def test_isa_consistency(requested, reported, expected):
    assert isa_is_consistent(requested, reported) is expected


def test_log_parser_extracts_isa_and_stages():
    parser = LogParser()
    t = time.time()
    for i, line in enumerate(
        [
            "onednn_verbose,info,cpu,isa:Intel AVX-512 with float16, AMX and bfloat16",
            "***** Running the command: /opt/deepvariant/bin/make_examples --mode calling *****",
            "Processing... 50%",
            "***** Running the command: /opt/deepvariant/bin/call_variants *****",
            "***** Running the command: /opt/deepvariant/bin/postprocess_variants *****",
        ]
    ):
        parser.feed(line, t + i)

    assert parser.reported_isa is not None and "AMX" in parser.reported_isa
    assert parser.stages["make_examples"].finished
    assert parser.stages["call_variants"].finished
    assert parser.stages["postprocess_variants"].started

    parser.finalize(t + 10)
    assert parser.overall_percent == 100.0


# ---------------------------------------------------------------------------
# Variant counting must reflect the VCF, not a guess.
# ---------------------------------------------------------------------------


VCF = """\
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
chr20\t100\t.\tA\tG\t50\tPASS\t.\tGT\t0/1
chr20\t200\t.\tC\tT\t60\tPASS\t.\tGT\t1/1
chr20\t300\t.\tAT\tA\t40\tPASS\t.\tGT\t0/1
chr20\t400\t.\tG\tGGG\t30\tRefCall\t.\tGT\t0/0
chr20\t500\t.\tT\t.\t0\t.\t.\tGT\t0/0
"""


def test_parse_vcf_plain(tmp_path):
    path = tmp_path / "out.vcf"
    path.write_text(VCF)
    counts = parse_vcf(path)
    assert counts.total == 4        # the ALT="." record is not a variant
    assert counts.snps == 2
    assert counts.indels == 2
    assert counts.passing == 3      # RefCall is not PASS
    assert counts.het == 2
    assert counts.hom_alt == 1
    assert len(counts.examples) == 3


def test_parse_vcf_gzipped(tmp_path):
    path = tmp_path / "out.vcf.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(VCF)
    assert parse_vcf(path).total == 4


def test_parse_vcf_missing_file_is_zero_not_an_error(tmp_path):
    assert parse_vcf(tmp_path / "nope.vcf.gz").total == 0


# ---------------------------------------------------------------------------
# TCO figures must always declare whether they came from a measurement.
# ---------------------------------------------------------------------------


def test_tco_flags_estimated_versus_measured(cfg):
    estimated = tco_compute(cfg, 50)
    assert estimated.runtime_is_measured is False
    assert "estimate" in estimated.runtime_source

    measured = tco_compute(cfg, 50, measured_seconds_per_genome=1800.0)
    assert measured.runtime_is_measured is True
    assert measured.seconds_per_genome == 1800.0
    assert measured.genomes_per_day > estimated.genomes_per_day


def test_tco_scales_with_volume(cfg):
    small = tco_compute(cfg, 10)
    large = tco_compute(cfg, 500)
    assert large.utilisation_pct >= small.utilisation_pct
    assert large.annual_energy_kwh > small.annual_energy_kwh


# ---------------------------------------------------------------------------
# Config integrity.
# ---------------------------------------------------------------------------


def test_every_sample_points_at_a_real_dataset(cfg):
    for sample in cfg.samples:
        assert sample.dataset in cfg.datasets


def test_core_dataset_checksums_are_pinned(cfg):
    for key in ("reference", "smoke_bam", "chr20_bam"):
        dataset = cfg.dataset(key)
        assert dataset.has_recorded_checksum, f"{key} has no pinned sha256"
        assert len(dataset.sha256) == 64


def test_fai_builder_matches_samtools_format(tmp_path):
    """Our dependency-free .fai must agree with samtools' format exactly."""
    from scripts.build_fai import build_fai

    fasta = tmp_path / "t.fasta"
    fasta.write_text(">chr1 desc\nACGTACGTAC\nACGTACGTAC\nACGT\n>chr2\nTTTTT\n")
    fai = build_fai(fasta)
    rows = [line.split("\t") for line in fai.read_text().splitlines()]

    assert rows[0] == ["chr1", "24", "11", "10", "11"]
    assert rows[1] == ["chr2", "5", "44", "5", "6"]


# ---------------------------------------------------------------------------
# Replay traces must be recordings, and must prove it.
# ---------------------------------------------------------------------------


def test_hand_authored_trace_is_rejected():
    """The shape a person would invent, with plausible-looking timings."""
    from app.replay import validate_trace

    fabricated = {
        "version": 1,
        "sample_id": "chr20",
        "legs": {
            "amx_on": {"result": {"wall_clock_s": 110}},
            "amx_off": {"result": {"wall_clock_s": 260}},
        },
    }
    problems = validate_trace(fabricated)
    assert problems
    assert any("recorded_at" in p for p in problems)
    assert any("reported_isa" in p for p in problems)


def test_recorded_trace_round_trips_and_validates(cfg, tmp_path, monkeypatch):
    from app import replay as replay_mod

    monkeypatch.setattr(type(cfg), "traces_dir", property(lambda self: tmp_path))

    legs = {}
    for key, amx_on, seconds in (("amx_on", True, 110.0), ("amx_off", False, 260.0)):
        result = RunResult(
            run_id=key,
            amx_on=amx_on,
            requested_isa="AVX512_CORE_AMX" if amx_on else "AVX512_CORE",
            reported_isa="Intel AVX-512 with AMX" if amx_on else "Intel AVX-512",
            started_at=1_700_000_000.0,
            finished_at=1_700_000_000.0 + seconds,
            exit_code=0,
            fingerprint="same",
        )
        legs[key] = (["line one", "line two"], result)

    path = replay_mod.record_trace(cfg, "t.json", "chr20", legs)
    assert replay_mod.validate_trace(json.loads(path.read_text())) == []
    assert replay_mod.load_trace(path) is not None


def test_load_trace_returns_none_for_untrustworthy_file(tmp_path):
    from app.replay import load_trace

    bad = tmp_path / "bad.json"
    bad.write_text('{"legs": {}}')
    assert load_trace(bad) is None
