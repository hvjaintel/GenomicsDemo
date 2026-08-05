"""Tests for the guarantees this demo makes.

These are not decorative. Each one locks down a claim the booth makes to a live
audience, and would catch a change that made the demo dishonest.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import time
from pathlib import Path

import pytest

from app.config import Config, ConfigError, get_config
from app.parsing import LogParser, isa_is_consistent, parse_vcf
from app.race import build_legs
from app.runner import DeepVariantRunner, RunResult, RunSpec, speedup, time_ratio
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


def test_downloaded_datasets_have_pinned_checksums(cfg):
    """Anything fetched over the network must be pinned to a known sha256."""
    for key in ("reference", "chr20_bam"):
        dataset = cfg.dataset(key)
        assert not dataset.is_derived
        assert dataset.has_recorded_checksum, f"{key} has no pinned sha256"
        assert len(dataset.sha256) == 64


def test_derived_datasets_declare_their_parent_and_recipe(cfg):
    """A derived file has no upstream checksum, so it must be reproducible."""
    derived = [d for d in cfg.datasets.values() if d.is_derived]
    assert derived, "expected at least the smoke BAM to be derived"
    for dataset in derived:
        assert dataset.derived_from in cfg.datasets
        assert dataset.derive_region, f"{dataset.key} has no region to slice"
        parent = cfg.dataset(dataset.derived_from)
        assert parent.has_recorded_checksum, (
            f"{dataset.key} is derived from {parent.key}, which is itself "
            "unverified — the chain of trust has no root"
        )
        assert "derived locally from" in dataset.provenance


def test_smoke_sample_shares_the_reference_build_of_the_booth_sample(cfg):
    """The smoke BAM must match the reference, or every run fails at 0% overlap.

    This regressed once: the DeepVariant quickstart NA12878 BAM is aligned to
    hg19, and against our GRCh38 reference DeepVariant reported "0 bases found
    in common among our input files".
    """
    smoke = cfg.dataset("smoke_bam")
    assert smoke.is_derived, "smoke BAM must be cut from a GRCh38 sample we already trust"
    assert smoke.derived_from == "chr20_bam"


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


# ---------------------------------------------------------------------------
# Container resource settings must be present, and equal across both legs.
# ---------------------------------------------------------------------------


def test_fd_limit_is_raised_for_both_legs(cfg, spec, tmp_path):
    """192 shards exceed Docker's default 1024 soft limit.

    Without this the run dies with "OSError: [Errno 24] Too many open files"
    during make_examples.
    """
    runner = DeepVariantRunner(cfg)
    for amx_on in (True, False):
        cmd = runner.build_command(spec, amx_on, tmp_path)
        assert "--ulimit" in cmd
        limit = cmd[cmd.index("--ulimit") + 1]
        assert limit.startswith("nofile=")
        soft = int(limit.split("=")[1].split(":")[0])
        assert soft > spec.num_shards


def test_container_cli_is_configurable(cfg, spec, tmp_path, monkeypatch):
    runner = DeepVariantRunner(cfg)
    assert runner.container_cli == ["docker"]

    monkeypatch.setitem(cfg.raw["compute"], "container_cli", ["podman"])
    assert runner.container_cli == ["podman"]
    assert runner.build_command(spec, True, tmp_path)[0] == "podman"


def test_fingerprint_digest_is_short_stable_and_amx_independent(spec):
    digest = spec.fingerprint_digest()
    assert len(digest) == 12
    assert digest == RunSpec(**spec.__dict__).fingerprint_digest()
    assert digest != RunSpec(**{**spec.__dict__, "regions": "chr21"}).fingerprint_digest()


# ---------------------------------------------------------------------------
# Core-scaling race — the honest headline. Exactly one thing may differ.
# ---------------------------------------------------------------------------


def test_scaling_legs_differ_only_by_cpuset(cfg, spec, tmp_path):
    """The two scaling legs must be byte-identical bar --cpuset-cpus.

    If anything else drifts, the race stops measuring core scaling and starts
    measuring an accident.
    """
    full, limited = build_legs(cfg, spec, "scaling")
    runner = DeepVariantRunner(cfg)

    full_cmd = runner.build_command(full.spec, full.amx_on, tmp_path, verbose_isa=False)
    lim_cmd = runner.build_command(limited.spec, limited.amx_on, tmp_path, verbose_isa=False)

    assert "--cpuset-cpus" not in full_cmd
    assert "--cpuset-cpus" in lim_cmd
    idx = lim_cmd.index("--cpuset-cpus")
    assert lim_cmd[idx + 1] == cfg.scaling["baseline_cpuset"]

    # Strip the one permitted difference; everything else must match exactly.
    stripped = lim_cmd[:idx] + lim_cmd[idx + 2:]
    assert stripped == full_cmd


def test_scaling_legs_hold_the_isa_constant(cfg, spec):
    """AMX state must NOT vary in a scaling race, or the result is confounded."""
    full, limited = build_legs(cfg, spec, "scaling")
    assert full.amx_on == limited.amx_on


def test_scaling_legs_share_a_fingerprint(cfg, spec):
    """cpuset is excluded from the fingerprint — it is the permitted difference."""
    full, limited = build_legs(cfg, spec, "scaling")
    assert full.spec.fingerprint() == limited.spec.fingerprint()
    assert full.spec.cpuset != limited.spec.cpuset


def test_scaling_legs_get_distinct_output_dirs(cfg, spec):
    """Both legs share an AMX state, so the leg key is what keeps them apart."""
    full, limited = build_legs(cfg, spec, "scaling")
    assert full.key != limited.key


def test_amx_mode_still_varies_amx(cfg, spec):
    on, off = build_legs(cfg, spec, "amx")
    assert on.amx_on and not off.amx_on
    assert on.spec.cpuset == off.spec.cpuset


@pytest.mark.parametrize(
    "cpuset,expected",
    [("0-15", 16), ("0-47", 48), ("0-3,8-11", 8), ("5", 1), (None, None)],
)
def test_core_count_parses_cpusets(spec, cpuset, expected):
    assert dataclasses.replace(spec, cpuset=cpuset).core_count == expected


def test_time_ratio_refuses_when_fingerprints_differ():
    fast = RunResult(run_id="a", amx_on=True, requested_isa="X", fingerprint="one")
    slow = RunResult(run_id="b", amx_on=True, requested_isa="X", fingerprint="two")
    for r in (fast, slow):
        r.started_at, r.exit_code = 0.0, 0
    fast.finished_at, slow.finished_at = 10.0, 20.0
    assert time_ratio(fast, slow) is None


def test_time_ratio_reports_a_valid_comparison():
    fast = RunResult(run_id="a", amx_on=True, requested_isa="X", fingerprint="same")
    slow = RunResult(run_id="b", amx_on=True, requested_isa="X", fingerprint="same")
    for r in (fast, slow):
        r.started_at, r.exit_code = 0.0, 0
    fast.finished_at, slow.finished_at = 100.0, 300.0
    assert time_ratio(fast, slow) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# bf16 injection is off by default: it is slower AND incorrect on stock
# DeepVariant 1.10. See docs/AMX-FINDINGS.md.
# ---------------------------------------------------------------------------


def test_bf16_injection_is_disabled_by_default(cfg):
    assert cfg.amx.get("use_bf16_injection") is False


def test_bf16_injection_is_not_mounted_when_disabled(cfg, spec, tmp_path):
    runner = DeepVariantRunner(cfg)
    cmd = runner.build_command(spec, True, tmp_path, verbose_isa=False)
    assert not any("container_inject" in part for part in cmd)
    assert not any(part.startswith("DV_FORCE_BF16=1") for part in cmd)


def test_findings_doc_referenced_by_config_exists():
    """config.yaml points readers at the evidence; the evidence must be there."""
    doc = Path(__file__).resolve().parent.parent / "docs" / "AMX-FINDINGS.md"
    assert doc.exists(), "docs/AMX-FINDINGS.md is referenced by config.yaml"
    assert "brgconv:avx512_core" in doc.read_text()


def test_chr20_runtimes_are_marked_measured_not_illustrative(cfg):
    """chr20 has been run on this box, so its numbers must not claim to be estimates."""
    chr20 = cfg.sample("chr20")
    assert chr20.illustrative is False
    assert chr20.runtime_fast_s and chr20.runtime_slow_s
    assert chr20.runtime_slow_s > chr20.runtime_fast_s
