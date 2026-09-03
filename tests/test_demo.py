"""Tests for the guarantees this demo makes.

These are not decorative. Each one locks down a claim the booth makes to a live
audience, and would catch a change that made the demo dishonest.
"""

from __future__ import annotations

import copy
import dataclasses
import gzip
import json
import re
import shutil
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
# The two ISA selections differ ONLY by the requested ceiling.
# ---------------------------------------------------------------------------


def test_isa_selection_changes_only_the_isa_ceiling(cfg, spec, tmp_path):
    runner = DeepVariantRunner(cfg)
    on = runner.build_command(spec, amx_on=True, out_dir=tmp_path)
    off = runner.build_command(spec, amx_on=False, out_dir=tmp_path)

    def strip_volatile(cmd: list[str]) -> list[str]:
        # The container name encodes the leg but has no effect on speed.
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
        "the two commands differ by more than the ISA ceiling — "
        "any measured speedup would not be attributable to one variable"
    )


def test_the_isa_ceiling_is_pinned_to_avx512_on_every_leg(cfg, spec, tmp_path):
    """Both legs must request the same instruction set, or the race is invalid."""
    runner = DeepVariantRunner(cfg)
    on = runner.build_command(spec, amx_on=True, out_dir=tmp_path)
    off = runner.build_command(spec, amx_on=False, out_dir=tmp_path)

    assert "ONEDNN_MAX_CPU_ISA=AVX512_CORE" in on
    assert "ONEDNN_MAX_CPU_ISA=AVX512_CORE" in off
    assert not any("AMX" in part.upper() for part in on)
    assert not any("AMX" in part.upper() for part in off)


def test_shard_count_and_inputs_identical_across_legs(cfg, spec, tmp_path):
    runner = DeepVariantRunner(cfg)
    on = " ".join(runner.build_command(spec, True, tmp_path))
    off = " ".join(runner.build_command(spec, False, tmp_path))
    for token in ("--num_shards=192", "--regions=chr20", "--ref=/ref/ref.fasta",
                  "--reads=/input/sample.bam"):
        assert token in on and token in off


def test_config_rejects_legs_that_differ_in_instruction_set():
    """A core-scaling race that also varied the ISA would misattribute the win."""
    raw = get_config().raw
    broken = {**raw, "amx": {**raw["amx"]}}
    broken["amx"]["disabled"] = {**broken["amx"]["disabled"],
                                 "onednn_max_cpu_isa": "AVX512_CORE_AMX"}
    with pytest.raises(ConfigError, match="different ceilings"):
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


def test_fingerprint_ignores_the_isa_but_catches_real_differences(spec):
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

    # Derive the measured value from the estimate rather than hardcoding one.
    # A literal here silently encodes whatever config.yaml happened to say when
    # the test was written, and starts failing for an unrelated reason the day
    # a real measurement replaces the estimate -- which is exactly what
    # happened when the WGS runtime went from a 3600 s guess to a measured
    # 1547 s. What this test is actually about is the measured/estimated flag
    # and the direction of the arithmetic, not any particular runtime.
    faster = estimated.seconds_per_genome / 2
    measured = tco_compute(cfg, 50, measured_seconds_per_genome=faster)
    assert measured.runtime_is_measured is True
    assert measured.seconds_per_genome == faster
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
            "full": {"result": {"wall_clock_s": 110}},
            "limited": {"result": {"wall_clock_s": 260}},
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


def test_fingerprint_digest_is_short_stable_and_isa_independent(spec):
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

    # BOTH legs are pinned now -- the race is core-to-core, so the fast leg
    # must not be allowed to quietly collect the SMT siblings as well.
    assert "--cpuset-cpus" in full_cmd
    assert "--cpuset-cpus" in lim_cmd
    full_idx = full_cmd.index("--cpuset-cpus")
    lim_idx = lim_cmd.index("--cpuset-cpus")
    assert full_cmd[full_idx + 1] == cfg.scaling["full_cpuset"]
    assert lim_cmd[lim_idx + 1] == cfg.scaling["baseline_cpuset"]

    # Strip the one permitted difference; everything else must match exactly.
    assert (full_cmd[:full_idx] + full_cmd[full_idx + 2:]
            == lim_cmd[:lim_idx] + lim_cmd[lim_idx + 2:])


def test_scaling_legs_hold_the_isa_constant(cfg, spec):
    """The ISA must NOT vary in a scaling race, or the result is confounded."""
    full, limited = build_legs(cfg, spec, "scaling")
    assert full.amx_on == limited.amx_on


def test_scaling_legs_share_a_fingerprint(cfg, spec):
    """cpuset is excluded from the fingerprint — it is the permitted difference."""
    full, limited = build_legs(cfg, spec, "scaling")
    assert full.spec.fingerprint() == limited.spec.fingerprint()
    assert full.spec.cpuset != limited.spec.cpuset


def test_scaling_legs_get_distinct_output_dirs(cfg, spec):
    """Both legs share every setting but the cpuset, so the key keeps them apart."""
    full, limited = build_legs(cfg, spec, "scaling")
    assert full.key != limited.key


def test_scaling_legs_vary_only_the_core_budget(cfg, spec):
    """The core budget must be the single difference, or the number is meaningless."""
    full, limited = build_legs(cfg, spec)
    assert full.spec.cpuset != limited.spec.cpuset
    assert full.amx_on == limited.amx_on
    assert full.spec.num_shards == limited.spec.num_shards
    assert full.spec.engine_image == limited.spec.engine_image


def test_no_other_race_mode_can_be_requested(cfg, spec):
    """An ISA race would attribute an instruction-set difference to the cores."""
    with pytest.raises(SystemExit, match="core budgets only"):
        build_legs(cfg, spec, "amx")


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
# This demo runs on AVX-512, and nothing may quietly reintroduce a second path.
# ---------------------------------------------------------------------------


def test_no_precision_altering_injection_reaches_the_container(cfg, spec, tmp_path):
    """The bf16 rewrite machinery is gone; it must not come back by accident.

    It forced a different numeric path into the model, which is precisely the
    kind of hidden variable that makes a timing comparison meaningless.
    """
    runner = DeepVariantRunner(cfg)
    cmd = runner.build_command(spec, True, tmp_path, verbose_isa=False)
    assert not any("container_inject" in part for part in cmd)
    assert not any("DV_FORCE_BF16" in part for part in cmd)
    assert not any("PYTHONPATH" in part for part in cmd)


def test_the_pinned_instruction_set_is_avx512(cfg):
    """The demo makes an AVX-512 claim, so it must actually request AVX-512."""
    for state in (True, False):
        isa = cfg.isa_for(state)
        assert "AVX512" in isa
        assert "AMX" not in isa.upper()
        assert "AVX-512" in cfg.amx_label(state)


def test_chr20_runtimes_are_marked_measured_not_illustrative(cfg):
    """chr20 has been run on this box, so its numbers must not claim to be estimates."""
    chr20 = cfg.sample("chr20")
    assert chr20.illustrative is False
    assert chr20.runtime_fast_s and chr20.runtime_slow_s
    assert chr20.runtime_slow_s > chr20.runtime_fast_s


# ---------------------------------------------------------------------------
# Unmounted data volume detection
#
# This is a disk-safety guard, not a cosmetic one: when the U.2 volume is not
# mounted, the config falls back to ./data inside the repo, which lives on the
# 100 GB OS disk. Silently downloading the 46 GB WGS BAM there fills the root
# filesystem. These tests pin the distinction between "not mounted" (fixable
# with `mount`, data still on the disk) and "never set up" (needs a download).
# ---------------------------------------------------------------------------


def _config_with_data_root(tmp_path, root):
    cfg = Config.load()
    raw = copy.deepcopy(cfg.raw)
    raw["paths"]["data_root"] = str(root)
    raw["paths"]["data_root_fallback"] = str(tmp_path / "fallback")
    return Config(raw, cfg.source)


def test_empty_mountpoint_is_reported_as_unmounted(tmp_path):
    mountpoint = tmp_path / "nvme"
    mountpoint.mkdir()  # exists, but nothing mounted on it
    cfg = _config_with_data_root(tmp_path, mountpoint / "genomics")
    assert cfg.data_disk_looks_unmounted is True


def test_populated_mountpoint_is_not_reported_as_unmounted(tmp_path):
    mountpoint = tmp_path / "nvme"
    (mountpoint / "genomics").mkdir(parents=True)
    cfg = _config_with_data_root(tmp_path, mountpoint / "genomics")
    assert cfg.data_disk_looks_unmounted is False


def test_absent_mountpoint_is_not_reported_as_unmounted(tmp_path):
    # Nothing at all at the path: this box was never set up, so a download is
    # the correct action and must not be blocked.
    cfg = _config_with_data_root(tmp_path, tmp_path / "nvme" / "genomics")
    assert cfg.data_disk_looks_unmounted is False


def test_mountpoint_holding_other_files_is_not_reported_as_unmounted(tmp_path):
    # Something is mounted or staged there, just not our directory yet.
    mountpoint = tmp_path / "nvme"
    mountpoint.mkdir()
    (mountpoint / "lost+found").mkdir()
    cfg = _config_with_data_root(tmp_path, mountpoint / "genomics")
    assert cfg.data_disk_looks_unmounted is False


def test_preflight_points_at_mount_not_download_when_unmounted(tmp_path):
    """An unmounted volume must never be answered with "re-download".

    Every dataset reports missing in that state. Following a fetch_data.sh
    suggestion would pull tens of GB onto the OS disk while the real files sit
    on the unmounted volume -- the exact failure this guards.
    """
    from app.preflight import _check_datasets, _check_reference

    mountpoint = tmp_path / "nvme"
    mountpoint.mkdir()
    cfg = _config_with_data_root(tmp_path, mountpoint / "genomics")

    advice = [c.remedy for c in _check_datasets(cfg)] + [_check_reference(cfg).remedy]
    assert advice, "expected at least one check to report a problem"
    for remedy in advice:
        assert "mount" in remedy
        assert "fetch_data.sh" not in remedy


def test_preflight_still_suggests_download_when_never_staged(tmp_path):
    # Nothing at the path at all: downloading really is the right answer.
    from app.preflight import _check_reference

    cfg = _config_with_data_root(tmp_path, tmp_path / "nvme" / "genomics")
    assert "fetch_data.sh" in _check_reference(cfg).remedy


# ---------------------------------------------------------------------------
# Booth legibility
#
# The app is read across a trade-show aisle, so contrast is a functional
# requirement rather than a cosmetic one. Gradio defaults the radio option
# label text to *body_text_color (near-white in this theme) but its background
# to *button_secondary_background_fill (a light gradient), which rendered the
# sample selector as white-on-white. These tests pin both halves.
# ---------------------------------------------------------------------------


def _relative_luminance(hex_colour: str) -> float:
    raw = hex_colour.lstrip("#")
    channels = [int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    r, g, b = linear
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(fg: str, bg: str) -> float:
    a, b = _relative_luminance(fg), _relative_luminance(bg)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def test_contrast_helper_matches_known_values():
    # Anchor the helper itself, so a bug in it cannot silently pass the checks
    # below. Black on white is the defined maximum of 21:1.
    assert _contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0, abs=0.01)
    assert _contrast_ratio("#FFFFFF", "#FFFFFF") == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize(
    "fg_token,bg_token",
    [
        ("checkbox_label_text_color", "checkbox_label_background_fill"),
        ("checkbox_label_text_color_selected", "checkbox_label_background_fill_selected"),
        ("button_secondary_text_color", "button_secondary_background_fill"),
    ],
)
def test_selector_labels_meet_wcag_aa(cfg, fg_token, bg_token):
    from app.theme import build_theme

    theme = build_theme(cfg)
    fg = getattr(theme, fg_token)
    bg = getattr(theme, bg_token)
    assert fg.startswith("#") and bg.startswith("#"), (
        f"{fg_token}/{bg_token} must be literal hex, not a Gradio *token -- "
        "an unresolved token is how the white-on-white bug got in"
    )
    ratio = _contrast_ratio(fg, bg)
    assert ratio >= 4.5, f"{fg} on {bg} is only {ratio:.2f}:1, below WCAG AA"


def test_accel_badge_gradient_is_legible_at_both_ends(cfg):
    """The hero badge's dark text must work across the whole gradient.

    A gradient only has to fail at one end to be a problem, and that end is
    easy to miss: the badge previously started at #0068B5, which gives 3.28:1
    against its #04121F text. The right-hand side passed, so the badge looked
    acceptable in a screenshot while its left third washed out on the booth
    display.
    """
    import re

    from app.theme import build_css

    css = build_css(cfg)
    block = re.search(r"\.accel-badge \{(.*?)\n\}", css, re.S)
    assert block, "accel-badge rule not found"

    gradient = re.search(r"background: linear-gradient\([^)]*\)", block.group(1))
    assert gradient, "accel-badge should use a linear-gradient background"
    stops = re.findall(r"#[0-9A-Fa-f]{6}", gradient.group(0))
    resolved = re.findall(r"var\(--([a-z-]+)\)", gradient.group(0))
    for name in resolved:
        found = re.search(rf"--{name}:\s*(#[0-9A-Fa-f]{{6}})", css)
        assert found, f"gradient references --{name} but it is not defined"
        stops.append(found.group(1))

    text = re.search(r"color: (#[0-9A-Fa-f]{6})", block.group(1)).group(1)
    assert len(stops) >= 2, f"expected at least two gradient stops, got {stops}"
    for stop in stops:
        ratio = _contrast_ratio(text, stop)
        assert ratio >= 4.5, f"badge text {text} on gradient stop {stop} is {ratio:.2f}:1"


# ---------------------------------------------------------------------------
# Run console activity helix
# ---------------------------------------------------------------------------


def test_dna_helix_rejects_unknown_state():
    from app.ui.components import dna_helix

    with pytest.raises(ValueError):
        dna_helix("spinning", "caption")


def test_dna_helix_only_animates_in_the_running_state():
    """The point of the helix is that stillness means something.

    If it kept turning after the run finished, or while the container had gone
    quiet, it would be decoration rather than a signal -- and a wedged run would
    look identical to a healthy one on the booth screen.
    """
    from app.theme import build_css
    from app.ui.components import DNA_STATES, dna_helix

    css = build_css(Config.load())
    assert ".dna-panel:not(.running) .dna-node" in css
    assert "animation-play-state: paused" in css

    for state in DNA_STATES:
        html = dna_helix(state, "caption")
        assert f'class="dna-panel {state}"' in html


def test_dna_helix_caption_is_escaped():
    from app.ui.components import dna_helix

    html = dna_helix("failed", "<script>alert(1)</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_running_caption_carries_no_changing_values():
    """A caption that ticks would restart the animation on every render.

    Gradio swaps the element's HTML whenever the value differs, and a fresh node
    starts its CSS animation from the first keyframe. An elapsed-time counter in
    the caption would therefore leave the helix twitching rather than turning,
    which is why elapsed time lives on the header pill instead.
    """
    import re

    from app.main import DNA_RUNNING_CAPTION

    assert not re.search(r"\d", DNA_RUNNING_CAPTION), (
        "the running caption must be constant; put changing values elsewhere"
    )


def test_helix_disclaims_being_a_progress_bar():
    """The demo's whole argument is that shown numbers mean something.

    An animation that looked like progress while being driven by nothing would
    undercut that, so the caption has to say what it is.
    """
    from app.main import DNA_IDLE_CAPTION, DNA_RUNNING_CAPTION

    for caption in (DNA_IDLE_CAPTION, DNA_RUNNING_CAPTION):
        assert "progress" in caption.lower()


def test_runner_emits_heartbeats_while_a_command_is_silent():
    """The liveness claim has to be backed by a real event stream.

    Iterating a pipe blocks until the next line, so a container that stops
    talking also stops the caller's loop -- the UI would keep animating a run
    that had wedged. This drives a process that prints, sleeps, then prints.
    """
    import subprocess

    from app.runner import DeepVariantRunner

    proc = subprocess.Popen(
        ["sh", "-c", "echo first; sleep 3; echo second"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    items = list(DeepVariantRunner._readlines(proc))
    proc.wait()

    lines = [i.strip() for i in items if i is not None]
    beats = [i for i in items if i is None]
    assert lines == ["first", "second"]
    assert beats, "no heartbeat was produced across a 3s silence"


def test_runner_readlines_terminates_when_output_closes():
    import subprocess

    from app.runner import DeepVariantRunner

    proc = subprocess.Popen(
        ["sh", "-c", "echo only"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    items = list(DeepVariantRunner._readlines(proc))
    proc.wait()
    assert [i.strip() for i in items if i is not None] == ["only"]


def test_heartbeat_is_the_event_kind_the_consumers_listen_for():
    """These branches were previously dead.

    They tested for kind == "progress", which the runner never emitted, so
    `python -m app.run --quiet` printed nothing at all for the length of a run.
    """
    from pathlib import Path

    for name in ("run.py", "race.py"):
        src = Path("app", name).read_text()
        assert 'kind == "progress"' not in src, f"{name} still listens for a dead event kind"
        assert 'kind == "heartbeat"' in src


def _helix_states(monkeypatch, beats: int, beat_seconds: float) -> list[str]:
    """Drive run_console over a synthetic event stream and collect helix states."""
    import types

    import app.main as M
    from app.runner import RunEvent, RunResult

    clock = {"t": 1_000.0}
    monkeypatch.setattr(M, "time", types.SimpleNamespace(time=lambda: clock["t"]))

    def events():
        yield RunEvent(kind="log", run_id="x", amx_on=True, line="starting", elapsed_s=0.1)
        for i in range(beats):
            clock["t"] += beat_seconds
            yield RunEvent(kind="heartbeat", run_id="x", amx_on=True, elapsed_s=i)
        yield RunEvent(kind="log", run_id="x", amx_on=True, line="talking again", elapsed_s=90)
        result = RunResult(
            run_id="x", amx_on=True, requested_isa="AVX512_CORE_AMX",
            output_dir="/tmp", fingerprint="f", command="c",
        )
        result.started_at, result.finished_at, result.exit_code = 0.0, 100.0, 0
        yield RunEvent(kind="done", run_id="x", amx_on=True, result=result)

    class FakeRunner:
        def __init__(self, cfg): pass
        def reset_cancel(self): pass
        def build_spec(self, sid): return object()
        def run(self, spec, amx_on=True): return events()

    monkeypatch.setattr(M, "DeepVariantRunner", FakeRunner)
    monkeypatch.setattr(M.replay_mod, "should_replay", lambda cfg, live: False)

    states = []
    for out in M.run_console(Config.load(), "smoke"):
        dna = out[1]
        if isinstance(dna, str) and "dna-panel" in dna:
            states.append(dna.split("dna-panel ")[1].split('"')[0])
    return states


def test_helix_stops_when_output_stops_and_restarts_when_it_returns(monkeypatch):
    """A wedged run must not look like a healthy one on the booth screen."""
    states = _helix_states(monkeypatch, beats=40, beat_seconds=2.0)

    assert states[0] == "running"
    assert "quiet" in states, "the helix never stopped despite a long silence"
    assert states.index("quiet") > 0
    # Output resumes, so it must start turning again before the run ends.
    assert states[-2] == "running"
    assert states[-1] == "done"


def test_helix_keeps_turning_through_ordinary_gaps(monkeypatch):
    """DeepVariant pauses routinely; the indicator must not cry wolf."""
    states = _helix_states(monkeypatch, beats=10, beat_seconds=1.0)

    assert "quiet" not in states, f"flagged a {10 * 1.0:.0f}s gap as a stall: {states}"
    assert states[-1] == "done"


def test_helix_is_not_re_sent_while_nothing_about_it_changes(monkeypatch):
    """Re-sending identical markup would restart the CSS animation."""
    states = _helix_states(monkeypatch, beats=10, beat_seconds=1.0)

    # running -> done, and nothing in between: ten beats produced no updates.
    assert states == ["running", "done"], states


def test_helix_respects_reduced_motion():
    """Looping motion on a large screen is a genuine accessibility problem.

    The helix still renders -- and its state colours still carry the meaning --
    it just holds still.
    """
    from app.theme import build_css

    css = build_css(Config.load())
    assert "@media (prefers-reduced-motion: reduce)" in css
    block = css.split("@media (prefers-reduced-motion: reduce)")[1]
    assert "animation: none !important" in block
    # Held still at the widest point rather than collapsed to a line.
    assert ".dna-node.a { top: 4px; }" in block.replace("  ", " ")


def _masthead_html(app) -> str:
    """Locate the masthead by content; block ordering is not a contract."""
    for block in app.blocks.values():
        value = str(getattr(block, "value", "") or "")
        if "booth-masthead" in value:
            return value
    raise AssertionError("masthead block not found")


def test_masthead_omits_the_partner_mark_when_it_is_blank(cfg):
    """A blank strapline must leave no trace, not an empty element.

    The masthead is a flex row, so an empty child still participates in its
    spacing, and the hardware note used to end "...fp32) — ." with the mark
    removed. Both read as a rendering bug from three metres away.
    """
    import copy

    from app.main import build_app, render_home

    raw = copy.deepcopy(cfg.raw)
    raw["demo"]["partner_mark"] = ""
    blank = Config(raw, cfg.source)

    masthead = _masthead_html(build_app(blank))
    assert 'class="partner-mark"' not in masthead

    home = render_home(blank)
    assert "—  ." not in home and "— ." not in home
    assert "rather than assuming it." in home


def test_partner_mark_still_renders_when_one_is_configured(cfg):
    """Removing the default must not delete the feature."""
    import copy

    from app.main import render_home

    raw = copy.deepcopy(cfg.raw)
    raw["demo"]["partner_mark"] = "Example strapline"
    marked = Config(raw, cfg.source)

    home = render_home(marked)
    assert "— Example strapline." in home


# ---------------------------------------------------------------------------
# Container scratch space
# ---------------------------------------------------------------------------


def test_container_tmp_is_mounted_off_the_os_disk(cfg):
    """DeepVariant's scratch must not land on the container's writable layer.

    run_deepvariant fans out one make_examples process per shard, and each is a
    self-extracting Bazel binary that unpacks its runfiles into $TMPDIR. At 192
    shards that measured 6.5 GB on the *smoke* sample -- it scales with shard
    count, not genome size. Unmounted, that goes to /var/lib/docker on the OS
    disk and the run dies with GNU parallel's "Cannot append to buffer file in
    /tmp. Is the disk full?".
    """
    from app.runner import CONTAINER_TMP_DIR, DeepVariantRunner

    runner = DeepVariantRunner(cfg)
    spec = runner.build_spec("smoke")
    out_dir = cfg.runs_dir / "unit-test-out"
    cmd = runner.build_command(spec, amx_on=True, out_dir=out_dir)

    scratch = DeepVariantRunner.scratch_dir(out_dir)
    assert f"{scratch}:{CONTAINER_TMP_DIR}" in cmd

    # The scratch must sit on the data volume, not inside the repo or on /.
    assert str(scratch).startswith(str(cfg.data_root))
    # A sibling, not a child: nesting it inside the /output bind mount would
    # also expose it in the directory users are invited to browse.
    assert scratch.parent == out_dir.parent
    assert out_dir not in scratch.parents


def test_everything_pointed_at_tmp_uses_the_mounted_path(cfg):
    """HOME and MPLCONFIGDIR also write to /tmp; they must share the mount."""
    from app.runner import CONTAINER_TMP_DIR, DeepVariantRunner

    runner = DeepVariantRunner(cfg)
    env = runner._env_for(amx_on=True, verbose_isa=False)

    assert env["TMPDIR"] == CONTAINER_TMP_DIR
    assert env["HOME"] == CONTAINER_TMP_DIR
    assert env["MPLCONFIGDIR"].startswith(CONTAINER_TMP_DIR + "/")


def test_scratch_mount_is_identical_for_both_race_legs(cfg):
    """A race is only valid if the legs differ by the one thing under test."""
    from app.runner import DeepVariantRunner

    runner = DeepVariantRunner(cfg)
    spec = runner.build_spec("smoke")
    out_dir = cfg.runs_dir / "unit-test-out"

    on = runner.build_command(spec, amx_on=True, out_dir=out_dir)
    off = runner.build_command(spec, amx_on=False, out_dir=out_dir)

    def mounts(cmd):
        return [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]

    assert mounts(on) == mounts(off)


def test_preflight_reports_the_os_disk_separately(cfg):
    """The data volume being healthy says nothing about the disk Docker uses."""
    from app.preflight import run_preflight

    names = [c.name for c in run_preflight(cfg).checks]
    assert "OS disk" in names
    assert "Data storage" in names


def test_os_disk_check_fails_when_nearly_full(cfg, monkeypatch):
    import shutil as _shutil

    import app.preflight as P

    Usage = type("Usage", (), {})

    def fake_usage(path):
        u = Usage()
        u.total, u.free = 100 * 1024**3, 4 * 1024**3
        u.used = u.total - u.free
        return u

    monkeypatch.setattr(P.shutil, "disk_usage", fake_usage)
    check = P._check_os_disk(cfg)
    assert check.status is P.Status.FAIL
    assert check.blocking
    assert "docker image prune" in check.remedy
    assert _shutil is not None


@pytest.mark.parametrize(
    "mapper_name,expected",
    [
        # This machine: group "ubuntu-vg-1", LV "ubuntu-lv". Splitting on "-"
        # before unescaping would yield the group "ubuntu" and a wrong device.
        ("ubuntu--vg--1-ubuntu--lv", ("ubuntu-vg-1", "ubuntu-lv")),
        # The idle install on the other NVMe -- one character apart from above.
        ("ubuntu--vg-ubuntu--lv", ("ubuntu-vg", "ubuntu-lv")),
        ("vg0-root", ("vg0", "root")),
    ],
)
def test_lv_hint_unescapes_device_mapper_names(monkeypatch, mapper_name, expected):
    """Pointing booth staff at the wrong volume group is worse than no hint."""
    import subprocess as _sp

    import app.preflight as P

    def fake_run(*args, **kwargs):
        return _sp.CompletedProcess(args, 0, stdout=f"/dev/mapper/{mapper_name}\n", stderr="")

    monkeypatch.setattr(P.subprocess, "run", fake_run)
    hint = P._lv_extend_hint()
    vg, lv = expected
    assert f"/dev/{vg}/{lv}" in hint
    assert f"sudo vgs {vg}" in hint


def test_lv_hint_is_silent_when_root_is_not_lvm(monkeypatch):
    """No LVM, no advice -- rather than advice that cannot work."""
    import subprocess as _sp

    import app.preflight as P

    monkeypatch.setattr(
        P.subprocess, "run",
        lambda *a, **k: _sp.CompletedProcess(a, 0, stdout="/dev/nvme0n1p2\n", stderr=""),
    )
    assert P._lv_extend_hint() == ""


def test_active_stage_picks_the_running_one():
    from app.run import _active_stage

    stages = {
        "make_examples": {"label": "Make examples", "started": True, "finished": True},
        "call_variants": {"label": "Call variants", "started": True, "finished": False},
        "postprocess_variants": {"label": "Postprocess", "started": False, "finished": False},
    }
    assert _active_stage(stages) == "Call variants"


def test_active_stage_is_none_before_any_stage_is_known():
    """Better to say nothing than to name a stage that has not started."""
    from app.run import _active_stage

    stages = {
        name: {"label": name, "started": False, "finished": False}
        for name in ("make_examples", "call_variants", "postprocess_variants")
    }
    assert _active_stage(stages) is None


def test_active_stage_ignores_payload_key_order():
    """event.stages is a plain dict; correctness must not rest on its ordering."""
    from app.run import _active_stage

    reversed_payload = {
        "postprocess_variants": {"label": "Postprocess", "started": False, "finished": False},
        "call_variants": {"label": "Call variants", "started": True, "finished": False},
        "make_examples": {"label": "Make examples", "started": True, "finished": True},
    }
    assert _active_stage(reversed_payload) == "Call variants"


def test_make_examples_reports_liveness_without_a_percentage():
    """The bug this guards: a WGS run sat at "0.0%" for ~14 minutes.

    make_examples never prints a percentage, so the CLI's only progress signal
    stayed at zero for the longest stage of the run -- indistinguishable from a
    hung job on a booth terminal.
    """
    from app.parsing import LogParser

    parser = LogParser()
    # The real banner run_deepvariant prints when it launches the stage.
    parser.feed("***** Running the command:*****", 0.0)
    parser.feed(
        'time seq 0 191 | parallel -q --halt 2 --line-buffer '
        '/opt/deepvariant/bin/make_examples --mode calling --task {}',
        0.1,
    )
    # ...and a real progress line from a live whole-genome run. Note it carries
    # no percentage, only a per-shard candidate count.
    parser.feed(
        "I0826 16:50:06.572551 1 make_examples_core.py:384] "
        "Task 82/192: 16002 candidates (4239 examples) [37.24s elapsed]",
        1.0,
    )
    assert parser.overall_percent == 0.0, "precondition: no percentage is available"

    from app.run import _active_stage
    from app.runner import DeepVariantRunner

    stages = DeepVariantRunner._stage_payload(parser)
    assert _active_stage(stages) is not None, (
        "with no percentage to show, the running stage is the only honest signal"
    )


def _wgs_sample(cfg):
    return next(s for s in cfg.samples if s.id == "wgs")


def test_expected_runtime_declines_to_guess_for_an_unpinned_run():
    """Once the race became core-to-core, runtime_fast_s stopped meaning "all CPUs".

    It is the 96-physical-core figure. An unpinned run also gets all 96 SMT
    siblings, so it is a third configuration with nothing on file. Quoting the
    96-core number for it would be exactly the "figure recorded under different
    conditions" the estimator exists to avoid.
    """
    from app.config import Config
    from app.run import _expected_runtime

    cfg = Config.load()
    est, qualifier = _expected_runtime(_wgs_sample(cfg), cfg, None)
    assert est is None
    assert qualifier == ""


def test_expected_runtime_does_not_quote_all_core_time_for_a_pinned_run():
    """The bug: launching the 16-core baseline announced '~25m 47s'.

    That is the all-core measurement, for a run that takes hours. An operator
    reading it would conclude the job had hung long before it was halfway.
    """
    from app.config import Config
    from app.run import _expected_runtime

    cfg = Config.load()
    sample = _wgs_sample(cfg)
    est, _ = _expected_runtime(sample, cfg, "0-15")
    assert est != sample.runtime_fast_s
    assert est == sample.runtime_slow_s


def test_expected_runtime_is_unknown_for_an_unrecorded_cpuset():
    """No figure on file beats a figure recorded under other conditions."""
    from app.config import Config
    from app.run import _expected_runtime

    cfg = Config.load()
    est, _ = _expected_runtime(_wgs_sample(cfg), cfg, "0-31")
    assert est is None


def test_baseline_cpuset_is_readable_from_config():
    """_expected_runtime silently returns None if this key ever moves."""
    from app.config import Config

    cfg = Config.load()
    baseline = getattr(getattr(cfg, "scaling", None), "baseline_cpuset", None)
    if baseline is None:
        baseline = (cfg.raw.get("scaling") or {}).get("baseline_cpuset")
    assert baseline, "scaling.baseline_cpuset must resolve, or pinned runs lose their estimate"


def test_wgs_runtimes_are_measured_not_illustrative():
    """Both WGS legs have now been run end to end on this box.

    Guards the pair: flipping illustrative to false while leaving a placeholder
    runtime in place would present an estimate as a measurement, which is the
    one thing this demo must not do.
    """
    from app.config import Config

    sample = next(s for s in Config.load().samples if s.id == "wgs")
    assert sample.illustrative is False
    assert sample.runtime_fast_s == 1979, "96-physical-core run (cpuset 0-95)"
    assert sample.runtime_slow_s == 7491, "16-core run, measured 2026-08-26"


def test_wgs_dataset_panel_claims_measurement_not_estimation():
    from app.config import Config
    import app.main as main

    cfg = Config.load()
    html = main.render_dataset(cfg, "wgs")
    assert "measured on this box" in html
    assert "estimate — for pacing only" not in html
    assert "Illustrative" not in html


def test_smoke_and_tco_remain_marked_illustrative():
    """Those figures really are assumptions; the label must survive this change."""
    from app.config import Config

    cfg = Config.load()
    smoke = next(s for s in cfg.samples if s.id == "smoke")
    assert smoke.illustrative is True
    assert cfg.tco.get("illustrative") is True


# --------------------------------------------------------------------------
# Offline capability. The booth machine may sit on a hostile network or none.
# --------------------------------------------------------------------------

def test_bundled_font_files_are_present_and_valid_woff2():
    """A truncated or missing font silently degrades the booth screen."""
    from app.theme import BUNDLED_FONTS, FONT_DIR

    for _family, filename, _weights in BUNDLED_FONTS:
        path = FONT_DIR / filename
        assert path.exists(), f"{filename} is missing from {FONT_DIR}"
        blob = path.read_bytes()
        assert blob[:4] == b"wOF2", f"{filename} is not a woff2 file"
        assert len(blob) > 10_000, f"{filename} looks truncated ({len(blob)} bytes)"


def test_font_licences_ship_alongside_the_binaries():
    """Both faces are OFL; redistribution requires the licence to travel with them."""
    from app.theme import FONT_DIR

    for name in ("Inter-OFL.txt", "JetBrainsMono-OFL.txt"):
        text = (FONT_DIR / name).read_text(encoding="utf-8", errors="replace")
        assert "SIL OPEN FONT LICENSE" in text.upper()


def test_font_faces_are_inlined_not_fetched():
    from app.theme import font_face_css

    css = font_face_css()
    assert css.count("@font-face") == 2
    assert "data:font/woff2;base64," in css
    assert "http://" not in css and "https://" not in css


def test_stylesheet_contains_no_outbound_urls():
    """The regression guard: any http(s) URL in our CSS is a network dependency.

    The demo previously pulled Inter and JetBrains Mono from fonts.googleapis.com
    via a render-blocking <link>. With no network that fails fast, but on venue
    wi-fi that accepts the connection and then stalls it can hold the booth
    screen blank for seconds.
    """
    from app.config import Config
    from app.theme import build_css

    css = build_css(Config.load())
    assert "http://" not in css
    assert "https://" not in css
    assert "googleapis" not in css and "gstatic" not in css


def test_theme_requests_no_cdn_stylesheets():
    """gr.themes.GoogleFont is what injects the render-blocking <link>.

    Asserting on theme.font is useless -- Gradio flattens it to the CSS string
    "\'Inter\', \'system-ui\', sans-serif" and the font objects are gone. The
    surviving evidence is Font.stylesheet(), which returns a fonts.googleapis.com
    URL for GoogleFont and None for a plain Font, and the _stylesheets list those
    URLs are collected into. An earlier version of this test checked isinstance
    against theme.font and passed happily with GoogleFont restored.
    """
    from app.config import Config
    from app.theme import build_theme

    theme = build_theme(Config.load())

    assert list(getattr(theme, "_stylesheets", [])) == [], (
        "the theme pulls a stylesheet from a CDN"
    )
    for attr in ("_font", "_font_mono"):
        for entry in getattr(theme, attr, ()) or ():
            sheet = entry.stylesheet() if hasattr(entry, "stylesheet") else None
            assert sheet is None, f"{attr} entry {entry!r} fetches {sheet}"


def test_font_face_css_survives_a_missing_font_file(monkeypatch, tmp_path):
    """A booth screen in the wrong typeface still runs; one that won't start doesn't."""
    import app.theme as T

    T.font_face_css.cache_clear()
    monkeypatch.setattr(T, "FONT_DIR", tmp_path)
    try:
        assert T.font_face_css() == ""
    finally:
        T.font_face_css.cache_clear()


def test_served_page_has_no_render_blocking_external_resources():
    """End-to-end guard against the served HTML, not just our own CSS.

    Skipped when the demo is not running. Gradio's own template emits two
    preconnect hints and one `async` script from cdnjs; neither blocks parsing
    or rendering, so those are tolerated. What must never come back is a
    synchronous stylesheet or script from a third-party host, because that is
    what can hold the booth screen blank on a stalled venue network.
    """
    import re
    import urllib.error
    import urllib.request

    try:
        html = urllib.request.urlopen("http://127.0.0.1:7860/", timeout=5).read().decode(
            "utf-8", errors="replace"
        )
    except (urllib.error.URLError, OSError):
        pytest.skip("demo not running on :7860")

    assert "css2?family=" not in html, "a Google Fonts stylesheet link is back"

    for tag in re.findall(r"<link[^>]*>", html):
        if "stylesheet" in tag and re.search(r'href="https?://', tag):
            raise AssertionError(f"external stylesheet: {' '.join(tag.split())}")

    for tag in re.findall(r"<script[^>]*>", html):
        if re.search(r'src="https?://', tag) and not re.search(r"\basync\b|\bdefer\b", tag):
            raise AssertionError(f"blocking external script: {' '.join(tag.split())}")


# ---------------------------------------------------------------------------
# The product is AVX-512 only. Nothing user-visible may say otherwise.
# ---------------------------------------------------------------------------


def test_no_amx_text_reaches_the_screen(cfg):
    """A stray "AMX" badge would be a claim this build cannot substantiate.

    The internal parameter names still say `amx_on` (they select the ISA
    ceiling), so this checks rendered output rather than source text.
    """
    import app.main as M

    rendered = "\n".join([
        M.render_home(cfg),
        M.render_dataset(cfg, "chr20"),
        M.render_efficiency(cfg, 50),
        M._isa_html(cfg, cfg.isa_for(True), "Intel AVX-512", True,
                    avx512_primitives=190, compute_primitives=190),
        M._isa_html(cfg, cfg.isa_for(True), None, None),
    ])
    assert "AMX" not in rendered.upper()


def test_the_config_subtitle_names_avx512_and_not_amx(cfg):
    subtitle = str(cfg.raw["demo"]["subtitle"])
    assert "AVX-512" in subtitle
    assert "AMX" not in subtitle.upper()


def test_isa_consistency_accepts_the_real_hyphenated_avx512_banner():
    """oneDNN prints "Intel AVX-512", hyphenated; a naive substring check misses it.

    This nearly shipped as a false mismatch, so it stays pinned by a test.
    """
    from app.parsing import isa_is_consistent

    assert isa_is_consistent(
        "AVX512_CORE",
        "Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ extensions",
    ) is True
    assert isa_is_consistent("AVX512_CORE", "Intel AVX2") is False


# ---------------------------------------------------------------------------
# The race is CORE to CORE. No leg may be handed an SMT sibling, and no leg may
# be left unpinned -- either would let hyperthreading masquerade as core scaling.
# ---------------------------------------------------------------------------


def _expand_cpuset(cpuset: str) -> list[int]:
    out: list[int] = []
    for part in cpuset.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _cpu_to_core() -> dict[int, int]:
    """Map logical CPU -> physical core id, from lscpu."""
    import subprocess

    proc = subprocess.run(
        ["lscpu", "-p=CPU,CORE"], capture_output=True, text=True, check=True
    )
    mapping = {}
    for line in proc.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cpu, core = line.split(",")[:2]
        mapping[int(cpu)] = int(core)
    return mapping


def test_both_race_legs_are_pinned(cfg, spec):
    """An unpinned leg would silently collect every SMT sibling on the box."""
    full, limited = build_legs(cfg, spec)
    assert full.spec.cpuset, "the fast leg must be pinned, not left to take the whole machine"
    assert limited.spec.cpuset


def test_race_refuses_to_run_with_an_unpinned_fast_leg(cfg, spec):
    raw = copy.deepcopy(cfg.raw)
    raw["scaling"].pop("full_cpuset", None)
    crippled = Config(raw, Path("test"))
    with pytest.raises(SystemExit, match="unpinned leg"):
        build_legs(crippled, spec)


@pytest.mark.skipif(shutil.which("lscpu") is None, reason="needs lscpu")
def test_neither_leg_is_given_an_smt_sibling(cfg, spec):
    """Every CPU in a leg must be a distinct physical core.

    This is what makes the race core-to-core. If a cpuset contained both
    threads of one core, the extra "core" would be hyperthreading and the
    headline would credit cores for a win they did not produce.
    """
    cpu_to_core = _cpu_to_core()
    full, limited = build_legs(cfg, spec)

    for leg in (full, limited):
        cpus = _expand_cpuset(leg.spec.cpuset)
        cores = [cpu_to_core[c] for c in cpus]
        assert len(set(cores)) == len(cores), (
            f"leg {leg.key!r} (cpuset {leg.spec.cpuset}) contains two threads of "
            f"the same physical core — that is SMT, not another core"
        )
        assert len(cores) == leg.spec.core_count


@pytest.mark.skipif(shutil.which("lscpu") is None, reason="needs lscpu")
def test_the_fast_leg_uses_every_physical_core(cfg):
    """96 cores is claimed on screen, so all 96 must actually be in the cpuset."""
    cpu_to_core = _cpu_to_core()
    total_cores = len(set(cpu_to_core.values()))
    full_cpus = _expand_cpuset(cfg.scaling["full_cpuset"])
    assert len(full_cpus) == total_cores


def test_the_scaling_labels_do_not_promise_threads(cfg):
    """The legs are core budgets; the labels must not say "threads"."""
    for key in ("full_label", "baseline_label"):
        assert "thread" not in str(cfg.scaling[key]).lower()


def test_expected_runtime_is_matched_to_the_leg_it_was_measured_on(cfg):
    """Recorded runtimes are per-cpuset; quoting one for another leg is a lie."""
    from app.run import _expected_runtime

    sample = cfg.sample("chr20")
    fast, _ = _expected_runtime(sample, cfg, cfg.scaling["full_cpuset"])
    slow, _ = _expected_runtime(sample, cfg, cfg.scaling["baseline_cpuset"])
    assert fast == sample.runtime_fast_s
    assert slow == sample.runtime_slow_s
    assert slow > fast

    # An unpinned run gets every SMT sibling as well -- that is neither leg, so
    # nothing on file applies and the honest answer is "no estimate".
    assert _expected_runtime(sample, cfg, None)[0] is None
    assert _expected_runtime(sample, cfg, "0-7")[0] is None


# ---------------------------------------------------------------------------
# Offline operation, and running someone else's data.
# ---------------------------------------------------------------------------


def test_the_pipeline_container_gets_no_network_by_default(cfg, spec, tmp_path):
    """Verified: a full run completes under --network none.

    This is enforced rather than assumed because the demo is used on closed
    show networks, and because a visitor's genomic data must not be able to
    leave the box.
    """
    runner = DeepVariantRunner(cfg)
    cmd = runner.build_command(spec, True, tmp_path, verbose_isa=False)
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"


def test_network_isolation_is_part_of_the_fingerprint(spec):
    """A networked leg and an isolated leg are not the same experiment."""
    networked = dataclasses.replace(spec, network=None)
    assert networked.fingerprint() != spec.fingerprint()
    assert speedup(
        _result(True, 100, networked.fingerprint()),
        _result(False, 250, spec.fingerprint()),
    ) is None


def test_a_dataset_may_be_local_only_with_no_url(cfg):
    """Someone's own BAM has nothing to download -- neither has its index.

    The sidecar parser used to require `url`, so a local-only index raised
    KeyError and surfaced as "unknown dataset", which pointed at the wrong
    problem entirely.
    """
    raw = copy.deepcopy(cfg.raw)
    raw["datasets"]["byo"] = {
        "name": "Someone's own BAM",
        "local": "byo/their.bam",
        "sidecars": [{"local": "byo/their.bam.bai"}],
    }
    ds = Config(raw, Path("test")).dataset("byo")
    assert ds.url == ""
    assert ds.sidecars[0].url == ""
    assert ds.sidecars[0].local == "byo/their.bam.bai"


def test_a_missing_dataset_and_a_malformed_one_report_differently(cfg):
    raw = copy.deepcopy(cfg.raw)
    raw["datasets"]["broken"] = {"name": "no local key"}
    config = Config(raw, Path("test"))

    with pytest.raises(ConfigError, match="unknown dataset"):
        config.dataset("not_defined_at_all")
    with pytest.raises(ConfigError, match="missing required field"):
        config.dataset("broken")


# ---------------------------------------------------------------------------
# Presenter guide
#
# This is the one document that gets read aloud to a customer, by someone who
# cannot spot a stale figure because they don't know the workload. A wrong
# number here is quoted with confidence to a prospect, which is the most
# expensive place in the repo for drift to hide.
# ---------------------------------------------------------------------------

PRESENTER_GUIDE = Path(__file__).resolve().parents[1] / "docs" / "PRESENTER-GUIDE.md"


def test_presenter_guide_exists():
    assert PRESENTER_GUIDE.is_file(), "docs/PRESENTER-GUIDE.md is referenced by README"


def test_presenter_guide_quotes_the_configured_cpusets(cfg):
    """The guide answers "is hyperthreading included?" with specific cpusets.

    If the race is ever re-pinned, that answer becomes false while still
    sounding authoritative.
    """
    text = PRESENTER_GUIDE.read_text()
    scaling = cfg.raw["scaling"]
    assert scaling["baseline_label"] in text
    assert scaling["full_label"] in text


def test_presenter_guide_headline_numbers_match_the_readme():
    """Both docs quote the same measured results, so they must not diverge."""
    guide = PRESENTER_GUIDE.read_text()
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    for claim in ["2.66", "339.4", "127.4", "210,390", "7,709,239", "3.79", "25m 47s"]:
        assert claim in guide, f"presenter guide lost {claim}"
        assert claim in readme, f"README lost {claim}"


def test_presenter_guide_marks_the_efficiency_panel_illustrative(cfg):
    """The TCO panel is modelled, not measured.

    The guide instructs the presenter to say so out loud; that instruction has
    to survive, because it is the only safeguard once the words leave the room.
    """
    assert cfg.raw["tco"]["illustrative"] is True
    assert "illustrative" in PRESENTER_GUIDE.read_text().lower()


def test_presenter_guide_does_not_claim_amx_is_used():
    """AMX is the question a well-informed visitor asks.

    The measured answer is that AMX does no work on this fp32 model. The guide
    must keep saying so rather than drifting toward the more flattering claim.
    """
    guide = PRESENTER_GUIDE.read_text()
    assert "no fp32 path through AMX" in guide
    assert "0.98" in guide


def test_docs_only_cite_tests_that_exist():
    """The docs name specific tests as evidence for their claims.

    Citing a test is a strong claim -- "don't take my word for it, run this".
    A renamed or deleted test turns that into a dead reference that reads as
    authoritative, and the first person to notice is someone trying to verify
    a number. Caught this the hard way: the replication section originally
    cited `test_scaling_cpusets_contain_no_smt_siblings`, which never existed.
    """
    root = Path(__file__).resolve().parents[1]
    source = Path(__file__).read_text()
    defined = set(re.findall(r"^def (test_\w+)", source, re.MULTILINE))

    for doc in (root / "README.md", root / "docs" / "PRESENTER-GUIDE.md"):
        cited = set(re.findall(r"\b(test_\w+)", doc.read_text()))
        missing = cited - defined
        assert not missing, f"{doc.name} cites tests that do not exist: {sorted(missing)}"
