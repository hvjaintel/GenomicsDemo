"""Parsing of real DeepVariant output — logs, ISA reports, and VCFs.

Two jobs:
  1. Turn the container's log stream into per-stage progress for the run console.
  2. Read the produced VCF so the results panel shows real variant counts,
     never a placeholder.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path

# The three DeepVariant stages, in execution order.
STAGES = ("make_examples", "call_variants", "postprocess_variants")

STAGE_LABELS = {
    "make_examples": "Make examples",
    "call_variants": "Call variants",
    "postprocess_variants": "Postprocess variants",
}

# run_deepvariant announces each stage with a banner line like:
#   ***** Running the command:*****
#   ... /opt/deepvariant/bin/make_examples --mode calling ...
_STAGE_PATTERNS = {
    "make_examples": re.compile(r"\bmake_examples\b"),
    "call_variants": re.compile(r"\bcall_variants\b"),
    "postprocess_variants": re.compile(r"\bpostprocess_variants\b"),
}

# oneDNN verbose header, e.g.
#   onednn_verbose,info,cpu,isa:Intel AVX-512 with float16, Intel DL Boost and ...
_ISA_PATTERNS = (
    re.compile(r"onednn_verbose,info,cpu,isa:\s*(?P<isa>.+)", re.IGNORECASE),
    re.compile(r"mkldnn_verbose,info,cpu,isa:\s*(?P<isa>.+)", re.IGNORECASE),
)

# Percentage progress emitted by make_examples / call_variants sharding.
_PERCENT = re.compile(r"(\d{1,3})%")

# A single oneDNN primitive execution, e.g.
#   onednn_verbose,primitive,exec,cpu,convolution,brgconv:avx512_core,...
# Field 5 is the primitive kind, field 6 the implementation that was chosen.
# The implementation name is the ONLY place oneDNN tells you whether the AMX
# tiles were really used; the ISA banner merely reports what was permitted.
_PRIMITIVE_EXEC = re.compile(
    r"onednn_verbose,(?:primitive,)?exec,\w+,(?P<kind>[\w_]+),(?P<impl>[^,]*),",
    re.IGNORECASE,
)

# Primitive kinds that carry the real neural-network arithmetic. Reorders and
# the like are excluded: they are plumbing, and counting them would dilute the
# "did AMX do the work?" signal.
_COMPUTE_KINDS = {"convolution", "inner_product", "matmul", "deconvolution"}


@dataclass
class IsaUsage:
    """What oneDNN ACTUALLY dispatched, as opposed to what it was allowed to.

    The distinction matters enormously. On a fp32 model oneDNN happily reports
    an AMX-capable ISA and then runs every convolution on AVX-512, because AMX
    has no fp32 path. Reporting only the banner would let the demo claim "AMX
    confirmed" while the AMX tiles sat idle for the entire run.
    """

    impl_counts: dict[str, int] = field(default_factory=dict)
    compute_impl_counts: dict[str, int] = field(default_factory=dict)
    compute_total: int = 0
    compute_amx: int = 0

    @property
    def amx_used(self) -> bool:
        return self.compute_amx > 0

    @property
    def amx_fraction(self) -> float:
        if not self.compute_total:
            return 0.0
        return self.compute_amx / self.compute_total

    @property
    def top_impls(self) -> list[tuple[str, int]]:
        """Busiest COMPUTE implementations. Reorders are excluded on purpose:
        they are data shuffling, and listing them alongside convolutions makes
        it look like AMX had work it declined."""
        return sorted(self.compute_impl_counts.items(), key=lambda kv: -kv[1])[:4]

    def summary(self) -> str:
        if not self.compute_total:
            return "no oneDNN compute primitives observed"
        if not self.compute_amx:
            names = ", ".join(f"{impl}×{n}" for impl, n in self.top_impls if impl)
            return (
                f"0 of {self.compute_total} compute primitives used AMX "
                f"({names or 'implementation not reported'})"
            )
        return (
            f"{self.compute_amx} of {self.compute_total} compute primitives "
            f"used AMX ({self.amx_fraction:.0%})"
        )


@dataclass
class StageProgress:
    name: str
    started: bool = False
    finished: bool = False
    percent: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def label(self) -> str:
        return STAGE_LABELS.get(self.name, self.name)

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at
        return None if end is None else end - self.started_at


class LogParser:
    """Incremental parser fed one log line at a time as the container runs."""

    def __init__(self) -> None:
        self.stages: dict[str, StageProgress] = {s: StageProgress(s) for s in STAGES}
        self.reported_isa: str | None = None
        self.current: str | None = None
        self.errors: list[str] = []
        self.isa_usage = IsaUsage()

    def feed(self, line: str, timestamp: float) -> None:
        stripped = line.strip()
        if not stripped:
            return

        for pattern in _ISA_PATTERNS:
            match = pattern.search(stripped)
            if match and not self.reported_isa:
                self.reported_isa = match.group("isa").strip()
                break

        exec_match = _PRIMITIVE_EXEC.search(stripped)
        if exec_match:
            kind = exec_match.group("kind").lower()
            impl = exec_match.group("impl").strip()
            usage = self.isa_usage
            usage.impl_counts[impl] = usage.impl_counts.get(impl, 0) + 1
            if kind in _COMPUTE_KINDS:
                usage.compute_total += 1
                usage.compute_impl_counts[impl] = usage.compute_impl_counts.get(impl, 0) + 1
                if "amx" in impl.lower():
                    usage.compute_amx += 1
            return  # verbose exec lines carry nothing else we need

        for name, pattern in _STAGE_PATTERNS.items():
            if not pattern.search(stripped):
                continue
            stage = self.stages[name]
            if not stage.started:
                stage.started = True
                stage.started_at = timestamp
                # Entering a stage means every earlier stage is done.
                for earlier in STAGES[: STAGES.index(name)]:
                    prior = self.stages[earlier]
                    if prior.started and not prior.finished:
                        prior.finished = True
                        prior.finished_at = timestamp
                        prior.percent = 100.0
                self.current = name
            break

        if self.current:
            match = _PERCENT.search(stripped)
            if match:
                pct = min(100.0, float(match.group(1)))
                stage = self.stages[self.current]
                stage.percent = max(stage.percent, pct)

        lowered = stripped.lower()
        if any(tok in lowered for tok in ("traceback (most recent call last)", "error:", "fatal")):
            self.errors.append(stripped)

    def finalize(self, timestamp: float) -> None:
        """Mark everything complete after a successful exit."""
        for stage in self.stages.values():
            if stage.started and not stage.finished:
                stage.finished = True
                stage.finished_at = timestamp
            if stage.finished:
                stage.percent = 100.0

    @property
    def overall_percent(self) -> float:
        return sum(s.percent for s in self.stages.values()) / len(STAGES)


def isa_is_consistent(requested_ceiling: str, reported_isa: str | None) -> bool | None:
    """Check the ISA oneDNN actually selected against the ceiling we requested.

    Returns True/False, or None when oneDNN did not report an ISA at all (in
    which case the UI must say "not reported" rather than claiming a match).
    """
    if not reported_isa:
        return None
    lowered = reported_isa.lower()
    mentions_amx = "amx" in lowered
    wants_amx = "amx" in requested_ceiling.lower()
    return mentions_amx == wants_amx


@dataclass
class VariantCounts:
    total: int = 0
    snps: int = 0
    indels: int = 0
    other: int = 0
    passing: int = 0
    het: int = 0
    hom_alt: int = 0
    examples: list[dict[str, str]] = field(default_factory=list)

    @property
    def ti_tv_note(self) -> str:
        return f"{self.snps:,} SNPs · {self.indels:,} indels"


def _open_vcf(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def parse_vcf(path: Path, max_examples: int = 8) -> VariantCounts:
    """Count real variants from a produced VCF.

    Only PASS/`.` records are counted as passing; the totals reflect what the
    pipeline actually emitted.
    """
    counts = VariantCounts()
    if not path.exists():
        return counts

    with _open_vcf(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos, _id, ref, alt, qual, filt = fields[:7]
            if alt in (".", "<*>", ""):
                continue

            alts = [a for a in alt.split(",") if a not in (".", "<*>")]
            if not alts:
                continue

            counts.total += 1
            is_snp = len(ref) == 1 and all(len(a) == 1 for a in alts)
            is_indel = any(len(a) != len(ref) for a in alts)
            if is_snp:
                counts.snps += 1
            elif is_indel:
                counts.indels += 1
            else:
                counts.other += 1

            if filt in ("PASS", "."):
                counts.passing += 1

            if len(fields) >= 10:
                sample = fields[9].split(":")[0].replace("|", "/")
                alleles = sample.split("/")
                if len(alleles) == 2 and all(a.isdigit() for a in alleles):
                    a1, a2 = int(alleles[0]), int(alleles[1])
                    if a1 != a2:
                        counts.het += 1
                    elif a1 > 0:
                        counts.hom_alt += 1

            if len(counts.examples) < max_examples and filt in ("PASS", "."):
                counts.examples.append(
                    {
                        "CHROM": chrom,
                        "POS": pos,
                        "REF": ref if len(ref) <= 12 else ref[:11] + "…",
                        "ALT": alt if len(alt) <= 12 else alt[:11] + "…",
                        "QUAL": qual,
                        "FILTER": filt,
                        "TYPE": "SNP" if is_snp else ("INDEL" if is_indel else "OTHER"),
                    }
                )

    return counts


def find_output_vcf(run_dir: Path) -> Path | None:
    """Locate the primary VCF a run produced, ignoring the much larger gVCF."""
    candidates = sorted(run_dir.glob("*.vcf.gz")) + sorted(run_dir.glob("*.vcf"))
    non_gvcf = [c for c in candidates if ".g.vcf" not in c.name]
    return (non_gvcf or candidates or [None])[0]
