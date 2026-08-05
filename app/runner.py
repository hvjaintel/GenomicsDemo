"""Docker orchestration for DeepVariant, and the AMX toggle itself.

THE CENTRAL GUARANTEE OF THIS MODULE
------------------------------------
An AMX-ON run and an AMX-OFF run differ by exactly one environment variable:

    ONEDNN_MAX_CPU_ISA = AVX512_CORE_AMX   (AMX ON)
    ONEDNN_MAX_CPU_ISA = AVX512_CORE       (AMX OFF)

Same image, same BAM, same reference, same shard count, same NUMA policy, same
thread counts. Both legs of a race are built from a single `RunSpec`, so the
parameters cannot drift apart and quietly invalidate the comparison.

We also ask oneDNN to report the ISA it actually selected and parse that back
out of the logs, so the toggle is *verifiable* rather than merely asserted. If
the reported ISA contradicts the requested ceiling, the run is marked failed
rather than reporting a number we cannot stand behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config
from .parsing import (
    LogParser,
    STAGES,
    VariantCounts,
    find_output_vcf,
    isa_is_consistent,
    parse_vcf,
)

CONTAINER_REF_DIR = "/ref"
CONTAINER_IN_DIR = "/input"
CONTAINER_OUT_DIR = "/output"
# Read-only mount point for app/container_inject, which carries the
# sitecustomize.py that switches DeepVariant's inference to bf16.
CONTAINER_INJECT_DIR = "/genomics-demo-inject"
INJECT_SOURCE_DIR = Path(__file__).resolve().parent / "container_inject"


class RunnerError(RuntimeError):
    pass


@dataclass
class RunSpec:
    """Everything needed to launch one run. Shared verbatim across a race."""

    sample_id: str
    bam: Path
    reference: Path
    regions: str | None
    num_shards: int
    engine_image: str
    model_type: str = "WGS"
    numa_policy: str = "interleave_all"
    numa_node: int = 0
    shm_size: str | None = "16g"
    memory: str | None = None
    ulimit_nofile: str | None = "65536:524288"
    # Docker --cpuset-cpus for this leg. This is the ONE permitted difference
    # in a core-scaling race (see docs/AMX-FINDINGS.md for why the AMX race
    # was retired), so like the AMX state it is excluded from the fingerprint.
    # None means "the whole machine".
    cpuset: str | None = None

    @property
    def core_count(self) -> int | None:
        """How many logical CPUs this leg may use, or None for all of them."""
        if not self.cpuset:
            return None
        total = 0
        for part in self.cpuset.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                total += int(hi) - int(lo) + 1
            else:
                total += 1
        return total

    def fingerprint(self) -> str:
        """Identity of everything that must be equal across the two legs.

        Deliberately excludes the AMX state and the cpuset — those are the
        permitted differences, one per race mode. The race view compares
        fingerprints and refuses to report a speedup if they differ.
        """
        payload = {
            "sample_id": self.sample_id,
            "bam": str(self.bam),
            "reference": str(self.reference),
            "regions": self.regions,
            "num_shards": self.num_shards,
            "engine_image": self.engine_image,
            "model_type": self.model_type,
            "numa_policy": self.numa_policy,
            "numa_node": self.numa_node,
        }
        return json.dumps(payload, sort_keys=True)

    def fingerprint_digest(self) -> str:
        """Short stable hash of the fingerprint, for display only.

        Comparisons always use the full fingerprint; this is just so the UI and
        CLI can show something a human can eyeball across two legs.
        """
        return hashlib.sha256(self.fingerprint().encode()).hexdigest()[:12]


@dataclass
class RunResult:
    run_id: str
    amx_on: bool
    requested_isa: str
    reported_isa: str | None = None
    isa_verified: bool | None = None
    # How many oneDNN compute primitives actually ran on AMX kernels. The ISA
    # banner says what was allowed; this says what happened.
    amx_primitives: int = 0
    compute_primitives: int = 0
    isa_impl_summary: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    stage_durations: dict[str, float] = field(default_factory=dict)
    variant_counts: VariantCounts | None = None
    vcf_path: str | None = None
    output_dir: str | None = None
    error: str | None = None
    cancelled: bool = False
    fingerprint: str = ""
    command: str = ""

    @property
    def wall_clock_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.error and not self.cancelled

    def to_dict(self) -> dict:
        data = asdict(self)
        data["wall_clock_s"] = self.wall_clock_s
        data["succeeded"] = self.succeeded
        if self.variant_counts is not None:
            data["variant_counts"] = asdict(self.variant_counts)
        return data


@dataclass
class RunEvent:
    """Streamed to the UI as a run progresses."""

    kind: str  # "log" | "progress" | "done"
    run_id: str
    amx_on: bool
    line: str = ""
    elapsed_s: float = 0.0
    overall_percent: float = 0.0
    stages: dict[str, dict] = field(default_factory=dict)
    reported_isa: str | None = None
    result: RunResult | None = None


class DeepVariantRunner:
    """Builds and executes `run_deepvariant` inside Docker, streaming events."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._cancel = threading.Event()
        self._active_container: str | None = None

    @property
    def container_cli(self) -> list[str]:
        """The container command, e.g. ["docker"] or ["podman"].

        Configurable because some sites run podman, and because a freshly
        added `docker` group membership does not apply to already-running
        login sessions -- there, ["sg", "docker", "-c"] style wrappers or a
        full path may be needed. It is NOT part of the run fingerprint: it
        cannot differ between the two legs of a race, since both legs are
        launched by this same runner instance.
        """
        raw = self.cfg.compute.get("container_cli") or ["docker"]
        if isinstance(raw, str):
            return shlex.split(raw)
        return [str(part) for part in raw]

    # -- spec construction ------------------------------------------------

    def build_spec(self, sample_id: str, engine_key: str | None = None) -> RunSpec:
        sample = self.cfg.sample(sample_id)
        engine = self.cfg.engine(engine_key) if engine_key else self.cfg.default_engine
        bam = self.cfg.dataset_path(sample.dataset)
        reference = self._reference_fasta()

        if not bam.exists():
            raise RunnerError(
                f"BAM not staged: {bam}\nRun ./scripts/fetch_data.sh first."
            )
        if not reference.exists():
            raise RunnerError(
                f"Reference not staged: {reference}\nRun ./scripts/fetch_data.sh first."
            )

        compute = self.cfg.compute
        return RunSpec(
            sample_id=sample_id,
            bam=bam,
            reference=reference,
            regions=sample.regions,
            num_shards=self.cfg.num_shards,
            engine_image=engine.image,
            numa_policy=compute.get("numa_policy", "interleave_all"),
            numa_node=int(compute.get("numa_node", 0)),
            shm_size=compute.get("docker_shm_size"),
            memory=compute.get("docker_memory"),
            ulimit_nofile=compute.get("docker_ulimit_nofile"),
        )

    def _reference_fasta(self) -> Path:
        """Prefer the decompressed FASTA; fall back to the bgzipped original."""
        gz = self.cfg.dataset_path("reference")
        plain = Path(str(gz)[:-3]) if str(gz).endswith(".gz") else gz
        return plain if plain.exists() else gz

    # -- command construction ---------------------------------------------

    def _env_for(self, amx_on: bool, verbose_isa: bool = True) -> dict[str, str]:
        """Environment for one leg. The ISA ceiling is the ONLY difference."""
        isa = self.cfg.isa_for(amx_on)
        env = {
            "ONEDNN_MAX_CPU_ISA": isa,
            # Legacy name, honoured by older oneDNN/TF builds. Kept in lockstep
            # so an older image cannot silently ignore the toggle.
            "DNNL_MAX_CPU_ISA": isa,
            # We run the container as the host UID, which has no home directory
            # inside it, so matplotlib fails to build its font cache and prints
            # a multi-line warning per shard. With 192 shards that buries the
            # booth log. Identical for both legs, so it changes no timing.
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "HOME": "/tmp",
        }
        if verbose_isa and self.cfg.amx.get("verify_isa_from_logs", True):
            verbose = str(self.cfg.amx.get("onednn_verbose", 1))
            env["ONEDNN_VERBOSE"] = verbose
            env["DNNL_VERBOSE"] = verbose

        # Per-leg extras from config (e.g. the fp32 math mode). Both legs
        # declare the SAME keys with different values, so the two commands stay
        # structurally identical and every difference is visible in config.yaml.
        leg = self.cfg.amx.get("enabled" if amx_on else "disabled", {})
        for key, value in (leg.get("env") or {}).items():
            env[str(key)] = str(value)

        # bf16 auto-mixed-precision. Without it the AMX tiles never receive any
        # work, because DeepVariant's model is fp32 and AMX has no fp32 path.
        # See app/container_inject/sitecustomize.py for the full reasoning.
        if self.cfg.amx.get("use_bf16_injection", True):
            env["PYTHONPATH"] = CONTAINER_INJECT_DIR
            env["DV_FORCE_BF16"] = "1" if amx_on else "0"
        return env

    def _numactl_prefix(self, spec: RunSpec) -> list[str]:
        if not shutil.which("numactl"):
            return []
        if spec.numa_policy == "interleave_all":
            return ["numactl", "--interleave=all"]
        if spec.numa_policy == "bind_node":
            return [
                "numactl",
                f"--cpunodebind={spec.numa_node}",
                f"--membind={spec.numa_node}",
            ]
        return []

    def build_command(
        self, spec: RunSpec, amx_on: bool, out_dir: Path, verbose_isa: bool = True
    ) -> list[str]:
        """Full `docker run` argv. Identical for both legs bar the ISA env."""
        env = self._env_for(amx_on, verbose_isa)

        cmd: list[str] = [
            *self.container_cli, "run", "--rm",
            "--name", self._container_name(out_dir.name, amx_on),
            "-u", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{spec.reference.parent}:{CONTAINER_REF_DIR}:ro",
            "-v", f"{spec.bam.parent}:{CONTAINER_IN_DIR}:ro",
            "-v", f"{out_dir}:{CONTAINER_OUT_DIR}",
        ]
        if self.cfg.amx.get("use_bf16_injection", True):
            # Mounted for BOTH legs, identically. Only DV_FORCE_BF16 differs,
            # so the AMX-OFF leg loads the same file and it deliberately does
            # nothing. Keeping the mount symmetric means the two command lines
            # differ by environment alone.
            cmd += ["-v", f"{INJECT_SOURCE_DIR}:{CONTAINER_INJECT_DIR}:ro"]
        for key, value in env.items():
            cmd += ["-e", f"{key}={value}"]
        if spec.shm_size:
            cmd += ["--shm-size", str(spec.shm_size)]
        if spec.memory:
            cmd += ["--memory", str(spec.memory)]
        if spec.ulimit_nofile:
            cmd += ["--ulimit", f"nofile={spec.ulimit_nofile}"]
        if spec.cpuset:
            cmd += ["--cpuset-cpus", spec.cpuset]

        cmd.append(spec.engine_image)

        # numactl must run INSIDE the container so it governs the actual
        # DeepVariant processes rather than the docker client.
        cmd += self._numactl_prefix(spec)
        cmd += [
            "/opt/deepvariant/bin/run_deepvariant",
            f"--model_type={spec.model_type}",
            f"--ref={CONTAINER_REF_DIR}/{spec.reference.name}",
            f"--reads={CONTAINER_IN_DIR}/{spec.bam.name}",
            f"--output_vcf={CONTAINER_OUT_DIR}/output.vcf.gz",
            f"--num_shards={spec.num_shards}",
            f"--intermediate_results_dir={CONTAINER_OUT_DIR}/intermediate",
        ]
        if spec.regions:
            cmd.append(f"--regions={spec.regions}")
        return cmd

    @staticmethod
    def _container_name(run_id: str, amx_on: bool) -> str:
        return f"gdemo-{run_id}-{'amxon' if amx_on else 'amxoff'}"

    # -- execution ---------------------------------------------------------

    def cancel(self) -> None:
        """Stop the in-flight container so booth staff can reset instantly."""
        self._cancel.set()
        if self._active_container:
            subprocess.run(
                [*self.container_cli, "kill", self._active_container],
                capture_output=True,
                check=False,
            )

    def reset_cancel(self) -> None:
        self._cancel.clear()

    def run(
        self,
        spec: RunSpec,
        amx_on: bool,
        run_id: str | None = None,
        verbose_isa: bool = True,
        leg_id: str | None = None,
    ) -> Iterator[RunEvent]:
        """Execute one leg, yielding log and progress events as it goes.

        `verbose_isa` controls ONEDNN_VERBOSE. Leave it on to PROVE which
        kernels ran; turn it off to MEASURE. The two cannot be done in the same
        run: oneDNN prints a line per primitive execution, which on a chr20 run
        is a 270 MB log, and bf16 emits more primitives than fp32 -- so the
        instrumentation penalises the AMX leg specifically. Timing with verbose
        on would understate AMX and quietly corrupt the headline number.

        `leg_id` names this leg's output directory and container. The AMX race
        derives it from the AMX state; a core-scaling race has to pass one
        explicitly, since both of its legs share the same AMX setting.
        """
        run_id = run_id or uuid.uuid4().hex[:8]
        suffix = leg_id or ("amx-on" if amx_on else "amx-off")
        out_dir = self.cfg.runs_dir / f"{run_id}-{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "intermediate").mkdir(exist_ok=True)

        requested_isa = self.cfg.isa_for(amx_on)
        cmd = self.build_command(spec, amx_on, out_dir, verbose_isa)
        self._active_container = self._container_name(out_dir.name, amx_on)

        result = RunResult(
            run_id=run_id,
            amx_on=amx_on,
            requested_isa=requested_isa,
            output_dir=str(out_dir),
            fingerprint=spec.fingerprint(),
            command=" ".join(shlex.quote(c) for c in cmd),
        )
        parser = LogParser()
        log_path = out_dir / "run.log"

        start = time.time()
        result.started_at = start

        yield RunEvent(
            kind="log",
            run_id=run_id,
            amx_on=amx_on,
            line=f"$ {result.command}",
        )
        yield RunEvent(
            kind="log",
            run_id=run_id,
            amx_on=amx_on,
            line=f"[demo] requested ONEDNN_MAX_CPU_ISA={requested_isa}",
        )

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            result.error = f"could not launch Docker: {exc}"
            result.exit_code = -1
            result.finished_at = time.time()
            yield RunEvent(kind="done", run_id=run_id, amx_on=amx_on, result=result)
            return

        with log_path.open("w") as log_file:
            assert proc.stdout is not None
            for raw in proc.stdout:
                now = time.time()
                log_file.write(raw)
                line = raw.rstrip("\n")
                parser.feed(line, now)

                # oneDNN verbose is extremely chatty; the per-primitive trace
                # lines would drown the booth log pane. Keep the informational
                # header lines (which carry the ISA) and drop the rest.
                if line.startswith(("onednn_verbose,exec", "dnnl_verbose,exec",
                                    "onednn_verbose,primitive", "mkldnn_verbose,exec")):
                    continue

                yield RunEvent(
                    kind="log",
                    run_id=run_id,
                    amx_on=amx_on,
                    line=line,
                    elapsed_s=now - start,
                    overall_percent=parser.overall_percent,
                    stages=self._stage_payload(parser),
                    reported_isa=parser.reported_isa,
                )

                if self._cancel.is_set():
                    proc.kill()
                    result.cancelled = True
                    break

            proc.wait()

        end = time.time()
        parser.finalize(end)
        result.finished_at = end
        result.exit_code = proc.returncode
        result.reported_isa = parser.reported_isa
        result.isa_verified = isa_is_consistent(requested_isa, parser.reported_isa)
        result.amx_primitives = parser.isa_usage.compute_amx
        result.compute_primitives = parser.isa_usage.compute_total
        result.isa_impl_summary = parser.isa_usage.summary()
        result.stage_durations = {
            name: stage.duration_s
            for name, stage in parser.stages.items()
            if stage.duration_s is not None
        }

        if result.cancelled:
            result.error = "cancelled by operator"
        elif proc.returncode != 0:
            tail = "\n".join(parser.errors[-5:]) or "see run.log for details"
            result.error = f"DeepVariant exited with code {proc.returncode}\n{tail}"
        elif result.isa_verified is False and self.cfg.amx.get("fail_on_isa_mismatch", True):
            # A number we cannot attribute to the requested ISA is worse than
            # no number at all, so we refuse to report it.
            result.error = (
                f"ISA VERIFICATION FAILED — requested {requested_isa} but oneDNN "
                f"reported '{result.reported_isa}'. Refusing to report this "
                f"timing as an AMX {'ON' if amx_on else 'OFF'} result."
            )

        if result.succeeded:
            vcf = find_output_vcf(out_dir)
            if vcf is not None:
                result.vcf_path = str(vcf)
                result.variant_counts = parse_vcf(vcf)

        (out_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2, default=str))
        self._active_container = None

        yield RunEvent(
            kind="done",
            run_id=run_id,
            amx_on=amx_on,
            elapsed_s=end - start,
            overall_percent=100.0 if result.succeeded else parser.overall_percent,
            stages=self._stage_payload(parser),
            reported_isa=result.reported_isa,
            result=result,
        )

    @staticmethod
    def _stage_payload(parser: LogParser) -> dict[str, dict]:
        return {
            name: {
                "label": parser.stages[name].label,
                "started": parser.stages[name].started,
                "finished": parser.stages[name].finished,
                "percent": parser.stages[name].percent,
                "duration_s": parser.stages[name].duration_s,
            }
            for name in STAGES
        }


def time_ratio(candidate: RunResult, baseline: RunResult) -> float | None:
    """How many times faster `candidate` was than `baseline` — or None.

    Returns None if either leg failed or if anything outside the one permitted
    difference (the AMX state for the AMX race, the cpuset for the core-scaling
    race) differed between them. Both of those are excluded from the
    fingerprint, so an unequal fingerprint means the comparison is genuinely
    invalid and no number should be shown at all.
    """
    if not (candidate.succeeded and baseline.succeeded):
        return None
    if candidate.fingerprint != baseline.fingerprint:
        return None
    fast_s, slow_s = candidate.wall_clock_s, baseline.wall_clock_s
    if not fast_s or not slow_s or fast_s <= 0:
        return None
    return slow_s / fast_s


def speedup(amx_on: RunResult, amx_off: RunResult) -> float | None:
    """AMX-ON vs AMX-OFF multiplier — only when the comparison is valid."""
    return time_ratio(amx_on, amx_off)


def throughput_mbases_per_hour(result: RunResult, mbases: float) -> float | None:
    seconds = result.wall_clock_s
    if not seconds or seconds <= 0:
        return None
    return mbases * 3600.0 / seconds
