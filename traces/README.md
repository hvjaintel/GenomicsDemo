# Replay traces

This directory is **intentionally empty**.

A trace in here is a JSON recording of a run that genuinely happened on this
machine. DEMO MODE replays it at real speed behind a permanent
"REPLAY — not a live run" banner, so booth staff can keep talking when Docker,
the network, or the data volume is unavailable.

**Traces are never authored by hand.** Every field — wall-clock seconds, stage
boundaries, the oneDNN ISA banner, the variant counts — comes from
`app/replay.py`'s recorder wrapping a real `DeepVariantRunner` execution. A
hand-written trace would make the demo a lie, and the speedup it displays would
be fabricated.

## Recording a trace

Once Docker access is available and a real race has completed:

```bash
./scripts/preflight.sh --smoke          # confirm the pipeline runs for real
./run_demo.sh                           # then run an AMX race in the UI
```

Enable `demo_mode.record: true` in `config.yaml` before the race, and the
runner writes `traces/<sample>-<timestamp>.json` on completion.

## Verifying a trace

```bash
.venv/bin/python -m app.replay --list          # show traces and their provenance
.venv/bin/python -m app.replay --verify FILE   # check required fields are present
```

`app/replay.py` refuses to load a trace that is missing its `recorded_at`,
`host`, or per-leg `reported_isa` fields — the markers that distinguish a
recording from an invention.
