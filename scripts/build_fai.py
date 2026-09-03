#!/usr/bin/env python3
"""Build a samtools-compatible .fai index for an uncompressed FASTA.

The booth box may not have samtools installed, and pulling the DeepVariant
container just to index a file is a poor trade on show morning. The .fai format
is simple and fully specified, so we generate it directly:

    name  length  offset  linebases  linewidth

  name       sequence name (first token of the header line)
  length     total bases in the sequence
  offset     byte offset of the first base
  linebases  bases per line
  linewidth  bytes per line (including the newline)

This refuses to index a FASTA with ragged line lengths, exactly as samtools
does, because the format cannot represent it.
"""

from __future__ import annotations

import sys
from pathlib import Path


def build_fai(fasta: Path, out: Path | None = None) -> Path:
    out = out or Path(str(fasta) + ".fai")
    records: list[tuple[str, int, int, int, int]] = []

    name: str | None = None
    length = 0
    offset = 0
    linebases = 0
    linewidth = 0
    ragged = False

    with fasta.open("rb") as handle:
        pos = 0
        for raw in handle:
            line_len = len(raw)
            if raw.startswith(b">"):
                if name is not None:
                    records.append((name, length, offset, linebases, linewidth))
                name = raw[1:].split()[0].decode() if len(raw) > 1 else ""
                length = 0
                offset = pos + line_len
                linebases = 0
                linewidth = 0
                ragged = False
            else:
                bases = len(raw.rstrip(b"\r\n"))
                if linebases == 0:
                    linebases = bases
                    linewidth = line_len
                elif bases and bases != linebases:
                    # A shorter line is only legal as the final line of a record.
                    if ragged:
                        raise ValueError(
                            f"{fasta.name}: sequence '{name}' has ragged line lengths "
                            "and cannot be indexed"
                        )
                    ragged = True
                length += bases
            pos += line_len

    if name is not None:
        records.append((name, length, offset, linebases, linewidth))

    with out.open("w") as handle:
        for rec in records:
            handle.write("\t".join(str(v) for v in rec) + "\n")
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: build_fai.py <reference.fasta>", file=sys.stderr)
        return 2
    fasta = Path(argv[1])
    if not fasta.exists():
        print(f"no such file: {fasta}", file=sys.stderr)
        return 1
    out = build_fai(fasta)
    print(f"wrote {out} ({sum(1 for _ in out.open())} sequences)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
