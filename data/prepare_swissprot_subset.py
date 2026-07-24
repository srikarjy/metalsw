#!/usr/bin/env python3
"""Prepare the pinned Swiss-Prot benchmark subset for metalsw v0.4.

Usage: python3 data/prepare_swissprot_subset.py <path-to-uniprot_sprot.fasta>

Does NOT download or commit the ~288MB raw Swiss-Prot release into the repo.
Instead it deterministically derives a small, pinned subset from a local copy
of the release and writes that subset (a few MB) to
data/swissprot_subset.fasta, which IS committed.

Provenance of the release used to build the committed subset (recorded here,
not re-fetched by this script, since UniProt's "current_release" URL is a
moving target over time):
    Source:        https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
    Downloaded:     2026-07-23
    Last-Modified:  Wed, 10 Jun 2026 20:00:00 GMT
    ETag:           "595d8e5-653ebb0451000"
    Content-Length: 93706469 (compressed .gz)

Selection recipe (deterministic, no RNG):
    1. Parse all records.
    2. Keep only sequences with length <= MAX_LEN residues.
    3. Take the first SUBSET_SIZE of those, in file order.
"""
import sys

MAX_LEN = 2000
SUBSET_SIZE = 750


def parse_fasta(path):
    with open(path) as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            yield header, "".join(seq_lines)


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path-to-uniprot_sprot.fasta>", file=sys.stderr)
        sys.exit(1)

    src_path = sys.argv[1]
    kept = []
    for header, seq in parse_fasta(src_path):
        if len(seq) <= MAX_LEN:
            kept.append((header, seq))
            if len(kept) >= SUBSET_SIZE:
                break

    with open("data/swissprot_subset.fasta", "w") as out:
        for header, seq in kept:
            out.write(f">{header}\n{seq}\n")

    print(f"wrote data/swissprot_subset.fasta: {len(kept)} sequences "
          f"(filter: length <= {MAX_LEN}, first {SUBSET_SIZE} after filtering)")


if __name__ == "__main__":
    main()
