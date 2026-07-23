#!/usr/bin/env python3
"""Deterministic generator for metalsw's Stage 0 test set.

Regenerate with: python3 data/generate_test_set.py
Writes data/query.fasta and data/db_small.fasta. Seeded (RNG seed = 42) so the
corpus is reproducible byte-for-byte across machines and sessions.
"""
import random

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"
SEED = 42


def write_fasta(path, records):
    with open(path, "w") as f:
        for rec_id, seq in records:
            f.write(f">{rec_id}\n{seq}\n")


def main():
    rng = random.Random(SEED)
    query = "".join(rng.choice(AMINO_ACIDS) for _ in range(50))

    db = []
    # db01: identical to query -> expect the maximum possible score (self-alignment).
    db.append(("db01_identical", query))

    # db02: one conservative point mutation (mid-sequence) -> expect just below db01.
    mutated = list(query)
    mutated[25] = "I" if mutated[25] != "I" else "L"
    db.append(("db02_point_mutation", "".join(mutated)))

    # db03: first 30 residues match query exactly, tail replaced with unrelated
    # padding -> tests that SW finds the best *local* region, not a global match.
    db.append(("db03_partial_match", query[:30] + "AAAAAAAAAAAAAAAAAAAA"))

    # db04: same composition as query, residues shuffled -> expect low score.
    shuffled = list(query)
    rng.shuffle(shuffled)
    db.append(("db04_shuffled", "".join(shuffled)))

    # db05: query reversed -> expect low score (same composition, different order).
    db.append(("db05_reversed", query[::-1]))

    # db06: independently drawn random sequence -> expect near-zero score (unrelated).
    unrelated = "".join(rng.choice(AMINO_ACIDS) for _ in range(50))
    db.append(("db06_unrelated", unrelated))

    write_fasta("data/query.fasta", [("query", query)])
    write_fasta("data/db_small.fasta", db)
    print(f"wrote data/query.fasta and data/db_small.fasta (seed={SEED})")


if __name__ == "__main__":
    main()
