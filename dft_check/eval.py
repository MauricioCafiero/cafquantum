"""Evaluate delta3 from the ORCA outputs produced by the dft_check inputs.

For each spot triple, reads the seven ORCA output files' final single-point
energies (Eh) and prints delta3 in kcal/mol next to the MLIP values.

Run from dft_check/:  python3 eval.py [--out results.tsv]
Expects <sys>/<stem>.out files next to the .inp/.xyz.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

EH_KCAL = 627.509474

# stem -> (system, frame role in the delta3 formula)
TRIPLES: dict[str, dict[str, str]] = {
    "sult_072_triple_LD_alanine_histidine2": {
        "lab": "sult_072_triple_LD_alanine_histidine2",
        "la": "sult_014_pair_LD_alanine",
        "lb": "sult_018_pair_LD_histidine2",
        "ab": "sult_027_pair_alanine_histidine2",
        "l": "sult_000_monomer_LD",
        "a": "sult_001_monomer_alanine",
        "b": "sult_005_monomer_histidine2",
    },
    "sult_099_triple_LD_histidine2_lysine": {
        "lab": "sult_099_triple_LD_histidine2_lysine",
        "la": "sult_018_pair_LD_histidine2",
        "lb": "sult_019_pair_LD_lysine",
        "ab": "sult_054_pair_histidine2_lysine",
        "l": "sult_000_monomer_LD",
        "a": "sult_005_monomer_histidine2",
        "b": "sult_006_monomer_lysine",
    },
    "ddc_116_triple_LD_histidine1_lysine": {
        "lab": "ddc_116_triple_LD_histidine1_lysine",
        "la": "ddc_018_pair_LD_histidine1",
        "lb": "ddc_021_pair_LD_lysine",
        "ab": "ddc_038_pair_histidine1_lysine",
        "l": "ddc_000_monomer_LD",
        "a": "ddc_002_monomer_histidine1",
        "b": "ddc_005_monomer_lysine",
    },
    "ddc_158_triple_LD_phenylalanine1_proline": {
        "lab": "ddc_158_triple_LD_phenylalanine1_proline",
        "la": "ddc_022_pair_LD_phenylalanine1",
        "lb": "ddc_025_pair_LD_proline",
        "ab": "ddc_080_pair_phenylalanine1_proline",
        "l": "ddc_000_monomer_LD",
        "a": "ddc_006_monomer_phenylalanine1",
        "b": "ddc_009_monomer_proline",
    },
    "comt_210_triple_LD_lysine_magnesiumw": {
        "lab": "comt_210_triple_LD_lysine_magnesiumw",
        "la": "comt_027_pair_LD_lysine",
        "lb": "comt_028_pair_LD_magnesiumw",
        "ab": "comt_105_pair_lysine_magnesiumw",
        "l": "comt_000_monomer_LD",
        "a": "comt_007_monomer_lysine",
        "b": "comt_008_monomer_magnesiumw",
    },
    "comt_222_triple_LD_magnesiumw_tryptophan2": {
        "lab": "comt_222_triple_LD_magnesiumw_tryptophan2",
        "la": "comt_028_pair_LD_magnesiumw",
        "lb": "comt_033_pair_LD_tryptophan2",
        "ab": "comt_117_pair_magnesiumw_tryptophan2",
        "l": "comt_000_monomer_LD",
        "a": "comt_008_monomer_magnesiumw",
        "b": "comt_013_monomer_tryptophan2",
    },
}

# MLIP delta3 (kcal/mol) from the 2026-10-19 assembly, for side-by-side
MLIP = {
    "sult_072_triple_LD_alanine_histidine2": dict(uma=-66.3, maceoff=-0.0, egret=-0.4, aimnet2=6.8, gfn2=-14.1, pm7=-11.8),
    "sult_099_triple_LD_histidine2_lysine": dict(uma=-538.7, maceoff=0.0, egret=0.0, aimnet2=10.3, gfn2=8.8, pm7=3.8),
    "ddc_116_triple_LD_histidine1_lysine": dict(uma=-196.2, maceoff=-0.3, egret=-0.2, aimnet2=3.4, gfn2=4.4, pm7=57.8),
    "ddc_158_triple_LD_phenylalanine1_proline": dict(uma=8.0, maceoff=-4.1, egret=-4.1, aimnet2=6.6, gfn2=43.2, pm7=-18.2),
    "comt_210_triple_LD_lysine_magnesiumw": dict(uma=43.9, gfn2=195.0, pm7=23.9),
    "comt_222_triple_LD_magnesiumw_tryptophan2": dict(uma=72.6, gfn2=209.1, pm7=31.6),
}

PAT = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")


def final_energy_eh(path: Path) -> float | None:
    m = PAT.findall(path.read_text(errors="replace"))
    return float(m[-1]) if m else None


def main() -> None:
    here = Path(__file__).resolve().parent
    rows = []
    for tname, parts in TRIPLES.items():
        eh = {}
        for role, stem in parts.items():
            out = here / f"{stem.split('_')[0]}" / f"{stem}.out"
            e = final_energy_eh(out) if out.exists() else None
            eh[role] = e
        if any(v is None for v in eh.values()):
            missing = [r for r, v in eh.items() if v is None]
            print(f"{tname}: missing {missing} -- skipped")
            continue
        d3 = (eh["lab"] - eh["la"] - eh["lb"] - eh["ab"]
              + eh["l"] + eh["a"] + eh["b"]) * EH_KCAL
        ref = MLIP[tname]
        refline = "  ".join(f"{k}:{v:+.1f}" for k, v in ref.items())
        print(f"{tname}: DFT delta3 = {d3:+7.2f} kcal/mol   (MLIP: {refline})")
        rows.append((tname, d3))
    out = here / "dft_delta3.tsv"
    if rows:
        out.write_text("system_frame\tdft_delta3_kcal\n" +
                       "".join(f"{n}\t{v:.2f}\n" for n, v in rows))


if __name__ == "__main__":
    main()