"""Weight/model availability for the five backends + optional downloads.

Nothing here downloads unless you call :func:`download`; the normal path is a
cheap existence check so a run can skip backends whose weights are missing.
Weights live in the shared caches used by the other projects on this machine
(~/.cache/fairchem, ~/.cache/mace, ~/.cache/aimnet).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

MACE_CACHE = Path(__import__("os").environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "mace"
AIMNET_CACHE = Path(os.environ.get("AIMNET_CACHE_DIR", str(Path.home() / ".cache" / "aimnet")))


def _cache_dir(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, str(default)))


def check_weights() -> dict[str, dict]:
    """Report status per method: {"ok": bool, "detail": path or note}."""
    out: dict[str, dict] = {}

    # uma: any fairchem HF snapshot checkpoint
    fc = _cache_dir("FAIRCHEM_CACHE_DIR", Path.home() / ".cache" / "fairchem")
    uma_ckpts = sorted((fc / "models--facebook--UMA").glob("snapshots/*/checkpoints/*.pt"))
    out["uma"] = {"ok": bool(uma_ckpts),
                  "detail": str(uma_ckpts[0]) if uma_ckpts else f"no uma checkpoint under {fc}"}

    # maceoff
    mace_file = MACE_CACHE / "MACE-OFF23_small.model"
    out["maceoff"] = {"ok": mace_file.is_file(),
                      "detail": str(mace_file) if mace_file.is_file() else f"missing: {mace_file}"}

    # aimnet2
    aim_file = AIMNET_CACHE / "aimnet2_2025_b973c_d3_0.pt"
    out["aimnet2"] = {"ok": aim_file.is_file(),
                      "detail": str(aim_file) if aim_file.is_file()
                      else f"missing: {aim_file}"}

    # gfn2: params are embedded in the tblite wheel
    try:
        import tblite.ase  # noqa: F401
        out["gfn2"] = {"ok": True, "detail": "tblite installed; GFN2 params embedded (no weights)"}
    except Exception as exc:
        out["gfn2"] = {"ok": False, "detail": f"tblite import failed: {exc}"}

    # pm7: mopac binary on PATH or ~/.local/bin
    exe = shutil.which("mopac") or (Path.home() / ".local/bin/mopac")
    ok = Path(exe).is_file() and shutil.which("mopac") is not None or (
        Path(exe).is_file() and os.access(exe, os.X_OK))
    out["pm7"] = {"ok": bool(ok), "detail": str(exe) if ok else
                  "missing: mopac binary (conda create -n mopac -c conda-forge mopac)"}
    return out


def check_available() -> list[str]:
    """Method names whose weights are available."""
    return [m for m, r in check_weights().items() if r["ok"]]


def format_report(report: dict[str, dict] | None = None) -> str:
    r = report or check_weights()
    lines = ["method     weights   detail"]
    for m, rec in r.items():
        lines.append(f"{m:10} {'OK' if rec['ok'] else 'MISSING':9} {rec['detail']}")
    return "\n".join(lines)


def download(method: str) -> str:
    """Fetch missing weights for one method. Returns a human-readable message."""
    import cafquant.env
    cafquant.env.patch_torch_for_e3nn()  # before fairchem/mace import e3nn

    if method == "uma":
        from fairchem.core.calculate.pretrained_mlip import (
            pretrained_checkpoint_path_from_name,
        )
        p = pretrained_checkpoint_path_from_name("uma-s-1p1")  # HF download into cache
        return f"uma weights at {p}"
    if method == "maceoff":
        import urllib.request
        MACE_CACHE.mkdir(parents=True, exist_ok=True)
        dst = MACE_CACHE / "MACE-OFF23_small.model"
        url = ("https://raw.githubusercontent.com/ACEsuit/mace-off/main/"
               "mace_off23/MACE-OFF23_small.model")
        urllib.request.urlretrieve(url, dst)
        return f"maceoff weights at {dst}"
    if method == "aimnet2":
        from aimnet.calculators.model_registry import get_model_path
        p = get_model_path("aimnet2-2025")  # sha-256-validated download if absent
        return f"aimnet2 assets at {p}"
    if method == "gfn2":
        return "gfn2 needs no download (params are embedded in tblite)"
    if method == "pm7":
        return ("pm7 needs the mopac binary: conda create -n mopac -c conda-forge mopac, "
                "then symlink .../envs/mopac/bin/mopac to ~/.local/bin/mopac")
    raise ValueError(f"unknown method: {method}")