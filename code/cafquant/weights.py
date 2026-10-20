"""Weight/model availability for the ten backends + optional downloads.

Nothing here downloads unless you call :func:`download`; the normal path is a
cheap existence check so a run can skip backends whose weights are missing.
Weights live in the shared caches used by the other projects on this machine
(~/.cache/fairchem, ~/.cache/mace, ~/.cache/aimnet).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .mlips import EGRET_DEFAULT_MODEL, find_omol_model

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"

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

    # egret: compiled TorchScript model in the repo weights/ dir (or wherever
    # CAFQUANT_EGRET_MODEL points) -- no public download
    egret_model = Path(os.environ.get("CAFQUANT_EGRET_MODEL", EGRET_DEFAULT_MODEL))
    out["egret"] = {"ok": egret_model.is_file(),
                    "detail": str(egret_model) if egret_model.is_file()
                    else f"missing: {egret_model} (set CAFQUANT_EGRET_MODEL to override)"}

    # aimnet2
    aim_file = AIMNET_CACHE / "aimnet2_2025_b973c_d3_0.pt"
    out["aimnet2"] = {"ok": aim_file.is_file(),
                      "detail": str(aim_file) if aim_file.is_file()
                      else f"missing: {aim_file}"}

    # uma1p2: a uma-s-1p2* checkpoint in the fairchem HF cache
    out["uma1p2"] = {"ok": bool(uma_1p2_ckpts := sorted(
                        (fc / "models--facebook--UMA").glob("snapshots/*/checkpoints/uma-s-1p2*.pt"))),
                     "detail": str(uma_1p2_ckpts[0]) if uma_1p2_ckpts
                               else f"no uma-s-1p2 checkpoint under {fc}"}

    # polar1-s / polar1-m: MACE-POLAR-1 checkpoints cached by mace
    for key, fname in (("polar1-s", "MACEPOLAR1Smodel"), ("polar1-m", "MACEPOLAR1Mmodel")):
        f = MACE_CACHE / fname
        out[key] = {"ok": f.is_file(), "detail": str(f) if f.is_file()
                    else f"missing: {f} (run --download)"}

    # omol: repo weights/ or mace cache (shared resolver in mlips)
    omol_model = find_omol_model()
    out["omol"] = {"ok": omol_model is not None,
                   "detail": omol_model if omol_model else
                   "missing: MACE-omol-0 model (repo weights/ or --download)"}

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
    if method == "uma1p2":
        from fairchem.core.calculate.pretrained_mlip import (
            pretrained_checkpoint_path_from_name,
        )
        p = pretrained_checkpoint_path_from_name("uma-s-1p2")
        return f"uma1p2 weights at {p}"
    if method.startswith("polar1"):
        from mace.calculators.foundations_models import download_mace_polar_checkpoint
        name = {"polar1-s": "polar-1-s", "polar1-m": "polar-1-m"}[method]
        p = download_mace_polar_checkpoint(name)  # cached under ~/.cache/mace
        return f"{method} weights at {p}"
    if method == "omol":
        local = WEIGHTS_DIR / "MACE-omol-0-extra-large-1024.model"
        if local.is_file():
            return f"omol weights at {local} (repo-local)"
        import cafquant.env
        cafquant.env.patch_torch_for_e3nn()
        from mace.calculators.foundations_models import mace_omol
        mace_omol(device="cpu")  # no model arg -> downloads the extra-large default
        return "omol weights downloaded via mace_omol()"
    if method == "maceoff":
        import urllib.request
        MACE_CACHE.mkdir(parents=True, exist_ok=True)
        dst = MACE_CACHE / "MACE-OFF23_small.model"
        url = ("https://raw.githubusercontent.com/ACEsuit/mace-off/main/"
               "mace_off23/MACE-OFF23_small.model")
        urllib.request.urlretrieve(url, dst)
        return f"maceoff weights at {dst}"
    if method == "egret":
        return ("egret models are distributed separately (no public download); drop the "
                "compiled .model file (EGRET_1.model, EGRET_1E.model, ...) into "
                f"{WEIGHTS_DIR.name}/ or point CAFQUANT_EGRET_MODEL at it")
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