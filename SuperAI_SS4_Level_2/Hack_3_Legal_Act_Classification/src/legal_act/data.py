"""Kaggle download + CSV -> tidy frames (signer lists, committee index, clause table).

Everything this module writes lands under `data.root` (gitignored), so a fresh clone rebuilds
its dataset with one command instead of carrying it in git.
"""

from __future__ import annotations

import ast
import logging
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .utils import normalise_name

log = logging.getLogger("legal_act.data")


def _kaggle_bin() -> str:
    exe = shutil.which("kaggle")
    if exe is None:
        raise RuntimeError(
            "kaggle CLI not found. Install it with `uv tool install kaggle`, "
            "then authenticate with `kaggle auth login`."
        )
    return exe


def download(cfg, force: bool = False) -> Path:
    """Download and unzip the competition data into `data.root`."""
    dest = cfg.resolve(cfg.data.root)
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".downloaded"
    if marker.exists() and not force:
        log.info("dataset already present at %s (use --force to re-download)", dest)
        return dest

    kind = cfg.data.kaggle_kind
    if kind == "competition":
        cmd = [_kaggle_bin(), "competitions", "download", "-c", cfg.data.kaggle_slug, "-p", str(dest)]
    elif kind == "dataset":
        cmd = [_kaggle_bin(), "datasets", "download", "-d", cfg.data.kaggle_slug, "-p", str(dest)]
    else:
        raise ValueError(f"data.kaggle_kind must be 'dataset' or 'competition', got {kind!r}")

    log.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    for zf in dest.glob("*.zip"):
        log.info("unzipping %s", zf.name)
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest)
        zf.unlink()

    marker.write_text(f"{kind}:{cfg.data.kaggle_slug}\n", encoding="utf-8")
    log.info("dataset ready at %s", dest)
    return dest


def _read(cfg, name: str) -> pd.DataFrame:
    path = cfg.resolve(cfg.data.root) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `make data` first.")
    return pd.read_csv(path)


def parse_signers(cell: str) -> list[str]:
    """`question` is a stringified Python list, but a few rows wrap across lines.

    ast.literal_eval chokes on those, so pull the quoted names out with a regex instead.
    """
    text = str(cell)
    try:
        value = ast.literal_eval(text)
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value]
    except (ValueError, SyntaxError):
        pass
    return [m.strip() for m in re.findall(r"'([^']*)'", text)]


def _rgno_key(value) -> str:
    """rgno arrives as 105529030059.0 in train and 105510003412 in test — same company."""
    return str(int(float(value)))


@dataclass
class Committee:
    """Directors of one company, keyed for name comparison."""

    rgno: str
    names: list[str]                # display names, in registration order
    by_key: dict[str, str]          # normalise_name(...) -> display name

    def __len__(self) -> int:
        return len(self.names)


def load_committees(cfg) -> dict[str, Committee]:
    df = _read(cfg, cfg.data.committee_csv)
    df["rg"] = df["rgno"].map(_rgno_key)
    out: dict[str, Committee] = {}
    for rg, g in df.groupby("rg"):
        g = g.sort_values("line")
        names = [f"{str(f).strip()} {str(l).strip()}" for f, l in zip(g["fname"], g["lname"])]
        out[rg] = Committee(rgno=rg, names=names, by_key={normalise_name(n): n for n in names})
    log.info("committee.csv: %d companies, %d directors", len(out), sum(len(c) for c in out.values()))
    return out


def load_patterns(cfg) -> dict[int, str]:
    df = _read(cfg, cfg.data.patterns_csv)
    return {int(p): str(t).strip() for p, t in zip(df["Pattern"], df.iloc[:, 2])}


def load_split(cfg, split: str) -> pd.DataFrame:
    """Read train/test and add the derived columns every later stage relies on."""
    name = cfg.data.train_csv if split == "train" else cfg.data.test_csv
    df = _read(cfg, name)
    df["rg"] = df["rgno"].map(_rgno_key)
    df["signers"] = df["question"].map(parse_signers)
    df["legal_act"] = df["legal_act"].fillna("").astype(str)
    df["condition"] = df["condition"].fillna("").astype(str)
    df["clause_id"] = [clause_id(rg, ctx) for rg, ctx in zip(df["rg"], df["context"])]
    df["split"] = split
    log.info("%s: %d rows, %d clauses, %d companies", split, len(df), df["clause_id"].nunique(), df["rg"].nunique())
    return df


def clause_id(rg: str, context: str) -> str:
    """One id per (company, clause text) — this is the unit the LLM is asked about."""
    import hashlib

    h = hashlib.sha1(f"{rg}\x00{context}".encode("utf-8")).hexdigest()[:12]
    return f"{rg}-{h}"


def clause_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Collapse rows to the unique clauses — 10,264 rows become 308 LLM calls."""
    parts = []
    for df in frames:
        cols = ["clause_id", "rg", "context", "pattern", "condition", "split"]
        parts.append(df[cols])
    allrows = pd.concat(parts, ignore_index=True)
    agg = (
        allrows.groupby("clause_id")
        .agg(
            rg=("rg", "first"),
            context=("context", "first"),
            pattern=("pattern", "first"),
            conditions=("condition", lambda s: sorted({v for v in s if v})),
            splits=("split", lambda s: sorted(set(s))),
            n_rows=("clause_id", "size"),
        )
        .reset_index()
    )
    return agg.sort_values("n_rows", ascending=False).reset_index(drop=True)


def legal_acts_per_clause(frames: list[pd.DataFrame]) -> dict[str, list[str]]:
    """Every distinct legal_act a clause is asked about — the input to the scope stage."""
    allrows = pd.concat([df[["clause_id", "legal_act"]] for df in frames], ignore_index=True)
    return {cid: sorted(set(g)) for cid, g in allrows.groupby("clause_id")["legal_act"]}
