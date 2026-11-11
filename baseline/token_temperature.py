#!/usr/bin/env python3
"""Token-level temperature baseline (paper Eq. 6).

Rescales every conditional distribution of an existing policy artifact by
pi(y_i | y_<i)^(1/t) and renormalises row by row.  Applied to the DPO policy
this is the paper's "DPO with token-level temperature" baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FILES = ("two_outcome.json", "sequence.json", "calibration.json")


def temper(row, t):
    r = [max(x, 0.0) ** (1.0 / t) for x in row]
    s = sum(r)
    return [x / s for x in r]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="checkpoints/dpo_alpha_equals_beta")
    parser.add_argument("--output-dir", default="checkpoints/dpo_token_temperature_1.5")
    parser.add_argument("--temperature", type=float, default=1.5)
    args = parser.parse_args()
    src, dst = Path(args.source_dir), Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        payload = json.loads((src / name).read_text())
        for ctx in payload["contexts"].values():
            ctx["initial"] = temper(ctx["initial"], args.temperature)
            if "transitions" in ctx:
                ctx["transitions"] = [[temper(row, args.temperature) for row in Ti] for Ti in ctx["transitions"]]
        payload["method_name"] = dst.name
        (dst / name).write_text(json.dumps(payload) + "\n")
    (dst / "config.json").write_text(json.dumps({"source": str(src), "token_temperature": args.temperature}, indent=2) + "\n")
    print(f"wrote token-temperature artifacts to {dst}")


if __name__ == "__main__":
    main()
