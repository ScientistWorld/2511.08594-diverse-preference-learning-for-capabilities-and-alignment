#!/usr/bin/env python3
"""Untrained baseline: report the fixed reference (pre-preference-learning) policy itself."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="checkpoints/reference_policy")
    args = parser.parse_args()
    data, dst = Path(args.data_dir), Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    two = json.loads((data / "two_outcome_contexts.json").read_text())
    cal = json.loads((data / "calibration_contexts.json").read_text())
    seq = json.loads((data / "sequence_contexts.json").read_text())
    (dst / "two_outcome.json").write_text(json.dumps({"contexts": {c["id"]: {"initial": c["reference_probability"]} for c in two["contexts"]}}) + "\n")
    (dst / "calibration.json").write_text(json.dumps({"contexts": {c["id"]: {"initial": c["reference_probability"]} for c in cal["contexts"]}}) + "\n")
    (dst / "sequence.json").write_text(json.dumps({"contexts": {
        c["id"]: {"initial": c["reference_initial"], "transitions": c["reference_transitions"]} for c in seq["contexts"]}}) + "\n")
    print(f"wrote reference-policy artifacts to {dst}")


if __name__ == "__main__":
    main()
