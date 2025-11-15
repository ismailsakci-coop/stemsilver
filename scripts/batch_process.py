from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import List

from pipeline import MusicRemovalPipeline, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch process multiple wav files.")
    parser.add_argument("--config", default="config/best_pipeline.yaml", help="Base config.")
    parser.add_argument("--in-dir", required=True, help="Directory with input wav files.")
    parser.add_argument("--out-dir", default="artifacts/cleaned", help="Directory for cleaned wav files.")
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern for input files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = sorted(in_dir.glob(args.pattern))
    for path in files:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["paths"]["input_wav"] = str(path)
        rel_path = path.relative_to(in_dir)
        alias = "__".join(rel_path.with_suffix("").parts)
        final_parent = out_dir / rel_path.parent
        final_parent.mkdir(parents=True, exist_ok=True)
        final_wav = final_parent / f"{rel_path.stem}__speech_only.wav"
        run_cfg["paths"]["final_wav"] = str(final_wav)
        run_cfg["paths"]["alias"] = alias
        run_cfg["paths"]["artifacts_dir"] = str(out_dir.parent)
        pipeline = MusicRemovalPipeline(run_cfg)
        pipeline.run()


if __name__ == "__main__":
    main()
