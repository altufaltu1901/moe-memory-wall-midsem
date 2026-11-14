"""Run the matched LLMSimulator cases into a new directory."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess

from check_coupled import CASES, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir()
    for name in CASES:
        case = output / name
        case.mkdir()
        data = executable.parent.parent / "data" / name
        data.mkdir(parents=True, exist_ok=True)
        if list(data.glob("*decode.csv")):
            raise ValueError(f"Old metrics in {data}; use a fresh build directory")
        config = case / "config.yaml"
        shutil.copyfile(ROOT / name / config.name, config)
        env = dict(os.environ, MOE_TRACE_PATH=str(case / "routes.csv"))
        with (case / "stdout.log").open("wb") as stdout, (case / "stderr.log").open("wb") as stderr:
            process = subprocess.run([str(executable), str(config)], cwd=executable.parent,
                                     env=env, stdout=stdout, stderr=stderr, timeout=1800)
        (case / "exit_code.txt").write_text(f"{process.returncode}\n")
        process.check_returncode()
        metrics, = data.glob("*decode.csv")
        shutil.copyfile(metrics, case / "metrics.csv")
        print(name, flush=True)


if __name__ == "__main__":
    main()
