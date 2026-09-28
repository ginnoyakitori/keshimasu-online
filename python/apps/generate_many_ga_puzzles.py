# python/apps/generate_many_ga_puzzles.py

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "ga_reverse"


def parse_targets(value):
    out = []

    for part in str(value).split(","):
        part = part.strip()

        if part:
            out.append(int(part))

    return out


def run_one(
    index,
    word_mode,
    target_f,
    generations,
    population,
    steps,
    mutation,
    workers,
    use_policy_trap,
    policy_trap_weight,
    policy_trap_top_k,
    output_dir,
    prefix,
):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    output_name = (
        f"{prefix}-{word_mode}-f{target_f}-"
        f"{index:06d}-{timestamp}.json"
    )

    cmd = [
        sys.executable,
        "-m",
        "keshimasu_py.ga_reverse",
        "--word-mode",
        str(word_mode),
        "--target-f",
        str(target_f),
        "--generations",
        str(generations),
        "--population",
        str(population),
        "--steps",
        str(steps),
        "--mutation",
        str(mutation),
        "--workers",
        str(workers),
        "--output-dir",
        str(output_dir),
        "--output-name",
        output_name,
    ]

    if use_policy_trap:
        cmd.extend(
            [
                "--use-policy-trap",
                "--policy-trap-weight",
                str(policy_trap_weight),
                "--policy-trap-top-k",
                str(policy_trap_top_k),
            ]
        )

    started = time.perf_counter()

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    elapsed = time.perf_counter() - started

    out_path = output_dir / output_name
    saved = out_path.exists()

    return {
        "index": int(index),
        "wordMode": str(word_mode),
        "targetF": int(target_f),
        "outputName": output_name,
        "outputPath": str(out_path),
        "saved": bool(saved),
        "returnCode": int(result.returncode),
        "elapsedSec": float(elapsed),
        "stdoutTail": "\n".join(result.stdout.splitlines()[-30:]),
        "stderrTail": "\n".join(result.stderr.splitlines()[-30:]),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate many GA reverse puzzles."
    )

    parser.add_argument("--count", type=int, default=20)

    parser.add_argument(
        "--word-mode",
        choices=["country", "capital", "pokemon", "custom"],
        default="country",
    )

    parser.add_argument(
        "--target-f",
        default="2",
        help="F count list, e.g. 2 or 2,3",
    )

    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--mutation", type=float, default=0.18)

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Workers passed to each ga_reverse run. 0 means all CPU cores.",
    )

    parser.add_argument(
        "--parallel-runs",
        type=int,
        default=1,
        help="Number of ga_reverse processes to run in parallel.",
    )

    parser.add_argument("--use-policy-trap", action="store_true")
    parser.add_argument("--policy-trap-weight", type=float, default=0.3)
    parser.add_argument("--policy-trap-top-k", type=int, default=10)

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--prefix",
        default="bulk",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_fs = parse_targets(args.target_f)

    tasks = []
    counter = 0

    for _ in range(int(args.count)):
        for target_f in target_fs:
            counter += 1

            tasks.append(
                {
                    "index": counter,
                    "word_mode": args.word_mode,
                    "target_f": int(target_f),
                    "generations": int(args.generations),
                    "population": int(args.population),
                    "steps": int(args.steps),
                    "mutation": float(args.mutation),
                    "workers": int(args.workers),
                    "use_policy_trap": bool(args.use_policy_trap),
                    "policy_trap_weight": float(args.policy_trap_weight),
                    "policy_trap_top_k": int(args.policy_trap_top_k),
                    "output_dir": output_dir,
                    "prefix": str(args.prefix),
                }
            )

    print("=== GENERATE MANY GA PUZZLES ===")
    print("Root          :", ROOT)
    print("Output dir    :", output_dir)
    print("Word mode     :", args.word_mode)
    print("Target F      :", target_fs)
    print("Count per F   :", args.count)
    print("Total runs    :", len(tasks))
    print("Generations   :", args.generations)
    print("Population    :", args.population)
    print("Workers/run   :", args.workers)
    print("Parallel runs :", args.parallel_runs)
    print("Policy trap   :", bool(args.use_policy_trap))

    results = []
    started_all = time.perf_counter()

    if int(args.parallel_runs) <= 1:
        for task in tasks:
            result = run_one(**task)
            results.append(result)

            status = "SAVED" if result["saved"] else "FAIL"

            print(
                f"[{len(results)}/{len(tasks)}] {status} "
                f"f{result['targetF']} {result['outputName']} "
                f"time={result['elapsedSec']:.1f}s"
            )
    else:
        with ThreadPoolExecutor(max_workers=int(args.parallel_runs)) as executor:
            futures = [
                executor.submit(run_one, **task)
                for task in tasks
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                status = "SAVED" if result["saved"] else "FAIL"

                print(
                    f"[{len(results)}/{len(tasks)}] {status} "
                    f"f{result['targetF']} {result['outputName']} "
                    f"time={result['elapsedSec']:.1f}s"
                )

    elapsed_all = time.perf_counter() - started_all

    saved_count = sum(1 for r in results if r["saved"])
    failed_count = len(results) - saved_count

    manifest = {
        "format": "keshimasu-ga-bulk-generation-v1",
        "createdAt": datetime.now().isoformat(),
        "wordMode": args.word_mode,
        "targetF": target_fs,
        "countPerF": int(args.count),
        "totalRuns": int(len(tasks)),
        "saved": int(saved_count),
        "failed": int(failed_count),
        "elapsedSec": float(elapsed_all),
        "settings": {
            "generations": int(args.generations),
            "population": int(args.population),
            "steps": int(args.steps),
            "mutation": float(args.mutation),
            "workers": int(args.workers),
            "parallelRuns": int(args.parallel_runs),
            "usePolicyTrap": bool(args.use_policy_trap),
            "policyTrapWeight": float(args.policy_trap_weight),
            "policyTrapTopK": int(args.policy_trap_top_k),
        },
        "results": results,
    }

    manifest_path = output_dir / f"{args.prefix}-{args.word_mode}-manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=== SUMMARY ===")
    print("Saved   :", saved_count)
    print("Failed  :", failed_count)
    print("Elapsed :", f"{elapsed_all:.1f}s")
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()