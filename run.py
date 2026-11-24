"""One entry point for the whole project.

    python run.py status            Show what is built and what to run next
    python run.py original          Original 50k benchmark: data -> features -> split -> LR + RF -> comparison
    python run.py m4                M4 extension: download -> features -> grouping -> split -> LR, transfer test, RF -> summary
    python run.py figures           Regenerate the exploratory figures for the original benchmark
    python run.py predict "text"    Classify a text snippet as Human or AI
    python run.py test              Run the unit tests

Pipeline steps whose output already exists are skipped, so re-running is cheap
and an interrupted run picks up where it stopped. Use --force to rebuild
everything, or --dry-run to see the plan without running it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
M4 = DATA / "m4"
MODELS = ROOT / "models"


@dataclass
class Step:
    title: str
    script: str
    args: tuple[str, ...] = ()
    # Step is skipped when this file exists. None means "always run" (cheap checks/summaries).
    output: Path | None = None


ORIGINAL_DATA = [
    Step("Download the Kaggle benchmark and prepare a copy", "prepare_data.py",
         output=DATA / "AI_vs_Human_Text_Classification_Benchmark_processed.csv"),
    Step("Derive text features from Raw_Text (several minutes)", "build_candidate_features.py",
         output=DATA / "text_derived_features_candidate.csv"),
    Step("Write the seven-feature model matrix", "prepare_final_data.py",
         output=DATA / "human_ai_text_processed.csv"),
    Step("Group near-duplicates and build the group-aware split (~2 min)", "build_splits.py",
         output=DATA / "human_ai_text_splits.csv"),
]

ORIGINAL_MODELS = [
    Step("Train and evaluate logistic regression (~10 s)", "train_logistic_regression.py",
         output=MODELS / "logistic_regression.joblib"),
    Step("Train and evaluate random forest (~3 min)", "train_random_forest.py",
         output=MODELS / "random_forest.joblib"),
    Step("Compare the two models on the same test rows", "compare_models.py"),
]

M4_STEPS = [
    Step("Download M4 (Kaggle, pinned to version 1)", "download_m4.py",
         output=ROOT / "data" / "raw" / "m4" / "v1" / "download_manifest.json"),
    Step("Audit M4 schema and coverage", "audit_m4.py",
         output=M4 / "feature_sample_exploratory.csv"),
    Step("Deduplicate M4 and extract the same seven features", "prepare_m4.py",
         output=M4 / "m4_processed.csv"),
    Step("Validate the M4 processed CSV", "validate_m4.py"),
    Step("Plot original vs. M4 feature distributions", "plot_m4_comparison.py"),
    Step("Exhaustive near-duplicate similarity (~2-35 min, depending on the machine)", "audit_m4_similarity.py",
         output=M4 / "similarity_edges_exploratory.npz"),
    Step("Build the group-aware M4 split (approved threshold 0.85)", "build_m4_splits.py",
         ("--threshold", "0.85"), output=M4 / "m4_splits.csv"),
    Step("Tune and evaluate logistic regression on M4 (~1-8 min)", "train_m4.py",
         ("--model", "logistic_regression", "--jobs", "2"), output=MODELS / "m4" / "logistic_regression.joblib"),
    Step("Apply the original models to M4 without retraining", "evaluate_m4_transfer.py",
         ("--reconstruct-original",), output=M4 / "transfer_logistic_regression_test_predictions.csv"),
]

M4_RF = Step("Tune and evaluate random forest on M4 (~15-30 min)", "train_m4.py",
             ("--model", "random_forest", "--jobs", "4"), output=MODELS / "m4" / "random_forest.joblib")

M4_SUMMARY = Step("Summarize M4 results with group-bootstrap intervals", "summarize_m4_results.py",
                  ("--bootstrap", "200"))

FIGURES = [
    Step("Print a summary of the raw benchmark", "explore_data.py"),
    Step("Write exploratory figures to reports/figures/", "visualize_data.py"),
    Step("Write candidate-feature review figures", "visualize_candidate_features.py"),
]


def run_pipeline(name: str, steps: list[Step], force: bool, dry_run: bool) -> None:
    print(f"\n=== {name}: {len(steps)} steps ===")
    rebuild = force  # once a step is rebuilt, every later step is rebuilt too
    for number, step in enumerate(steps, 1):
        command = [sys.executable, f"src/{step.script}", *step.args]
        label = f"[{number}/{len(steps)}] {step.title}"
        if step.output is not None and step.output.exists() and not rebuild:
            print(f"{label}\n      skip: {step.output.relative_to(ROOT)} already exists")
            continue
        print(f"{label}\n      run:  {' '.join(['python', *command[1:]])}")
        if dry_run:
            rebuild = rebuild or step.output is not None
            continue
        start = time.time()
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"\nStep failed: {step.title}. Fix the error above, then re-run; finished steps are skipped.")
        print(f"      done in {time.time() - start:.0f} s")
        rebuild = rebuild or step.output is not None


def show_status() -> None:
    sections = [
        ("Original benchmark  (python run.py original)", ORIGINAL_DATA + ORIGINAL_MODELS),
        ("M4 extension        (python run.py m4)", M4_STEPS + [M4_RF]),
    ]
    for title, steps in sections:
        print(f"\n{title}")
        for step in steps:
            if step.output is None:
                continue
            mark = "x" if step.output.exists() else " "
            print(f"  [{mark}] {step.title}")
    print("\nproject_notebook.ipynb and the figures in reports/ are committed, so you can read them without running anything.")
    if not (MODELS / "logistic_regression.joblib").exists():
        print("Next: python run.py original   (then: python run.py predict \"some text\")")
    else:
        print('Ready: python run.py predict "Paste a passage of text here."')


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # keep step headers in order with the scripts' own output
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "original", "m4", "figures", "predict", "test"])
    parser.add_argument("--force", action="store_true", help="Rebuild every step even if its output exists")
    parser.add_argument("--dry-run", action="store_true", help="Show the steps without running them")
    args, rest = parser.parse_known_args()

    if args.command == "predict":
        model = "random_forest" if "random_forest" in rest else "logistic_regression"
        if not (MODELS / f"{model}.joblib").exists():
            sys.exit(f"No trained {model} model yet. Run this first (about 5-15 minutes):\n  python run.py original")
        sys.exit(subprocess.run([sys.executable, "src/predict_text.py", *rest], cwd=ROOT).returncode)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")

    if args.command == "status":
        show_status()
    elif args.command == "test":
        sys.exit(subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT).returncode)
    elif args.command == "original":
        run_pipeline("Original benchmark", ORIGINAL_DATA + ORIGINAL_MODELS, args.force, args.dry_run)
    elif args.command == "figures":
        run_pipeline("Exploratory figures", ORIGINAL_DATA[:2] + FIGURES, args.force, args.dry_run)
    elif args.command == "m4":
        # The transfer step needs the original features/split; the similarity audit reads the original raw CSV.
        steps = ORIGINAL_DATA + M4_STEPS + [M4_RF, M4_SUMMARY]
        run_pipeline("M4 extension", steps, args.force, args.dry_run)


if __name__ == "__main__":
    main()
