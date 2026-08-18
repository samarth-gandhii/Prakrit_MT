import argparse
import os
import subprocess
import sys


def run(cmd):
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--output", default=os.path.join("inferences", "eng_to_prakrit.baseline.txt"))
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    input_file = os.path.join("data", f"100_test_eng.txt")

    cmd = [
        sys.executable,
        os.path.join("finetuning_and_inferencing_using_indictrans2_models", "inferencing_using_indictrans2_baseline_models.py"),
        "--input",
        input_file,
        "--output",
        args.output,
    ]
    run(cmd)

    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
