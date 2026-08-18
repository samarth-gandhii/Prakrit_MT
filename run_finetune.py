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
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--model-dir", default=os.path.join("models", "eng_to_prakrit_v3"))
    parser.add_argument("--train", default=os.path.join("data", "final_train_clean.tsv"))
    parser.add_argument(
        "--base-model",
        default=os.environ.get("BASE_MODEL_PATH", "ai4bharat/indictrans2-en-indic-1B"),
        help="HF model id or local checkpoint path for the base IndicTrans2 model",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from local files (offline-safe)",
    )
    parser.add_argument("--test", default="")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    cmd = [
        sys.executable,
        os.path.join("finetuning_and_inferencing_using_indictrans2_models", "finetuning_using_indictrans2_model.py"),
        "--train",
        args.train,
        "--base_model",
        args.base_model,
        "--model",
        args.model_dir,
        "--epoch",
        str(args.epoch),
        "--test_size",
        str(args.test_size),
        "--seed",
        str(args.seed),
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.test:
        cmd.extend(["--test", args.test])
    run(cmd)


if __name__ == "__main__":
    main()
