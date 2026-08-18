import argparse
import os
import subprocess
import sys
import pandas as pd


def run(cmd):
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--model-dir", default=os.path.join("models", "eng_to_prakrit_v4-final"))
    parser.add_argument("--output", default=os.path.join("inferences", "eng_to_prakrit_v4.txt"))
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    os.makedirs("Prakrit_MT/", exist_ok=True)

    # hf_path = "hf://datasets/VIITPune/Deshika-Maharashtri_Prakrit_to_English_Parallel_Corpus/prakrit_translation.csv"
    # df = pd.read_csv(hf_path)

    input_file = os.path.join("data", f"100_test_eng.txt")
    ref_file = os.path.join("data", f"100_test_pra.txt")

    # df["english"].head(args.sample_count).to_csv(input_file, index=False, header=False)
    # df["prakrit"].head(args.sample_count).to_csv(ref_file, index=False, header=False)

    cmd = [
        sys.executable,
        os.path.join("finetuning_and_inferencing_using_indictrans2_models", "inferencing_using_saved_model.py"),
        "--input",
        input_file,
        "--model",
        args.model_dir,
        "--output",
        args.output,
    ]
    run(cmd)

    print(f"Input: {input_file}")
    print(f"Reference: {ref_file}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
