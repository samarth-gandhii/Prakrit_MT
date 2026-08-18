import argparse
import os
import sys
import sacrebleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp", default=os.path.join("inferences", "eng_to_prakrit.txt"))
    parser.add_argument("--ref", default=os.path.join("inferences", "reference_prakrit_100.txt"))
    args = parser.parse_args()

    with open(args.ref, encoding="utf-8") as ref_file:
        refs = [ref_file.read().splitlines()]
    with open(args.hyp, encoding="utf-8") as hyp_file:
        hyps = hyp_file.read().splitlines()

    print(sacrebleu.corpus_bleu(hyps, refs).format())


if __name__ == "__main__":
    main()
