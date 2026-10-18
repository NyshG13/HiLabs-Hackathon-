"""python -m termmap_plus predict --input eval.csv --output preds.csv [--memory train,dev,test] [--no-llm]"""
import argparse

from termmap_plus.predict import main as predict_main


def main():
    ap = argparse.ArgumentParser(prog="termmap_plus")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("predict")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--memory", default="train", help="labelled splits used as gold memory (e.g. train,dev,test)")
    p.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    predict_main(a)


if __name__ == "__main__":
    main()
