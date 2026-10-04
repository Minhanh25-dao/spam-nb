"""Phân loại thư / tin nhắn bằng mô hình đã huấn luyện.

Ví dụ:
    python predict.py --model models/enron_mnb.pkl --text "Congratulations! You won $1000..."
    python predict.py --model models/enron_mnb.pkl --file mail1.txt mail2.txt --explain
    python predict.py --model models/sms_mnb.pkl --lines messages.txt
    python predict.py --model models/sms_mnb.pkl            # chế độ tương tác
"""
import argparse
import sys
from pathlib import Path

from spamfilter import SpamFilter

WIDTH = 78


def shorten(text: str, n: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def show_result(clf: SpamFilter, text: str, threshold: float, explain: bool, top_k: int) -> None:
    res = clf.predict_one(text, threshold)
    print("-" * WIDTH)
    print(f"Nội dung : {shorten(text)}")
    print(f"Kết luận : {res['name']:<5}  P(spam)={res['spam_prob']:.4f}  "
          f"P(ham)={res['ham_prob']:.4f}  [ngưỡng {threshold:.2f}, {res['n_tokens']} token]")
    if explain:
        ex = clf.explain(text, top_k=top_k)
        print(f"Log-odds : {ex['log_odds']:+.3f} = prior {ex['prior_log_odds']:+.3f}"
              f" + tổng đóng góp của các từ {ex['log_odds'] - ex['prior_log_odds'] - ex['bias']:+.3f}"
              + (f" + hằng số {ex['bias']:+.3f}" if ex['bias'] else ""))
        if ex["top_spam"]:
            print("  Từ đẩy về SPAM : " + ", ".join(f"{t}({c:+.2f})" for t, _, c in ex["top_spam"]))
        if ex["top_ham"]:
            print("  Từ đẩy về HAM  : " + ", ".join(f"{t}({c:+.2f})" for t, _, c in ex["top_ham"]))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="file mô hình .pkl do train.py tạo ra")
    ap.add_argument("--text", action="append", default=[], help="nội dung cần phân loại (lặp được)")
    ap.add_argument("--file", nargs="*", default=[], help="mỗi file là MỘT thư")
    ap.add_argument("--lines", help="file văn bản, mỗi dòng là một tin nhắn")
    ap.add_argument("--threshold", type=float, default=0.5, help="ngưỡng P(spam) để gán SPAM")
    ap.add_argument("--explain", action="store_true", help="hiển thị các từ ảnh hưởng nhất")
    ap.add_argument("--top-k", type=int, default=6)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    clf = SpamFilter.load(args.model)
    print("=" * WIDTH)
    print(f"Mô hình  : {args.model}  ({clf.model_name}, alpha={clf.alpha}, "
          f"|V|={clf.vocabulary_size})")

    inputs = list(args.text)
    for f in args.file:
        inputs.append(Path(f).read_text(encoding="utf-8", errors="replace"))
    if args.lines:
        inputs += [ln for ln in Path(args.lines).read_text(encoding="utf-8").splitlines() if ln.strip()]

    if inputs:
        for t in inputs:
            show_result(clf, t, args.threshold, args.explain, args.top_k)
        return

    print("Chế độ tương tác - nhập nội dung rồi Enter (dòng trống hoặc Ctrl+D để thoát).")
    for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("> "), None):
        if not line.strip():
            break
        show_result(clf, line.strip(), args.threshold, args.explain, args.top_k)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print()
