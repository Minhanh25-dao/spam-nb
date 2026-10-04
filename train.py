"""Huấn luyện bộ lọc thư rác Naïve Bayes và lưu mô hình.

Ví dụ:
    python train.py --dataset enron --data data/enron_spam_data.csv --out models/enron_mnb.pkl
    python train.py --dataset sms   --data data/sms.tsv             --out models/sms_mnb.pkl
"""
import argparse
import time
from pathlib import Path


from spamfilter import SpamFilter, PreprocessConfig
from spamfilter.data import deduplicate, load_dataset, stratified_split
from spamfilter.metrics import classification_metrics, roc_auc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["sms", "enron"], required=True)
    ap.add_argument("--data", required=True, help="đường dẫn file dữ liệu")
    ap.add_argument("--out", required=True, help="đường dẫn lưu mô hình (.pkl)")
    ap.add_argument("--model", choices=["multinomial", "bernoulli"], default="multinomial")
    ap.add_argument("--alpha", type=float, default=1.0, help="hệ số làm trơn Laplace")
    ap.add_argument("--min-df", type=int, default=2, help="ngưỡng df tối thiểu của từ")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stopwords", action="store_true", help="loại bỏ stopword")
    ap.add_argument("--no-dedup", action="store_true", help="không loại thư trùng lặp")
    ap.add_argument("--full", action="store_true",
                    help="huấn luyện lại trên TOÀN BỘ dữ liệu sau khi đánh giá")
    return ap


def run_training(dataset: str, data: str, out: str, model: str = "multinomial", alpha: float = 1.0,
                 min_df: int = 2, test_size: float = 0.2, seed: int = 42, stopwords: bool = False,
                 dedup: bool = True, full: bool = False, log=print) -> dict:
    """Huấn luyện, đánh giá trên tập kiểm tra rồi lưu mô hình. Dùng chung cho dòng lệnh (train.py) và giao diện (gui.py).

    `log` là hàm nhận một dòng chữ (mặc định in ra màn hình). Trả về dict chứa mô hình và các chỉ số.
    """
    df = load_dataset(dataset, data)
    info = None
    if dedup:
        df, info = deduplicate(df)
        log(f"[dữ liệu] loại trùng: {info['before']} -> {info['after']} "
            f"(bỏ {info['removed']})")
    y = df["label"].to_numpy()
    log(f"[dữ liệu] {len(df)} mẫu | ham={int((y == 0).sum())} | spam={int((y == 1).sum())}")

    tr, te = stratified_split(y, test_size, seed)
    texts = df["text"].to_numpy()
    cfg = PreprocessConfig(remove_stopwords=stopwords)
    clf = SpamFilter(model=model, alpha=alpha, min_df=min_df, preprocess=cfg)

    t0 = time.perf_counter()
    clf.fit(texts[tr], y[tr])
    t_fit = time.perf_counter() - t0
    log(f"[huấn luyện] {len(tr)} mẫu, |V|={clf.vocabulary_size}, {t_fit:.2f}s")

    t0 = time.perf_counter()
    pred = clf.predict(texts[te])
    scores = clf.decision_function(texts[te])
    t_pred = time.perf_counter() - t0
    m = classification_metrics(y[te], pred)
    m["roc_auc"] = roc_auc(y[te], scores)
    log(f"[kiểm tra ] {len(te)} mẫu, {t_pred:.2f}s")
    log(f"  Accuracy  = {m['accuracy']:.4f}")
    log(f"  Precision = {m['precision']:.4f}")
    log(f"  Recall    = {m['recall']:.4f}")
    log(f"  F1-score  = {m['f1']:.4f}")
    log(f"  ROC-AUC   = {m['roc_auc']:.4f}")
    log(f"  Ma trận nhầm lẫn: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")

    if full:
        clf = SpamFilter(model=model, alpha=alpha, min_df=min_df, preprocess=cfg).fit(texts, y)
        log(f"[huấn luyện lại trên toàn bộ {len(df)} mẫu]")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    clf.save(out)
    size_kb = Path(out).stat().st_size / 1024
    log(f"[đã lưu] {out} ({size_kb:.0f} KB)")
    return {"clf": clf, "metrics": m, "t_fit": t_fit, "t_pred": t_pred, "n_train": len(tr), "n_test": len(te),
            "n_total": len(df), "vocab": clf.vocabulary_size, "size_kb": size_kb, "out": str(out), "dedup_info": info}


def main() -> None:
    args = build_parser().parse_args()
    run_training(args.dataset, args.data, args.out, model=args.model, alpha=args.alpha, min_df=args.min_df,
                 test_size=args.test_size, seed=args.seed, stopwords=args.stopwords,
                 dedup=not args.no_dedup, full=args.full)


if __name__ == "__main__":
    main()
