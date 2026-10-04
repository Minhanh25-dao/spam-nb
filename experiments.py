"""Chạy toàn bộ thực nghiệm của báo cáo và ghi kết quả vào results/results.json.

    python experiments.py                 # chạy mọi giai đoạn
    python experiments.py --stages main cv    # chỉ chạy một số giai đoạn

Các giai đoạn: stats, main, cv, alpha, ablation, features, models, threshold,
               learning, words, errors, cross, toy, protocol
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn import metrics as skm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB as SkMultinomialNB
from sklearn.svm import LinearSVC

from spamfilter import BagOfWords, BernoulliNB, MultinomialNB, PreprocessConfig, TextPreprocessor
from spamfilter.data import deduplicate, load_dataset, stratified_kfold, stratified_split
from spamfilter.metrics import classification_metrics, roc_auc, weighted_accuracy

ROOT = Path(__file__).parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 42
DATASETS = {"enron": ROOT / "data/enron_spam_data.csv", "sms": ROOT / "data/sms.tsv"}
DEFAULT_CFG = PreprocessConfig()          # chốt trước khi đánh giá trên tập kiểm tra
DEFAULT_MIN_DF = 2

PRE_CONFIGS = {   # thang loại bỏ thành phần tiền xử lý (mục 2.9.6)
    "A0": ("Chỉ chữ thường + token chữ cái", PreprocessConfig(strip_html=False, normalize_entities=False)),
    "A1": ("A0 + bỏ thẻ HTML", PreprocessConfig(strip_html=True, normalize_entities=False)),
    "A2": ("A1 + chuẩn hoá URL/email/tiền/số (MẶC ĐỊNH)", PreprocessConfig()),
    "A3": ("A2 + đánh dấu CHỮ HOA và dấu '!'", PreprocessConfig(mark_caps=True, mark_exclaim=True)),
    "A4": ("A2 + loại bỏ stopword", PreprocessConfig(remove_stopwords=True)),
    "A5": ("A2 nhưng KHÔNG đổi chữ thường", PreprocessConfig(lowercase=False)),
}


# ------------------------------------------------------------------ tiện ích
def load_json() -> dict:
    p = RES / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_json(obj: dict) -> None:
    (RES / "results.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def load_data(name: str):
    df, info = deduplicate(load_dataset(name, DATASETS[name]))
    return df["text"].to_numpy(), df["label"].to_numpy(), info


def tokens_for(name: str, texts, cfg_key: str = "A2"):
    """Tách token có lưu đệm (cache) ra đĩa để chạy lại nhanh."""
    cache = RES / f"tok_{name}_{cfg_key}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    pre = TextPreprocessor(PRE_CONFIGS[cfg_key][1])
    toks = [pre(t) for t in texts]
    cache.write_bytes(pickle.dumps(toks))
    return toks


def split(name):
    texts, y, info = load_data(name)
    tr, te = stratified_split(y, 0.2, SEED)
    return texts, y, tr, te, info


def vectorize(tok_tr, tok_te, **bow_kw):
    bow = BagOfWords(**bow_kw)
    Xtr = bow.fit_transform(tok_tr)
    return bow, Xtr, bow.transform(tok_te)


def evaluate(nb, Xte, yte, threshold=None):
    pred = nb.predict(Xte, threshold=threshold)
    m = classification_metrics(yte, pred)
    m["roc_auc"] = roc_auc(yte, nb.decision_function(Xte))
    return m


def pick(lst, idx):
    return [lst[i] for i in idx]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------ giai đoạn
def stage_stats(R):
    out = {}
    for name in DATASETS:
        raw = load_dataset(name, DATASETS[name])
        texts, y, tr, te, info = split(name)
        toks = tokens_for(name, texts)
        n_tok = np.array([len(t) for t in toks])
        n_chr = np.array([len(t) for t in texts])
        vocab1 = len(BagOfWords(min_df=1).fit(pick(toks, tr)).vocabulary_)
        vocab2 = len(BagOfWords(min_df=2).fit(pick(toks, tr)).vocabulary_)
        out[name] = {
            **info, "raw_ham": int((raw.label == 0).sum()), "raw_spam": int((raw.label == 1).sum()),
            "ham": int((y == 0).sum()), "spam": int((y == 1).sum()),
            "spam_ratio": float(y.mean()),
            "n_train": len(tr), "n_test": len(te),
            "train_spam": int(y[tr].sum()), "test_spam": int(y[te].sum()),
            "chars_mean": float(n_chr.mean()), "chars_median": float(np.median(n_chr)),
            "tok_mean": float(n_tok.mean()), "tok_median": float(np.median(n_tok)),
            "tok_max": int(n_tok.max()), "tok_ham_mean": float(n_tok[y == 0].mean()),
            "tok_spam_mean": float(n_tok[y == 1].mean()),
            "tok_total": int(n_tok.sum()), "empty_docs": int((n_tok == 0).sum()),
            "vocab_mindf1": vocab1, "vocab_mindf2": vocab2,
            "tok_lengths_ham": np.clip(n_tok[y == 0], 0, 5000).tolist() if name == "sms" else
            np.clip(n_tok[y == 0], 0, 5000)[::5].tolist(),
            "tok_lengths_spam": np.clip(n_tok[y == 1], 0, 5000).tolist() if name == "sms" else
            np.clip(n_tok[y == 1], 0, 5000)[::5].tolist(),
        }
        log(f"stats {name}: {out[name]['after']} mẫu, |V|={vocab2}")
    R["stats"] = out


def stage_main(R):
    out = {}
    for name in DATASETS:
        texts, y, tr, te, _ = split(name)
        toks = tokens_for(name, texts)
        t0 = time.perf_counter()
        bow, Xtr, Xte = vectorize(pick(toks, tr), pick(toks, te), min_df=DEFAULT_MIN_DF)
        t_vec = time.perf_counter() - t0
        t0 = time.perf_counter()
        nb = MultinomialNB(alpha=1.0).fit(Xtr, y[tr])
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred_scores = nb.decision_function(Xte)
        t_pred = time.perf_counter() - t0
        m = evaluate(nb, Xte, y[te])
        # đối chiếu với scikit-learn
        sk = SkMultinomialNB(alpha=1.0).fit(Xtr, y[tr])
        lp_mine, lp_sk = nb.predict_log_proba(Xte), sk.predict_log_proba(Xte)
        agree = float(np.mean(nb.predict(Xte) == sk.predict(Xte)))
        out[name] = {
            "metrics": m, "vocab": len(bow.vocabulary_),
            "time_vectorize_fit_s": t_vec, "time_nb_fit_s": t_fit, "time_nb_predict_s": t_pred,
            "n_test": len(te), "docs_per_s": len(te) / (t_pred + 1e-9),
            "sklearn": {"max_abs_diff_logproba": float(np.max(np.abs(lp_mine - lp_sk))),
                        "prediction_agreement": agree,
                        "accuracy": float(np.mean(sk.predict(Xte) == y[te]))},
            "prior_spam": float(np.exp(nb.class_log_prior_[1])),
            "n_params": int(nb.feature_log_prob_.size + nb.class_log_prior_.size),
        }
        log(f"main {name}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")
    R["main"] = out


def stage_cv(R):
    out = {}
    for name in DATASETS:
        texts, y, _, _, _ = split(name)
        toks = tokens_for(name, texts)
        rows = []
        for i, (a, b) in enumerate(stratified_kfold(y, 5, SEED)):
            _, Xa, Xb = vectorize(pick(toks, a), pick(toks, b), min_df=DEFAULT_MIN_DF)
            nb = MultinomialNB(1.0).fit(Xa, y[a])
            rows.append(evaluate(nb, Xb, y[b]))
            log(f"cv {name} fold {i + 1}: f1={rows[-1]['f1']:.4f}")
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        out[name] = {"folds": rows,
                     "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
                     "std": {k: float(np.std([r[k] for r in rows], ddof=1)) for k in keys}}
    R["cv"] = out


ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def stage_alpha(R):
    """Chọn alpha bằng CV 5 lần TRÊN TẬP HUẤN LUYỆN; báo cáo thêm đường cong trên tập kiểm tra."""
    out = {}
    for name in DATASETS:
        texts, y, tr, te, _ = split(name)
        toks = tokens_for(name, texts)
        ttr = pick(toks, tr)
        cv = {a: [] for a in ALPHAS}
        for a_idx, b_idx in stratified_kfold(y[tr], 5, SEED):
            _, Xa, Xb = vectorize(pick(ttr, a_idx), pick(ttr, b_idx), min_df=DEFAULT_MIN_DF)
            for a in ALPHAS:
                nb = MultinomialNB(a).fit(Xa, y[tr][a_idx])
                cv[a].append(classification_metrics(y[tr][b_idx], nb.predict(Xb)))
        _, Xtr, Xte = vectorize(ttr, pick(toks, te), min_df=DEFAULT_MIN_DF)
        rows = []
        for a in ALPHAS:
            mt = evaluate(MultinomialNB(a).fit(Xtr, y[tr]), Xte, y[te])
            rows.append({"alpha": a,
                         "cv_f1": float(np.mean([r["f1"] for r in cv[a]])),
                         "cv_acc": float(np.mean([r["accuracy"] for r in cv[a]])),
                         "test": mt})
        best = max(rows, key=lambda r: r["cv_f1"])
        out[name] = {"rows": rows, "best_alpha": best["alpha"], "best_cv_f1": best["cv_f1"],
                     "test_at_best": best["test"]}
        log(f"alpha {name}: best={best['alpha']} cvF1={best['cv_f1']:.4f} testF1={best['test']['f1']:.4f}")
    R["alpha"] = out


def stage_ablation(R):
    out = {}
    for name in DATASETS:
        texts, y, tr, te, _ = split(name)
        rows = []
        for key, (desc, _cfg) in PRE_CONFIGS.items():
            toks = tokens_for(name, texts, key)
            bow, Xtr, Xte = vectorize(pick(toks, tr), pick(toks, te), min_df=DEFAULT_MIN_DF)
            m = evaluate(MultinomialNB(1.0).fit(Xtr, y[tr]), Xte, y[te])
            rows.append({"key": key, "desc": desc, "vocab": len(bow.vocabulary_), **m})
            log(f"ablation {name} {key}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} |V|={len(bow.vocabulary_)}")
        out[name] = rows
    R["ablation"] = out


def stage_features(R):
    out = {}
    for name in DATASETS:
        texts, y, tr, te, _ = split(name)
        toks = tokens_for(name, texts)
        ttr, tte = pick(toks, tr), pick(toks, te)
        rows = []
        for min_df in (1, 2, 5, 10):
            bow, Xtr, Xte = vectorize(ttr, tte, min_df=min_df)
            m = evaluate(MultinomialNB(1.0).fit(Xtr, y[tr]), Xte, y[te])
            rows.append({"desc": f"Đếm tần suất, min_df={min_df}", "vocab": len(bow.vocabulary_), **m})
        bow, Xtr, Xte = vectorize(ttr, tte, min_df=DEFAULT_MIN_DF, binary=True)
        m = evaluate(MultinomialNB(1.0).fit(Xtr, y[tr]), Xte, y[te])
        rows.append({"desc": "Nhị phân (có/không), min_df=2", "vocab": len(bow.vocabulary_), **m})
        log(f"features {name}: min_df & binary xong")
        if name == "sms":     # bigram: đủ nhẹ để chạy trên SMS
            bow, Xtr, Xte = vectorize(ttr, tte, min_df=2, ngram_range=(1, 2))
            m = evaluate(MultinomialNB(1.0).fit(Xtr, y[tr]), Xte, y[te])
            rows.append({"desc": "Unigram + bigram, min_df=2", "vocab": len(bow.vocabulary_), **m})
        else:                 # Enron: cắt 300 token đầu mỗi thư để bigram vừa bộ nhớ
            t_tr = [t[:300] for t in ttr]
            t_te = [t[:300] for t in tte]
            bow, Xtr, Xte = vectorize(t_tr, t_te, min_df=5)
            m0 = evaluate(MultinomialNB(1.0).fit(Xtr, y[tr]), Xte, y[te])
            rows.append({"desc": "Unigram, cắt 300 token đầu, min_df=5", "vocab": len(bow.vocabulary_), **m0})
            bow, Xtr, Xte = vectorize(t_tr, t_te, min_df=5, ngram_range=(1, 2))
            m = evaluate(MultinomialNB(1.0).fit(Xtr, y[tr]), Xte, y[te])
            rows.append({"desc": "Unigram + bigram, cắt 300 token đầu, min_df=5", "vocab": len(bow.vocabulary_), **m})
        out[name] = rows
        log(f"features {name}: xong")
    R["features"] = out


def stage_models(R):
    out = {}
    for name in DATASETS:
        texts, y, tr, te, _ = split(name)
        toks = tokens_for(name, texts)
        bow, Xtr, Xte = vectorize(pick(toks, tr), pick(toks, te), min_df=DEFAULT_MIN_DF)
        ytr, yte = y[tr], y[te]
        rows = []

        def add(label, fit_fn, pred_fn, score_fn):
            t0 = time.perf_counter()
            model = fit_fn()
            t_fit = time.perf_counter() - t0
            pred = pred_fn(model)
            m = classification_metrics(yte, pred)
            m["roc_auc"] = roc_auc(yte, score_fn(model))
            rows.append({"model": label, "time_fit_s": t_fit, **m})
            log(f"models {name} {label}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} ({t_fit:.1f}s)")

        maj = int(round(ytr.mean()))
        rows.append({"model": "Baseline: luôn đoán lớp đa số", "time_fit_s": 0.0,
                     **classification_metrics(yte, np.full_like(yte, maj)), "roc_auc": 0.5})
        add("Multinomial NB (cài đặt riêng)", lambda: MultinomialNB(1.0).fit(Xtr, ytr),
            lambda m: m.predict(Xte), lambda m: m.decision_function(Xte))
        add("Bernoulli NB (cài đặt riêng)", lambda: BernoulliNB(1.0).fit(Xtr, ytr),
            lambda m: m.predict(Xte), lambda m: m.decision_function(Xte))
        add("Complement NB (scikit-learn)", lambda: ComplementNB(alpha=1.0).fit(Xtr, ytr),
            lambda m: m.predict(Xte), lambda m: m.predict_log_proba(Xte)[:, 1] - m.predict_log_proba(Xte)[:, 0])
        tf = TfidfTransformer().fit(Xtr)
        Ttr, Tte = tf.transform(Xtr), tf.transform(Xte)
        add("Multinomial NB + TF-IDF (scikit-learn)", lambda: SkMultinomialNB(alpha=0.1).fit(Ttr, ytr),
            lambda m: m.predict(Tte), lambda m: m.predict_log_proba(Tte)[:, 1] - m.predict_log_proba(Tte)[:, 0])
        add("Logistic Regression + TF-IDF", lambda: LogisticRegression(C=10, max_iter=1000, solver="liblinear").fit(Ttr, ytr),
            lambda m: m.predict(Tte), lambda m: m.decision_function(Tte))
        add("Linear SVM + TF-IDF", lambda: LinearSVC(C=1.0).fit(Ttr, ytr),
            lambda m: m.predict(Tte), lambda m: m.decision_function(Tte))
        out[name] = rows
    R["models"] = out


THRESHOLDS = [0.5, 0.9, 0.99, 0.999, 0.9999]


def _downsample(a, b, n=300):
    idx = np.unique(np.linspace(0, len(a) - 1, n).astype(int))
    return np.asarray(a)[idx].tolist(), np.asarray(b)[idx].tolist()


def stage_threshold(R):
    out = {}
    for name in DATASETS:
        texts, y, tr, te, _ = split(name)
        toks = tokens_for(name, texts)
        _, Xtr, Xte = vectorize(pick(toks, tr), pick(toks, te), min_df=DEFAULT_MIN_DF)
        nb = MultinomialNB(1.0).fit(Xtr, y[tr])
        yte, score = y[te], nb.decision_function(Xte)
        rows = []
        for t in THRESHOLDS:
            pred = nb.predict(Xte, threshold=t)
            m = classification_metrics(yte, pred)
            rows.append({"threshold": t, **m,
                         "wacc_1": weighted_accuracy(yte, pred, 1), "wacc_9": weighted_accuracy(yte, pred, 9),
                         "wacc_999": weighted_accuracy(yte, pred, 999)})
        fpr, tpr, _ = skm.roc_curve(yte, score)
        pr, rc, _ = skm.precision_recall_curve(yte, score)
        out[name] = {"rows": rows, "roc_auc": float(skm.roc_auc_score(yte, score)),
                     "avg_precision": float(skm.average_precision_score(yte, score)),
                     "roc": dict(zip(("fpr", "tpr"), _downsample(fpr, tpr))),
                     "pr": dict(zip(("recall", "precision"), _downsample(rc, pr))),
                     "score_ham": np.clip(score[yte == 0], -300, 300)[::3].tolist(),
                     "score_spam": np.clip(score[yte == 1], -300, 300)[::3].tolist()}
        log(f"threshold {name}: AUC={out[name]['roc_auc']:.4f} AP={out[name]['avg_precision']:.4f}")
    R["threshold"] = out


FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def stage_learning(R):
    out = {}
    for name in DATASETS:
        texts, y, tr, te, _ = split(name)
        toks = tokens_for(name, texts)
        ttr, tte, ytr, yte = pick(toks, tr), pick(toks, te), y[tr], y[te]
        rows = []
        for f in FRACTIONS:
            accs, f1s, vocs = [], [], []
            for rep in range(3 if f < 1 else 1):
                rng = np.random.default_rng(SEED + rep)
                sub = np.concatenate([
                    rng.choice(np.flatnonzero(ytr == c), max(2, int(round(f * (ytr == c).sum()))), replace=False)
                    for c in (0, 1)])
                bow = BagOfWords(min_df=1 if f < 0.2 else DEFAULT_MIN_DF).fit(pick(ttr, sub))
                nb = MultinomialNB(1.0).fit(bow.transform(pick(ttr, sub)), ytr[sub])
                m = classification_metrics(yte, nb.predict(bow.transform(tte)))
                accs.append(m["accuracy"]); f1s.append(m["f1"]); vocs.append(len(bow.vocabulary_))
            rows.append({"fraction": f, "n_train": int(len(sub)), "accuracy": float(np.mean(accs)),
                         "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s)), "vocab": int(np.mean(vocs))})
            log(f"learning {name} {f}: n={len(sub)} acc={rows[-1]['accuracy']:.4f} f1={rows[-1]['f1']:.4f}")
        out[name] = rows
    R["learning"] = out


def stage_words(R):
    out = {}
    for name, min_total in (("enron", 30), ("sms", 8)):
        texts, y, tr, te, _ = split(name)
        toks = tokens_for(name, texts)
        bow = BagOfWords(min_df=DEFAULT_MIN_DF).fit(pick(toks, tr))
        Xtr = bow.transform(pick(toks, tr))
        nb = MultinomialNB(1.0).fit(Xtr, y[tr])
        names = np.array(bow.feature_names_)
        ratio = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
        df_all = np.asarray((Xtr > 0).sum(axis=0)).ravel()
        ok = np.flatnonzero(df_all >= min_total)
        o = ok[np.argsort(ratio[ok])]
        cnt = nb.feature_count_
        top_spam_freq = np.argsort(-cnt[1])[:15]
        top_ham_freq = np.argsort(-cnt[0])[:15]
        out[name] = {
            "min_total_df": min_total,
            "spam_ratio": [(names[i], float(ratio[i]), int(df_all[i]), float(np.exp(ratio[i]))) for i in o[::-1][:15]],
            "ham_ratio": [(names[i], float(ratio[i]), int(df_all[i]), float(np.exp(ratio[i]))) for i in o[:15]],
            "spam_freq": [(names[i], int(cnt[1][i])) for i in top_spam_freq],
            "ham_freq": [(names[i], int(cnt[0][i])) for i in top_ham_freq],
        }
    R["words"] = out


def stage_errors(R):
    from spamfilter import SpamFilter
    out = {}
    for name in DATASETS:
        texts, y, tr, te, _ = split(name)
        clf = SpamFilter(min_df=DEFAULT_MIN_DF).fit(texts[tr], y[tr])
        p = clf.predict_proba(texts[te])
        yte = y[te]
        fp = np.flatnonzero((yte == 0) & (p >= 0.5))
        fn = np.flatnonzero((yte == 1) & (p < 0.5))
        fp = fp[np.argsort(-p[fp])]
        fn = fn[np.argsort(p[fn])]

        def rows(idx, k=25):
            out_rows = []
            for i in idx[:k]:
                t = " ".join(texts[te][i].split())
                out_rows.append({"p_spam": float(p[i]), "chars": len(t), "text": t[:400]})
            return out_rows
        out[name] = {"n_fp": int(len(fp)), "n_fn": int(len(fn)), "fp": rows(fp), "fn": rows(fn),
                     "fn_p_median": float(np.median(p[fn])) if len(fn) else None,
                     "fp_p_median": float(np.median(p[fp])) if len(fp) else None,
                     "fp_short": int(np.sum([len(texts[te][i]) < 200 for i in fp])),
                     "fn_short": int(np.sum([len(texts[te][i]) < 200 for i in fn]))}
        log(f"errors {name}: FP={len(fp)} FN={len(fn)}")
    R["errors"] = out


def stage_cross(R):
    """Huấn luyện trên miền này, kiểm tra trên miền kia (dịch chuyển miền)."""
    out = {}
    for src, dst in (("enron", "sms"), ("sms", "enron")):
        ts, ys, trs, tes, _ = split(src)
        td, yd, trd, ted, _ = split(dst)
        toks_s, toks_d = tokens_for(src, ts), tokens_for(dst, td)
        bow = BagOfWords(min_df=DEFAULT_MIN_DF).fit(pick(toks_s, trs))
        nb = MultinomialNB(1.0).fit(bow.transform(pick(toks_s, trs)), ys[trs])
        Xd = bow.transform(toks_d)              # toàn bộ miền đích
        m_dst = evaluate(nb, Xd, yd)
        same = evaluate(nb, bow.transform(pick(toks_s, tes)), ys[tes])
        # tỉ lệ token đích nằm trong từ vựng nguồn
        tot = sum(len(t) for t in toks_d)
        cov = sum(1 for t in toks_d for w in t if w in bow.vocabulary_) / max(tot, 1)
        out[f"{src}->{dst}"] = {"in_domain": same, "cross_domain": m_dst, "token_coverage": cov,
                                "predicted_spam_rate": float(nb.predict(Xd).mean()),
                                "true_spam_rate": float(yd.mean())}
        log(f"cross {src}->{dst}: acc={m_dst['accuracy']:.4f} f1={m_dst['f1']:.4f} coverage={cov:.3f}")
    R["cross"] = out


def stage_toy(R):
    spam = ["free prize win now", "win free money", "claim your free prize"]
    ham = ["project meeting tomorrow", "meeting report due tomorrow", "lunch tomorrow now"]
    docs, y = spam + ham, [1, 1, 1, 0, 0, 0]
    bow = BagOfWords()
    X = bow.fit_transform([d.split() for d in docs])
    nb = MultinomialNB(1.0).fit(X, y)
    test = "free prize tomorrow today"
    Xt = bow.transform([test.split()])
    names = bow.feature_names_
    R["toy"] = {"vocab": names, "counts_spam": dict(zip(names, X[:3].sum(axis=0).A1.astype(int).tolist())),
                "counts_ham": dict(zip(names, X[3:].sum(axis=0).A1.astype(int).tolist())),
                "theta_spam": dict(zip(names, np.exp(nb.feature_log_prob_[1]).tolist())),
                "theta_ham": dict(zip(names, np.exp(nb.feature_log_prob_[0]).tolist())),
                "joint": np.exp(nb.predict_joint_log_proba(Xt)[0]).tolist(),
                "log_joint": nb.predict_joint_log_proba(Xt)[0].tolist(),
                "posterior": nb.predict_proba(Xt)[0].tolist()}


# ------------------------------------------------------------------ đánh giá theo nguyên tắc độc lập + chi phí
KEYWORDS = {"free", "win", "winner", "won", "prize", "cash", "claim", "urgent", "congratulations", "viagra", "cialis",
            "lottery", "offer", "guaranteed", "credit", "click", "buy", "cheap", "pills", "casino", "bonus", "loan",
            "mortgage", "earn", "million", "subscribe", "discount", "unsubscribe"}   # danh sách viết tay, KHÔNG rút từ dữ liệu
LAMBDAS = (1, 9, 999)
PROB_GRID = [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
LOGIT_GRID = [float(np.log(p / (1 - p))) for p in PROB_GRID]


def _cost_metrics(y, pred, lam=9):
    m = classification_metrics(y, pred)
    n_ham, n_spam = m["tn"] + m["fp"], m["tp"] + m["fn"]
    cost = lam * m["fp"] + m["fn"]
    m[f"wacc_{lam}"] = 1 - cost / (lam * n_ham + n_spam)
    m[f"tcr_{lam}"] = (n_spam / cost) if cost else None      # TCR = chi phí khi không lọc / chi phí khi lọc (>1: lọc có ích)
    return m


def stage_protocol(R):
    """Quy trình đánh giá chuẩn: mọi lựa chọn (mô hình, ngưỡng, luật từ khoá) chỉ dựa trên tập huấn luyện
    (CV 5 lần); tập kiểm tra chỉ dùng MỘT lần ở cuối. Chi phí: chặn nhầm thư tốt nặng gấp lam lần để lọt thư rác."""
    out = {}
    for name in DATASETS:
        texts, y, tr, te, _ = split(name)
        toks = tokens_for(name, texts)
        ttr, tte, ytr, yte = pick(toks, tr), pick(toks, te), y[tr], y[te]
        folds = list(stratified_kfold(ytr, 5, SEED))
        kw = lambda docs: np.array([len(KEYWORDS & set(d)) for d in docs])
        kw_tr = kw(ttr)
        # luật từ khoá: chọn k (số từ khoá tối thiểu) chỉ trên tập huấn luyện, theo chi phí lam = 9
        k_best = min((1, 2, 3, 4), key=lambda k: (9 * int(((kw_tr >= k) & (ytr == 0)).sum()) + int(((kw_tr < k) & (ytr == 1)).sum())))
        cands = ["Không lọc (cho qua tất cả)", f"Luật từ khoá thủ công (≥ {k_best} từ)", "Multinomial NB", "Bernoulli NB",
                 "Logistic Regression + TF-IDF", "Linear SVM + TF-IDF"]
        per = {c: [] for c in cands}
        oof = np.zeros(len(ytr))
        for i, (a, b) in enumerate(folds):
            _, Xa, Xb = vectorize(pick(ttr, a), pick(ttr, b), min_df=DEFAULT_MIN_DF)
            ya, yb = ytr[a], ytr[b]
            preds = {cands[0]: np.zeros(len(yb), dtype=int), cands[1]: (kw_tr[b] >= k_best).astype(int)}
            mn = MultinomialNB(1.0).fit(Xa, ya)
            oof[b] = mn.decision_function(Xb)
            preds[cands[2]] = (oof[b] >= 0).astype(int)
            preds[cands[3]] = BernoulliNB(1.0).fit(Xa, ya).predict(Xb)
            tf = TfidfTransformer().fit(Xa)
            Ta, Tb = tf.transform(Xa), tf.transform(Xb)
            preds[cands[4]] = LogisticRegression(C=10, max_iter=1000, solver="liblinear").fit(Ta, ya).predict(Tb)
            preds[cands[5]] = LinearSVC(C=1.0).fit(Ta, ya).predict(Tb)
            for c in cands:
                per[c].append(_cost_metrics(yb, preds[c], 9))
            log(f"protocol {name} fold {i + 1}/5")
        keys = ["precision", "recall", "f1", "wacc_9", "tcr_9", "fp", "fn"]
        cv = {}
        for c in cands:
            cv[c] = {}
            for k in keys:
                v = [r[k] for r in per[c] if r[k] is not None]
                cv[c][k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
        # chọn ngưỡng theo chi phí bằng xác suất "ngoài fold" (OOF) của NB trên tập huấn luyện
        thr_choice = {}
        for lam in LAMBDAS:
            costs = [lam * int(((oof >= t) & (ytr == 0)).sum()) + int(((oof < t) & (ytr == 1)).sum()) for t in LOGIT_GRID]
            j = int(np.argmin(costs))
            thr_choice[str(lam)] = {"prob": PROB_GRID[j], "oof_costs": costs}
        # ---- MỘT lần đo trên tập kiểm tra ----
        bow, Xtr, Xte = vectorize(ttr, tte, min_df=DEFAULT_MIN_DF)
        mn = MultinomialNB(1.0).fit(Xtr, ytr)
        lo = mn.decision_function(Xte)
        test = {"Không lọc (cho qua tất cả)": np.zeros(len(yte), dtype=int),
                cands[1]: (kw(tte) >= k_best).astype(int), "Multinomial NB, ngưỡng 0,5": (lo >= 0).astype(int)}
        for lam in LAMBDAS:
            t = thr_choice[str(lam)]["prob"]
            test[f"Multinomial NB, ngưỡng chọn theo λ={lam} ({t})"] = (lo >= float(np.log(t / (1 - t)))).astype(int)
        tf = TfidfTransformer().fit(Xtr)
        Ttr, Tte = tf.transform(Xtr), tf.transform(Xte)
        test["Logistic Regression + TF-IDF"] = LogisticRegression(C=10, max_iter=1000, solver="liblinear").fit(Ttr, ytr).predict(Tte)
        test["Linear SVM + TF-IDF"] = LinearSVC(C=1.0).fit(Ttr, ytr).predict(Tte)
        test_rows = {}
        for label, pr in test.items():
            m = classification_metrics(yte, pr)
            for lam in LAMBDAS:
                m.update({k: v for k, v in _cost_metrics(yte, pr, lam).items() if k.startswith(("wacc_", "tcr_"))})
            test_rows[label] = m
        # ---- kiểm tra rò rỉ gần-trùng: mỗi thư kiểm tra có thư huấn luyện gần giống (cosine TF-IDF ≥ 0,9) không? ----
        from sklearn.preprocessing import normalize
        A, B = normalize(Tte), normalize(Ttr).T.tocsc()
        mx = np.concatenate([np.asarray((A[i:i + 400] @ B).max(axis=1).todense()).ravel() for i in range(0, A.shape[0], 400)])
        dup = mx >= 0.9
        pr = (lo >= 0).astype(int)
        nd = {"threshold": 0.9, "n_test": int(len(yte)), "n_dup": int(dup.sum()), "frac_dup": float(dup.mean())}
        for tag, msk in (("dup", dup), ("indep", ~dup)):
            if msk.sum():
                m = classification_metrics(yte[msk], pr[msk])
                nd[tag] = {"n": int(msk.sum()), "accuracy": m["accuracy"], "f1": m["f1"], "fp": m["fp"], "fn": m["fn"]}
        out[name] = {"k_keyword": int(k_best), "cv": cv, "thr_choice": thr_choice, "test": test_rows, "neardup": nd,
                     "n_train": int(len(ytr)), "n_test": int(len(yte)), "n_ham_test": int((yte == 0).sum()), "n_spam_test": int((yte == 1).sum())}
        log(f"protocol {name}: k={k_best}, thr={ {l: v['prob'] for l, v in thr_choice.items()} }, near-dup={nd['frac_dup']:.3f}")
    R["protocol"] = out


STAGES = {"stats": stage_stats, "main": stage_main, "cv": stage_cv, "alpha": stage_alpha,
          "ablation": stage_ablation, "features": stage_features, "models": stage_models,
          "threshold": stage_threshold, "learning": stage_learning, "words": stage_words,
          "errors": stage_errors, "cross": stage_cross, "toy": stage_toy, "protocol": stage_protocol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    args = ap.parse_args()
    for s in args.stages:
        R = load_json()
        t0 = time.perf_counter()
        log(f"=== giai đoạn: {s}")
        STAGES[s](R)
        save_json(R)
        log(f"=== xong {s} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
