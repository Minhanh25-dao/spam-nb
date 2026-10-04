"""Vẽ toàn bộ hình minh hoạ của báo cáo từ results/results.json và results/demo/*.txt."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch, Circle, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
R = json.loads((ROOT / "results/results.json").read_text(encoding="utf-8"))

HAM, SPAM, GREY, NAVY = "#2C6FBB", "#D9622B", "#6B7280", "#1F3A5F"


def _pick_font(candidates):
    """Chọn phông đầu tiên có trên máy (Linux/Windows/macOS), tránh lỗi findfont."""
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    return next((c for c in candidates if c in have), "DejaVu Serif")


SERIF = _pick_font(["Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"])
_MPL_TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"      # DejaVu luôn đi kèm matplotlib


def _mono(bold=False):
    """Phông đơn cách hỗ trợ đủ dấu tiếng Việt, dùng cho ảnh 'terminal'."""
    cands = [Path("/usr/share/fonts/truetype/liberation/LiberationMono-%s.ttf" % ("Bold" if bold else "Regular")),
             Path("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf"),      # Windows
             _MPL_TTF / ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf")]
    for c in cands:
        if c.exists():
            return str(c)
    raise FileNotFoundError("không tìm thấy phông đơn cách")


plt.rcParams.update({
    "font.family": SERIF, "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight",
    "mathtext.fontset": "stix", "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.frameon": False})
NAME = {"enron": "Enron-Spam (email)", "sms": "SMS Spam Collection"}


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("ok", name)


# ---------------------------------------------------------------- Hình 1.1
def fig_timeline():
    ev = [(1943, "McCulloch–Pitts:\nnơ-ron nhân tạo"), (1950, "Turing:\nphép thử Turing"),
          (1956, "Hội thảo Dartmouth:\nra đời thuật ngữ AI"), (1958, "Perceptron\n(Rosenblatt)"),
          (1966, "ELIZA\n(Weizenbaum)"), (1986, "Lan truyền ngược\n(Rumelhart, Hinton, Williams)"),
          (1997, "Deep Blue thắng\nKasparov"), (2006, "Mạng niềm tin sâu\n(Hinton)"),
          (2012, "AlexNet thắng\nImageNet"), (2016, "AlphaGo thắng\nLee Sedol"),
          (2017, "Transformer"), (2022, "ChatGPT")]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_xlim(1938, 2028); ax.set_ylim(-4.6, 4.6); ax.axis("off")
    ax.axhline(0, color=NAVY, lw=2)
    for a, b, lab in ((1974, 1980, "Mùa đông AI\nthứ nhất"), (1987, 1993, "Mùa đông AI\nthứ hai")):
        ax.axvspan(a, b, ymin=0.46, ymax=0.54, color="#9CA3AF", alpha=.8)
        ax.text((a + b) / 2, -0.55, lab, ha="center", va="top", fontsize=8, color=GREY, style="italic")
    ax.axvspan(1980, 1987, ymin=0.46, ymax=0.54, color="#F2C078", alpha=.9)
    ax.text(1983.5, 0.5, "Hệ chuyên gia", ha="center", va="bottom", fontsize=8, color="#92590B", style="italic")
    lv = [2.2, -2.3, 3.7, -3.9, 2.2, -2.3, 3.7, -3.9, 2.2, -2.3, 3.7, -3.9]
    for (y, t), h in zip(ev, lv):
        ax.plot([y, y], [0, h * .80], color=GREY, lw=.8)
        ax.plot(y, 0, "o", color=SPAM if y >= 2006 else HAM, ms=6, zorder=3)
        ax.text(y, h, f"{y}\n{t}", ha="center", va="bottom" if h > 0 else "top", fontsize=7.6, linespacing=1.15)
    save(fig, "f_timeline_ai.png")


# ---------------------------------------------------------------- Hình 1.2
def fig_ai_venn():
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.add_patch(Ellipse((5, 3), 9.6, 5.6, fc="#EAF1FB", ec=NAVY, lw=1.6))
    ax.add_patch(Ellipse((5.8, 2.75), 6.4, 4.0, fc="#D5E4F7", ec=NAVY, lw=1.4))
    ax.add_patch(Ellipse((7.2, 2.2), 3.0, 2.0, fc="#B9D2F0", ec=NAVY, lw=1.2))
    ax.text(5, 5.25, "TRÍ TUỆ NHÂN TẠO (AI)", ha="center", fontweight="bold", color=NAVY)
    ax.text(1.9, 3.7, "Tìm kiếm &\nlập kế hoạch\n\nBiểu diễn tri thức\n& suy luận logic\n\nHệ chuyên gia\n\nRobot học", ha="center", va="center", fontsize=8.6, color="#374151")
    ax.text(5.8, 4.4, "HỌC MÁY (Machine Learning)", ha="center", fontweight="bold", fontsize=10, color=NAVY)
    ax.text(4.55, 3.3, "Naïve Bayes\nCây quyết định\nSVM · k-NN\nHồi quy logistic", ha="center", va="center", fontsize=8.4, color=SPAM, fontweight="bold")
    ax.text(7.2, 2.7, "HỌC SÂU", ha="center", fontweight="bold", fontsize=9.5, color=NAVY)
    ax.text(7.2, 2.05, "Mạng nơ-ron sâu\nTransformer", ha="center", va="center", fontsize=8.2)
    save(fig, "f_ai_venn.png")


# ---------------------------------------------------------------- Hình 1.3
def fig_graphical():
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.add_patch(Circle((5, 4.1), .55, fc="#FBE3D6", ec=SPAM, lw=1.8))
    ax.text(5, 4.1, "C", ha="center", va="center", fontsize=15, style="italic")
    ax.text(5.9, 4.5, "lớp: spam / ham", fontsize=9, color=GREY)
    xs = [1.4, 3.3, 5.2, 8.6]
    labs = ["$w_1$", "$w_2$", "$w_3$", "$w_n$"]
    for x, l in zip(xs, labs):
        ax.add_patch(Circle((x, 1.2), .5, fc="#DCE9F8", ec=HAM, lw=1.6))
        ax.text(x, 1.2, l, ha="center", va="center", fontsize=13)
        ax.add_patch(FancyArrowPatch((5, 3.55), (x, 1.72), arrowstyle="-|>", mutation_scale=13, color=NAVY, lw=1.2))
    ax.text(6.9, 1.2, "· · ·", fontsize=16, ha="center", va="center")
    ax.text(0.1, 4.1, "$P(C)$", fontsize=12, color=SPAM)
    ax.text(0.1, 2.3, "$P(w_i\\mid C)$", fontsize=12, color=HAM)
    ax.text(5, 0.25, "Các từ $w_i$ độc lập có điều kiện khi biết lớp $C$ (giả thiết ngây thơ)",
            ha="center", fontsize=9.5, color="#374151")
    save(fig, "f_graphical.png")


# ---------------------------------------------------------------- Graphviz
def dot(name, src, dpi=200):
    if shutil.which("dot") is None:
        print(f"BỎ QUA {name}.png: chưa cài Graphviz (lệnh 'dot'). Giữ nguyên hình có sẵn trong figures/.")
        return
    src = src.replace("Liberation Serif", SERIF)
    (FIG / f"{name}.dot").write_text(src, encoding="utf-8")
    subprocess.run(["dot", "-Tpng", f"-Gdpi={dpi}", str(FIG / f"{name}.dot"), "-o", str(FIG / f"{name}.png")], check=True)
    print("ok", name)


NODE = 'fontname="Liberation Serif" fontsize=12 shape=box style="rounded,filled" fillcolor="#EAF1FB" color="#1F3A5F" margin="0.12,0.07"'


def fig_flow():
    dot("f_flow_system", f'''digraph G {{
  rankdir=TB; nodesep=0.28; ranksep=0.30; fontname="Liberation Serif";
  node [{NODE}]; edge [color="#1F3A5F" arrowsize=0.8 fontname="Liberation Serif" fontsize=10];
  raw [label="Dữ liệu thô\\n(Enron-Spam CSV, SMS TSV)" fillcolor="#F3F4F6"];
  load [label="① Đọc dữ liệu & gán nhãn\\nham = 0, spam = 1"];
  dedup [label="② Loại trùng lặp\\nvà thư rỗng"];
  split [label="③ Chia phân tầng\\n80 % huấn luyện / 20 % kiểm tra"];
  subgraph cluster_train {{ label="GIAI ĐOẠN HUẤN LUYỆN" labeljust=l labelloc=b style="rounded,dashed" color="#2C6FBB" fontcolor="#2C6FBB" fontname="Liberation Serif" fontsize=12;
    pre1 [label="④ Tiền xử lý & tách token"];
    voc [label="⑤ Xây từ điển (min_df = 2)"];
    vec1 [label="⑥ Vector hoá: ma trận đếm\\nCSR  n × |V|"];
    fit [label="⑦ Huấn luyện Naïve Bayes\\nlog P(c),  log θ(c,w) có làm trơn Laplace" fillcolor="#FBE3D6" color="#D9622B"];
    model [label="Mô hình đã huấn luyện\\n(lưu .pkl)" fillcolor="#FDF3C8"];
    pre1 -> voc -> vec1 -> fit -> model; }}
  subgraph cluster_test {{ label="GIAI ĐOẠN KIỂM TRA / SỬ DỤNG" labeljust=l labelloc=b style="rounded,dashed" color="#D9622B" fontcolor="#D9622B" fontname="Liberation Serif" fontsize=12;
    pre2 [label="⑧ Tiền xử lý & vector hoá\\ntheo từ điển đã có"];
    inf [label="⑨ Tính log P(c) + Σ nᵥ·log θ(c,w)\\nchuẩn hoá bằng log-sum-exp" fillcolor="#FBE3D6" color="#D9622B"];
    dec [label="⑩ So sánh P(spam | d) với ngưỡng\\n→ nhãn SPAM / HAM"];
    ev [label="⑪ Đánh giá: Accuracy, Precision,\\nRecall, F1, ROC-AUC"];
    pre2 -> inf -> dec -> ev; }}
  raw -> load -> dedup -> split;
  split -> pre1 [label=" tập huấn luyện"];
  split -> pre2 [label=" tập kiểm tra"];
  model -> inf [style=dashed label=" tham số" constraint=false];
}}''')
    dot("f_flow_infer", f'''digraph G {{
  rankdir=LR; nodesep=0.2; ranksep=0.28;
  node [{NODE}]; edge [color="#1F3A5F" arrowsize=0.8 fontname="Liberation Serif" fontsize=10];
  a [label="Thư /\\ntin nhắn\\nmới" fillcolor="#F3F4F6"];
  b [label="Chuẩn hoá\\n(HTML, URL,\\nsố, tiền)"];
  c [label="Tách\\ntoken"];
  d [label="Đếm theo\\ntừ điển\\n(bỏ OOV)"];
  e [label="log P(c|d) cho\\nham và spam" fillcolor="#FBE3D6" color="#D9622B"];
  f [label="P(spam|d)\\n≥ ngưỡng ?" shape=diamond style=filled fillcolor="#FDF3C8" margin="0.02,0.02"];
  g [label="SPAM" fillcolor="#F8C9B3" color="#D9622B"]; h [label="HAM" fillcolor="#CFE2F7"];
  a -> b -> c -> d -> e -> f; f -> g [label="có"]; f -> h [label="không"];
}}''')


def fig_arch():
    dot("f_arch", f'''digraph G {{
  rankdir=TB; nodesep=0.3; ranksep=0.5;
  node [{NODE}]; edge [color="#1F3A5F" arrowsize=0.8];
  subgraph cluster_ui {{ label="Giao diện dòng lệnh & thực nghiệm" style="rounded" color="#9CA3AF" fontname="Liberation Serif" fontsize=11;
    train [label="train.py\\nhuấn luyện & lưu mô hình" fillcolor="#FDF3C8"];
    predict [label="predict.py\\nphân loại + giải thích" fillcolor="#FDF3C8"];
    exp [label="experiments.py\\nthực nghiệm" fillcolor="#FDF3C8"];
    tests [label="tests/\\n23 kiểm thử đơn vị" fillcolor="#FDF3C8"]; }}
  pipe [label="pipeline.py\\nSpamFilter: fit · predict · explain · save/load" fillcolor="#FBE3D6" color="#D9622B"];
  pre [label="preprocessing.py\\nTextPreprocessor"];
  vec [label="vectorizer.py\\nBagOfWords"];
  nb [label="naive_bayes.py\\nMultinomialNB · BernoulliNB"];
  met [label="metrics.py\\nP · R · F1 · ROC-AUC"];
  dat [label="data.py\\nđọc · loại trùng · chia tập"];
  train -> pipe; predict -> pipe; tests -> pipe; exp -> pipe;
  exp -> met; exp -> dat; train -> dat; train -> met;
  pipe -> pre; pipe -> vec; pipe -> nb;
}}''')


# ---------------------------------------------------------------- Dữ liệu
def fig_classdist():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, k in zip(axes, ("enron", "sms")):
        s = R["stats"][k]
        x = np.arange(2); w = .36
        b1 = ax.bar(x - w / 2, [s["raw_ham"], s["raw_spam"]], w, color="#9CA3AF", label="Gốc")
        b2 = ax.bar(x + w / 2, [s["ham"], s["spam"]], w, color=[HAM, SPAM], label="Sau loại trùng")
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height()):,}".replace(",", "."),
                    ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels(["Ham (thư tốt)", "Spam (thư rác)"])
        ax.set_title(NAME[k]); ax.set_ylabel("Số mẫu"); ax.set_ylim(0, max(s["raw_ham"], s["raw_spam"]) * 1.15)
    axes[1].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "f_classdist.png")


def fig_lengths():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, k in zip(axes, ("enron", "sms")):
        s = R["stats"][k]
        h, sp = np.array(s["tok_lengths_ham"]) + 1, np.array(s["tok_lengths_spam"]) + 1
        bins = np.logspace(0, np.log10(max(h.max(), sp.max())), 30)
        ax.hist(h, bins, color=HAM, alpha=.65, label="Ham", density=True)
        ax.hist(sp, bins, color=SPAM, alpha=.65, label="Spam", density=True)
        ax.set_xscale("log"); ax.set_xlabel("Số token / văn bản (thang log)"); ax.set_ylabel("Mật độ")
        ax.set_title(NAME[k]); ax.legend()
    fig.tight_layout()
    save(fig, "f_lengths.png")


# ---------------------------------------------------------------- Kết quả
def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, k in zip(axes, ("enron", "sms")):
        m = R["main"][k]["metrics"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1.3)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}".replace(",", ".") + f"\n({norm[i, j] * 100:.1f} %)", ha="center", va="center",
                        color="white" if norm[i, j] > .5 else "#111", fontsize=10.5)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Dự đoán\nHAM", "Dự đoán\nSPAM"]); ax.set_yticklabels(["Thực tế\nHAM", "Thực tế\nSPAM"])
        ax.set_title(NAME[k]); ax.spines[:].set_visible(False); ax.tick_params(length=0)
    fig.tight_layout()
    save(fig, "f_confusion.png")


def fig_alpha():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, k in zip(axes, ("enron", "sms")):
        rows = R["alpha"][k]["rows"]
        a = [r["alpha"] for r in rows]
        ax.plot(a, [r["test"]["f1"] * 100 for r in rows], "o-", color=SPAM, label="F1 (tập kiểm tra)")
        ax.plot(a, [r["cv_f1"] * 100 for r in rows], "s--", color=HAM, label="F1 (CV 5 lần trên tập train)")
        ax.axvline(1.0, color=GREY, ls=":", lw=1); ax.text(1.08, ax.get_ylim()[0], "α = 1", color=GREY, fontsize=8.5, va="bottom")
        ax.set_xscale("log"); ax.set_xlabel("Hệ số làm trơn α"); ax.set_ylabel("F1 (%)"); ax.set_title(NAME[k])
    axes[0].legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    save(fig, "f_alpha.png")


def fig_roc_pr():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for k, c in (("enron", HAM), ("sms", SPAM)):
        t = R["threshold"][k]
        axes[0].plot(t["roc"]["fpr"], t["roc"]["tpr"], color=c, lw=1.8, label=f"{NAME[k]} (AUC = {t['roc_auc']:.4f})")
        axes[1].plot(t["pr"]["recall"], t["pr"]["precision"], color=c, lw=1.8, label=f"{NAME[k]} (AP = {t['avg_precision']:.4f})")
    axes[0].plot([0, 1], [0, 1], ":", color=GREY)
    axes[0].set_xlabel("Tỉ lệ dương giả (FPR)"); axes[0].set_ylabel("Tỉ lệ dương thật (Recall)"); axes[0].set_title("Đường cong ROC")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Đường cong Precision–Recall")
    axes[1].set_ylim(0.4, 1.02)
    for ax in axes: ax.legend(fontsize=7.8, loc="lower right" if ax is axes[0] else "lower left")
    fig.tight_layout()
    save(fig, "f_roc_pr.png")


def fig_scores():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, k in zip(axes, ("enron", "sms")):
        t = R["threshold"][k]
        lo, hi = -300, 300
        bins = np.linspace(lo, hi, 61)
        ax.hist(np.clip(t["score_ham"], lo, hi), bins, color=HAM, alpha=.7, label="Ham")
        ax.hist(np.clip(t["score_spam"], lo, hi), bins, color=SPAM, alpha=.7, label="Spam")
        ax.axvline(0, color="k", lw=1, ls="--"); ax.set_yscale("log")
        ax.set_xlabel("log-odds  log P(spam|d) − log P(ham|d)"); ax.set_ylabel("Số văn bản (log)")
        ax.set_title(NAME[k]); ax.legend()
    fig.tight_layout()
    save(fig, "f_scores.png")


def fig_learning():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, k in zip(axes, ("enron", "sms")):
        rows = R["learning"][k]
        n = [r["n_train"] for r in rows]
        ax.plot(n, [r["f1"] * 100 for r in rows], "o-", color=SPAM, label="F1")
        ax.plot(n, [r["accuracy"] * 100 for r in rows], "s--", color=HAM, label="Accuracy")
        ax.set_xscale("log"); ax.set_xlabel("Số mẫu huấn luyện (thang log)"); ax.set_ylabel("%"); ax.set_title(NAME[k])
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "f_learning.png")


def fig_words():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for col, k in enumerate(("enron", "sms")):
        w = R["words"][k]
        for row, (key, color, ttl) in enumerate((("spam_ratio", SPAM, "nghiêng về SPAM"), ("ham_ratio", HAM, "nghiêng về HAM"))):
            ax = axes[row, col]
            items = w[key][:10][::-1]
            ax.barh([i[0] for i in items], [i[1] for i in items], color=color)
            ax.set_title(f"{NAME[k].split(' (')[0]} – {ttl}", fontsize=10)
            ax.set_xlabel("log θ(spam,w) − log θ(ham,w)", fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    save(fig, "f_words.png")


def fig_models():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=False)
    short = {"Baseline: luôn đoán lớp đa số": "Baseline", "Multinomial NB (cài đặt riêng)": "Multinomial NB",
             "Bernoulli NB (cài đặt riêng)": "Bernoulli NB", "Complement NB (scikit-learn)": "Complement NB",
             "Multinomial NB + TF-IDF (scikit-learn)": "MNB + TF-IDF", "Logistic Regression + TF-IDF": "Logistic Reg.",
             "Linear SVM + TF-IDF": "Linear SVM"}
    for ax, k in zip(axes, ("enron", "sms")):
        rows = [r for r in R["models"][k] if r["model"] in short and r["f1"] > 0]
        names = [short[r["model"]] for r in rows][::-1]
        f1 = [r["f1"] * 100 for r in rows][::-1]
        cols = [SPAM if "Multinomial NB" == n or n == "Bernoulli NB" else HAM for n in names]
        ax.barh(names, f1, color=cols)
        lo = min(f1) - 3
        ax.set_xlim(lo, 100.6)
        for i, v in enumerate(f1): ax.text(v + .05, i, f"{v:.2f}", va="center", fontsize=8.5)
        ax.set_title(NAME[k]); ax.set_xlabel("F1 (%)")
    fig.tight_layout()
    save(fig, "f_models.png")


# ---------------------------------------------------------------- Ảnh terminal
def term_image(name, txt_path, max_cols=104):
    import unicodedata; text = unicodedata.normalize("NFC", Path(txt_path).read_text(encoding="utf-8")).rstrip("\n")
    lines = []
    for ln in text.splitlines():
        while len(ln) > max_cols:
            cut = ln.rfind(" ", 0, max_cols)
            cut = cut if cut > 30 else max_cols
            lines.append(ln[:cut]); ln = "    " + ln[cut:].lstrip()
        lines.append(ln)
    font = ImageFont.truetype(_mono(), 15)
    bold = ImageFont.truetype(_mono(True), 15)
    lh, pad, top = 21, 16, 34
    cw = font.getlength("M")
    w = int(max(len(l) for l in lines) * cw + 2 * pad + 8)
    h = top + pad + lh * len(lines)
    im = Image.new("RGB", (w, h), "#1E1E1E")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, top - 4], fill="#3A3A3A")
    for i, c in enumerate(("#FF5F56", "#FFBD2E", "#27C93F")):
        d.ellipse([12 + i * 20, 9, 24 + i * 20, 21], fill=c)
    d.text((w // 2 - 60, 7), "Terminal — spam_nb", font=font, fill="#BBBBBB")
    y = top + 6
    for ln in lines:
        col, f = "#D4D4D4", font
        if ln.startswith("$ "):
            col, f = "#7EE787", bold
        elif re.search(r"\bSPAM\b", ln) and ln.startswith("Kết luận"):
            col, f = "#FF7B72", bold
        elif re.search(r"\bHAM\b", ln) and ln.startswith("Kết luận"):
            col, f = "#79C0FF", bold
        elif ln.strip() in ("OK",) or ln.endswith("... ok"):
            col = "#7EE787"
        elif ln.startswith("Từ đẩy") or ln.startswith("  Từ đẩy"):
            col = "#FF7B72" if "SPAM" in ln.split(":")[0] else "#79C0FF"
        elif ln.startswith(("Nội dung", "Log-odds", "Mô hình")):
            col = "#E3B341" if ln.startswith("Log-odds") else "#D4D4D4"
        elif set(ln.strip()) <= set("-="):
            col = "#666666"
        d.text((pad, y), ln, font=f, fill=col)
        y += lh
    im.save(FIG / f"{name}.png")
    print("ok", name, im.size)


# ---------------------------------------------------------------- Công thức
EQS = {
    "eq_conditional": r"$P(A\mid B)=\dfrac{P(A\cap B)}{P(B)}$",
    "eq_bayes": r"$P(A\mid B)=\dfrac{P(B\mid A)\,P(A)}{P(B)}$",
    "eq_total": r"$P(B)=\sum_{i=1}^{k}P(B\mid A_i)\,P(A_i)$",
    "eq_bayes_class": r"$P(c\mid d)=\dfrac{P(d\mid c)\,P(c)}{P(d)}$",
    "eq_map": r"$\hat{c}=\arg\max_{c\in\{\mathrm{ham},\,\mathrm{spam}\}}\;P(c\mid d)=\arg\max_{c}\;P(d\mid c)\,P(c)$",
    "eq_indep": r"$P(d\mid c)=P(w_1,w_2,\ldots,w_n\mid c)\approx\prod_{i=1}^{n}P(w_i\mid c)$",
    "eq_nb": r"$\hat{c}=\arg\max_{c}\;\left[\log P(c)+\sum_{i=1}^{n}\log P(w_i\mid c)\right]$",
    "eq_multi": r"$P(d\mid c)=\dfrac{(\sum_{w}n_w)!}{\prod_{w}n_w!}\prod_{w\in V}\theta_{c,w}^{\,n_w},\qquad \sum_{w\in V}\theta_{c,w}=1$",
    "eq_prior": r"$\hat{P}(c)=\dfrac{M_c}{M}$",
    "eq_mle": r"$\hat{\theta}_{c,w}=\dfrac{N_{c,w}}{\sum_{w'\in V}N_{c,w'}}$",
    "eq_laplace": r"$\hat{\theta}_{c,w}=\dfrac{N_{c,w}+\alpha}{\sum_{w'\in V}N_{c,w'}+\alpha\,|V|}=\dfrac{N_{c,w}+\alpha}{N_c+\alpha\,|V|}$",
    "eq_dirichlet": r"$\theta_c\sim\mathrm{Dirichlet}(\alpha,\ldots,\alpha)\;\Rightarrow\;\hat{\theta}_{c,w}=\mathbb{E}[\theta_{c,w}\mid \mathcal{D}]=\dfrac{N_{c,w}+\alpha}{N_c+\alpha|V|}$",
    "eq_multi_score": r"$\mathrm{score}(c\mid d)=\log\hat{P}(c)+\sum_{w\in V}n_w(d)\,\log\hat{\theta}_{c,w}$",
    "eq_posterior": r"$P(c\mid d)=\dfrac{\exp\,\mathrm{score}(c\mid d)}{\sum_{c'}\exp\,\mathrm{score}(c'\mid d)}$",
    "eq_lse": r"$\log\sum_{c'}e^{s_{c'}}=m+\log\sum_{c'}e^{s_{c'}-m},\qquad m=\max_{c'}s_{c'}$",
    "eq_bern": r"$P(d\mid c)=\prod_{w\in V}p_{c,w}^{\,b_w}\,(1-p_{c,w})^{1-b_w},\qquad b_w=\mathbb{1}[n_w(d)>0]$",
    "eq_bern_est": r"$\hat{p}_{c,w}=\dfrac{\mathrm{df}_{c,w}+\alpha}{M_c+2\alpha}$",
    "eq_logodds": r"$\log\dfrac{P(\mathrm{spam}\mid d)}{P(\mathrm{ham}\mid d)}=\log\dfrac{P(\mathrm{spam})}{P(\mathrm{ham})}+\sum_{w}n_w(d)\,\log\dfrac{\theta_{\mathrm{spam},w}}{\theta_{\mathrm{ham},w}}$",
    "eq_thresh": r"$\mathrm{SPAM}\;\Longleftrightarrow\; P(\mathrm{spam}\mid d)\geq\dfrac{\lambda}{1+\lambda}$",
    "eq_acc": r"$\mathrm{Accuracy}=\dfrac{TP+TN}{TP+TN+FP+FN}$",
    "eq_prec": r"$\mathrm{Precision}=\dfrac{TP}{TP+FP}$",
    "eq_rec": r"$\mathrm{Recall}=\dfrac{TP}{TP+FN}$",
    "eq_f1": r"$F_1=\dfrac{2\cdot\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}$",
    "eq_fpr": r"$\mathrm{FPR}=\dfrac{FP}{FP+TN}$",
    "eq_wacc": r"$\mathrm{WAcc}_\lambda=\dfrac{\lambda\cdot TN+TP}{\lambda\cdot N_{\mathrm{ham}}+N_{\mathrm{spam}}}$",
    "eq_auc": r"$\mathrm{AUC}=P\!\left(s(d^{+})>s(d^{-})\right)=\dfrac{\sum_{d\in\mathrm{spam}}\mathrm{rank}(d)-\frac{n_+(n_++1)}{2}}{n_+\,n_-}$",
    "eq_cv": r"$\bar{m}=\dfrac{1}{K}\sum_{k=1}^{K}m_k,\qquad s=\sqrt{\dfrac{1}{K-1}\sum_{k=1}^{K}(m_k-\bar{m})^2}$",
    "eq_tfidf": r"$\mathrm{tf\text{-}idf}(w,d)=n_w(d)\cdot\log\dfrac{N}{\mathrm{df}(w)}$",
    "eq_neuron": r"$y=\varphi\!\left(\sum_{i}w_ix_i+b\right)$",
}


def fig_equations():
    for name, tex in EQS.items():
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, tex, fontsize=13.5)
        fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=False, facecolor="white")
        plt.close(fig)
    print("ok equations", len(EQS))


if __name__ == "__main__":
    fig_timeline(); fig_ai_venn(); fig_graphical(); fig_flow(); fig_arch()
    fig_classdist(); fig_lengths(); fig_confusion(); fig_alpha(); fig_roc_pr(); fig_scores()
    fig_learning(); fig_words(); fig_models(); fig_equations()
    for n in ("tests", "train_enron", "train_sms", "pred_email_spam", "pred_email_ham", "pred_sms", "pred_limit", "pred_phish", "pred_thr"):
        term_image("t_" + n, ROOT / "results/demo" / f"{n}.txt")
