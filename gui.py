"""Giao diện đồ họa cho bộ lọc thư rác Naïve Bayes (Tkinter – có sẵn trong Python, không cần cài thêm).

Chạy:   python gui.py
Dùng:   chọn mô hình (SMS hoặc Email) -> dán nội dung -> bấm "Phân loại" (hoặc Ctrl+Enter).
"""
import contextlib
import os
import queue
import re
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from spamfilter import SpamFilter  # noqa: E402
import train  # noqa: E402   (hàm run_training dùng chung với dòng lệnh)

MODELS = {
    "sms": ("SMS – tin nhắn", BASE / "models" / "sms_mnb.pkl"),
    "enron": ("Email (Enron-Spam)", BASE / "models" / "enron_mnb.pkl"),
}

DATASETS = {
    "sms": ("SMS Spam Collection", BASE / "data" / "sms.tsv", BASE / "models" / "sms_mnb.pkl"),
    "enron": ("Enron-Spam (email)", BASE / "data" / "enron_spam_data.csv", BASE / "models" / "enron_mnb.pkl"),
}

SAMPLES = [
    ("— Chọn một mẫu có sẵn —", "", None),
    ("SMS spam: trúng thưởng, gọi số điện thoại", "Congratulations! You've won a £1000 cash prize. Call 09061701234 now to claim your reward.", "sms"),
    ("SMS spam: nghỉ dưỡng miễn phí", "URGENT! Your mobile number has been selected for a FREE holiday. Text WIN to 80082 now!", "sms"),
    ("SMS ham: hẹn gặp bạn bè", "Ok, I will call you after class. Wait for me at the canteen, ok?", "sms"),
    ("SMS ham: hỏi đi ăn trưa", "Are we still meeting for lunch tomorrow? I'm running a bit late, sorry.", "sms"),
    ("Email spam: trúng xổ số", "Subject: Congratulations!!! You have WON $1,000,000 in the International Lottery\n\nDear Winner, your email was "
     "selected in our lottery draw. To claim your prize, reply with your full name and bank account today. Click "
     "http://claim-prize.example.com now!", "enron"),
    ("Email spam: thuốc giá rẻ", "Subject: Cheap V1agra and Cialis - save 80% today\n\nBuy cheap pills online, no prescription needed! "
     "Best prices, fast shipping. Order now at http://pills.example.biz", "enron"),
    ("Email ham: thư công việc", "Subject: Project schedule and budget review\n\nHi Kevin, can we move tomorrow's project review to 3 pm? "
     "I attached the updated budget spreadsheet. Please check it before the conference call. Thanks, Sara", "enron"),
    ("Hạn chế: spam tiếng Việt không dấu", "Ban da trung thuong 500 trieu dong. Lien he ngay 0912345678 de nhan giai thuong!", "sms"),
]

SPECIAL = {"__phone__": "‹số điện thoại / mã›", "__url__": "‹đường link›", "__email__": "‹địa chỉ email›",
           "__money__": "‹số tiền›", "__num__": "‹con số›"}

SPAM_C, HAM_C, GREY = "#D9622B", "#2C6FBB", "#6B7280"
OK_GREEN, NAVY = "#2E9E5B", "#1F3A5F"


def vn(x, nd=2, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bộ lọc thư rác Naïve Bayes – Nhóm 16")
        if os.environ.get("SPAM_GUI_SCALING"):                      # chỉ dùng để thử nghiệm: giả lập màn hình DPI cao
            self.tk.call("tk", "scaling", float(os.environ["SPAM_GUI_SCALING"]) * 96 / 72)
        self.sc = max(1.0, self.winfo_fpixels("1i") / 96)           # hệ số co giãn màn hình (100% = 1.0, 150% = 1.5 ...)
        self.configure(bg="#F3F4F6")
        self._setup_style()

        self.clfs = {}
        self.model_key = tk.StringVar(value="sms")
        self.threshold = tk.DoubleVar(value=0.5)
        self.last_ex = None
        self.last_prob = None

        self._build_header()
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        tab1, tab2 = ttk.Frame(self.nb), ttk.Frame(self.nb)
        self.nb.add(tab1, text="   Phân loại   ")
        self.nb.add(tab2, text="   Huấn luyện   ")
        self._build_history(tab1)
        body = ttk.Frame(tab1, padding=(4, 0, 4, 0))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=5, uniform="c")
        body.columnconfigure(1, weight=6, uniform="c")
        body.rowconfigure(0, weight=1)
        self._build_left(body)
        self._build_right(body)
        self._build_train(tab2)

        self.bind("<Control-Return>", lambda e: self.classify() if self.nb.index("current") == 0 else None)
        self._load_model(silent=False)
        self._reset_result()
        self._fit_window()

    def _fit_window(self):
        """Cỡ cửa sổ theo nội dung và độ co giãn màn hình, không vượt quá màn hình; đặt giữa màn hình."""
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(max(int(1080 * self.sc), self.winfo_reqwidth()), sw - 40)
        h = min(max(int(760 * self.sc), self.winfo_reqheight()), sh - 90)
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        self.minsize(min(int(900 * self.sc), w), min(int(620 * self.sc), h))

    # ------------------------------------------------------------------ giao diện
    def _setup_style(self):
        fams = set(tkfont.families())
        fam = next((f for f in ("Segoe UI", "Helvetica Neue", "Noto Sans", "DejaVu Sans") if f in fams), "TkDefaultFont")
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(name).configure(family=fam, size=10)
        self.f_big = tkfont.Font(family=fam, size=24, weight="bold")
        self.f_title = tkfont.Font(family=fam, size=17, weight="bold")
        self.f_bold = tkfont.Font(family=fam, size=10, weight="bold")
        self.f_small = tkfont.Font(family=fam, size=9)
        self.f_mono = tkfont.Font(family="Consolas" if "Consolas" in fams else "DejaVu Sans Mono", size=10)
        st = ttk.Style(self)
        st.theme_use("vista" if "vista" in st.theme_names() else "clam")
        st.configure("TFrame", background="#F3F4F6")
        st.configure("TLabelframe", background="#F3F4F6")
        st.configure("TLabelframe.Label", background="#F3F4F6", foreground=NAVY, font=self.f_bold)
        st.configure("TLabel", background="#F3F4F6")
        st.configure("TRadiobutton", background="#F3F4F6")
        st.configure("Main.TButton", font=self.f_bold, padding=(14, 6))
        st.configure("TNotebook.Tab", font=self.f_bold, padding=(10, 5))
        st.configure("TCheckbutton", background="#F3F4F6")
        st.configure("Treeview", rowheight=int(22 * self.sc))

    def _build_header(self):
        h = tk.Frame(self, bg=NAVY)
        h.pack(fill="x")
        tk.Label(h, text="PHÂN LOẠI THƯ RÁC BẰNG NAÏVE BAYES", bg=NAVY, fg="white", font=self.f_title).pack(anchor="w", padx=16, pady=(10, 0))
        tk.Label(h, text="Đề tài 42 – Nhóm 16 – Trí tuệ nhân tạo · Chỉ hỗ trợ nội dung tiếng Anh", bg=NAVY, fg="#C7D2E5",
                 font=self.f_small).pack(anchor="w", padx=16, pady=(0, 10))

    def _build_left(self, parent):
        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=10)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        f1 = ttk.LabelFrame(left, text=" 1. Chọn mô hình ", padding=8)
        f1.grid(row=0, column=0, sticky="ew")
        for i, (k, (label, _)) in enumerate(MODELS.items()):
            ttk.Radiobutton(f1, text=label, value=k, variable=self.model_key, command=self._on_model_change).grid(row=0, column=i, sticky="w", padx=(0, 18))
        ttk.Button(f1, text="Chọn tệp…", command=self._pick_model).grid(row=0, column=2, sticky="e")
        f1.columnconfigure(2, weight=1)
        self.model_info = ttk.Label(f1, text="", foreground=GREY, font=self.f_small)
        self.model_info.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        f2 = ttk.LabelFrame(left, text=" 2. Nhập nội dung thư / tin nhắn ", padding=8)
        f2.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        f2.rowconfigure(1, weight=1)
        f2.columnconfigure(0, weight=1)
        self.sample_box = ttk.Combobox(f2, state="readonly", values=[s[0] for s in SAMPLES])
        self.sample_box.current(0)
        self.sample_box.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.sample_box.bind("<<ComboboxSelected>>", self._on_sample)
        wrap = ttk.Frame(f2)
        wrap.grid(row=1, column=0, sticky="nsew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        self.text = tk.Text(wrap, wrap="word", width=10, height=8, font=self.f_mono, undo=True, relief="solid", borderwidth=1, padx=6, pady=6)
        sb = ttk.Scrollbar(wrap, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.text.tag_configure("spamw", background="#FAD3BF")
        self.text.tag_configure("hamw", background="#CBE0F6")
        self.text.bind("<Key>", lambda e: self.text.tag_remove("spamw", "1.0", "end") or self.text.tag_remove("hamw", "1.0", "end"))

        bt = ttk.Frame(f2)
        bt.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(bt, text="▶  Phân loại", style="Main.TButton", command=self.classify).pack(side="left")
        ttk.Button(bt, text="Xoá", command=self._clear).pack(side="left", padx=6)
        ttk.Button(bt, text="Mở tệp .txt…", command=self._open_file).pack(side="left")
        ttk.Label(bt, text="Ctrl+Enter", foreground=GREY, font=self.f_small).pack(side="right")

        f3 = ttk.LabelFrame(left, text=" Ngưỡng quyết định (gán SPAM khi P(spam) ≥ ngưỡng) ", padding=8)
        f3.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        f3.columnconfigure(0, weight=1)
        self.thr_label = ttk.Label(f3, text="0,50", font=self.f_bold, width=5, anchor="e")
        self.thr_label.grid(row=0, column=1, padx=(8, 0))
        ttk.Scale(f3, from_=0.5, to=0.99, variable=self.threshold, command=self._on_thr).grid(row=0, column=0, sticky="ew")
        ttk.Label(f3, text="Nâng ngưỡng → ít chặn nhầm thư tốt hơn, nhưng dễ bỏ lọt thư rác hơn (mục 2.9.9 của báo cáo).",
                  foreground=GREY, font=self.f_small, wraplength=430).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _build_right(self, parent):
        right = ttk.Frame(parent)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=10)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        f = ttk.LabelFrame(right, text=" 3. Kết quả ", padding=8)
        f.grid(row=0, column=0, sticky="ew")
        f.columnconfigure(0, weight=1)
        self.verdict = tk.Label(f, text="", font=self.f_big, fg="white", pady=8)
        self.verdict.grid(row=0, column=0, sticky="ew")
        self.gauge = tk.Canvas(f, width=300, height=int(58 * self.sc), bg="#F3F4F6", highlightthickness=0)
        self.gauge.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.gauge.bind("<Configure>", lambda e: self._draw_gauge())
        self.detail = ttk.Label(f, text="", font=self.f_small, foreground="#374151", justify="left", wraplength=520)
        self.detail.grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.warn = tk.Label(f, text="", fg="#B45309", bg="#F3F4F6", font=self.f_small, justify="left", wraplength=560, anchor="w")
        self.warn.grid(row=3, column=0, sticky="w")

        g = ttk.LabelFrame(right, text=" Những từ ảnh hưởng nhất đến quyết định (đóng góp vào log-odds) ", padding=8)
        g.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        g.rowconfigure(0, weight=1)
        g.columnconfigure(0, weight=1)
        self.chart = tk.Canvas(g, width=300, height=int(240 * self.sc), bg="white", highlightthickness=1, highlightbackground="#D1D5DB")
        self.chart.grid(row=0, column=0, sticky="nsew")
        self.chart.bind("<Configure>", lambda e: self._draw_chart())
        self.formula = ttk.Label(g, text="", font=self.f_small, foreground=GREY, wraplength=520)
        self.formula.grid(row=1, column=0, sticky="w", pady=(6, 0))
        right.bind("<Configure>", lambda e: [w.configure(wraplength=max(200, e.width - 50)) for w in (self.detail, self.warn, self.formula)])

    # ------------------------------------------------------------------ tab huấn luyện
    def _build_train(self, tab):
        self.tr_dataset = tk.StringVar(value="sms")
        self.tr_data = tk.StringVar(value=str(DATASETS["sms"][1]))
        self.tr_out = tk.StringVar(value=str(DATASETS["sms"][2]))
        self.tr_model = tk.StringVar(value="multinomial")
        self.tr_alpha, self.tr_mindf = tk.StringVar(value="1.0"), tk.StringVar(value="2")
        self.tr_test, self.tr_seed = tk.StringVar(value="0.2"), tk.StringVar(value="42")
        self.tr_dedup, self.tr_stop, self.tr_full = tk.BooleanVar(value=True), tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        self.trained = None
        self.q = None

        bar = ttk.Frame(tab, padding=(6, 4, 6, 8))            # thanh nút luôn nằm sát đáy, không bị khuất khi màn hình thấp
        bar.pack(side="bottom", fill="x")
        wrap = ttk.Frame(tab, padding=(4, 8, 4, 4))
        wrap.pack(fill="both", expand=True)
        wrap.columnconfigure(0, weight=5, uniform="t")
        wrap.columnconfigure(1, weight=6, uniform="t")
        wrap.rowconfigure(0, weight=1)
        left, right = ttk.Frame(wrap), ttk.Frame(wrap)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        left.columnconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        f1 = ttk.LabelFrame(left, text=" 1. Dữ liệu huấn luyện ", padding=8)
        f1.grid(row=0, column=0, sticky="ew")
        f1.columnconfigure(0, weight=1)
        for i, (k, (label, _, _)) in enumerate(DATASETS.items()):
            ttk.Radiobutton(f1, text=label, value=k, variable=self.tr_dataset, command=self._on_train_dataset).grid(row=0, column=i, sticky="w", padx=(0, 16))
        ttk.Entry(f1, textvariable=self.tr_data).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(f1, text="Duyệt…", command=lambda: self._browse(self.tr_data, [("Dữ liệu", "*.csv *.tsv *.txt"), ("Tất cả", "*.*")])).grid(row=1, column=2, padx=(6, 0), pady=(8, 0))
        self.dl_btn = ttk.Button(f1, text="Tải dữ liệu mẫu từ Internet", command=self._download_data)
        self.dl_btn.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        f2 = ttk.LabelFrame(left, text=" 2. Tham số (mặc định như trong báo cáo) ", padding=8)
        f2.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        f2.columnconfigure(1, weight=1)
        f2.columnconfigure(3, weight=1)
        ttk.Label(f2, text="Mô hình").grid(row=0, column=0, sticky="w")
        ttk.Combobox(f2, textvariable=self.tr_model, values=["multinomial", "bernoulli"], state="readonly", width=12).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(f2, text="Hệ số làm trơn α").grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Spinbox(f2, textvariable=self.tr_alpha, from_=0.001, to=10, increment=0.1, width=7).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(f2, text="min_df (từ tối thiểu)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(f2, textvariable=self.tr_mindf, from_=1, to=50, increment=1, width=7).grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(f2, text="Tỉ lệ tập kiểm tra").grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(6, 0))
        ttk.Spinbox(f2, textvariable=self.tr_test, from_=0.1, to=0.5, increment=0.05, width=7).grid(row=1, column=3, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(f2, text="Hạt giống (seed)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(f2, textvariable=self.tr_seed, width=9).grid(row=2, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Checkbutton(f2, text="Loại thư trùng lặp trước khi chia tập", variable=self.tr_dedup).grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Checkbutton(f2, text="Loại bỏ stopword", variable=self.tr_stop).grid(row=4, column=0, columnspan=4, sticky="w")
        ttk.Checkbutton(f2, text="Huấn luyện lại trên TOÀN BỘ dữ liệu sau khi đánh giá (mô hình lưu ra)", variable=self.tr_full).grid(row=5, column=0, columnspan=4, sticky="w")

        f3 = ttk.LabelFrame(left, text=" 3. Nơi lưu mô hình ", padding=8)
        f3.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        f3.columnconfigure(0, weight=1)
        ttk.Entry(f3, textvariable=self.tr_out).grid(row=0, column=0, sticky="ew")
        ttk.Button(f3, text="Lưu tại…", command=self._browse_save).grid(row=0, column=1, padx=(6, 0))

        self.tr_btn = ttk.Button(bar, text="▶  Bắt đầu huấn luyện", style="Main.TButton", command=self._start_training)
        self.tr_btn.pack(side="left")
        self.use_btn = ttk.Button(bar, text="Dùng mô hình này để phân loại →", command=self._use_trained, state="disabled")
        self.use_btn.pack(side="left", padx=8)
        ttk.Label(bar, text="Huấn luyện chạy nền nên cửa sổ không bị treo. SMS mất dưới 1 giây, Enron khoảng 10–30 giây.",
                  foreground=GREY, font=self.f_small).pack(side="left", padx=10)

        # ---- bên phải: tiến trình + kết quả
        f4 = ttk.LabelFrame(right, text=" Tiến trình ", padding=8)
        f4.grid(row=0, column=0, sticky="nsew")
        f4.rowconfigure(1, weight=1)
        f4.columnconfigure(0, weight=1)
        self.pbar = ttk.Progressbar(f4, mode="indeterminate")
        self.pbar.grid(row=0, column=0, sticky="ew", columnspan=2)
        self.logbox = tk.Text(f4, height=9, width=10, font=self.f_mono, bg="#1E1E1E", fg="#E5E7EB", relief="flat", padx=8, pady=6, state="disabled", wrap="word")
        lsb = ttk.Scrollbar(f4, command=self.logbox.yview)
        self.logbox.configure(yscrollcommand=lsb.set)
        self.logbox.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        lsb.grid(row=1, column=1, sticky="ns", pady=(8, 0))

        f5 = ttk.LabelFrame(right, text=" Kết quả trên tập kiểm tra ", padding=8)
        f5.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        f5.columnconfigure(0, weight=1)
        self.tr_banner = tk.Label(f5, text="Chưa huấn luyện", font=self.f_bold, fg="white", bg="#9CA3AF", pady=6)
        self.tr_banner.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.metric_box = tk.Frame(f5, bg="#F3F4F6")
        self.metric_box.grid(row=1, column=0, sticky="nw", pady=(8, 0))
        self.cm = tk.Canvas(f5, width=int(260 * self.sc), height=int(180 * self.sc), bg="#F3F4F6", highlightthickness=0)
        self.cm.grid(row=1, column=1, sticky="ne", pady=(8, 0), padx=(10, 0))
        self.tr_note = ttk.Label(f5, text="", font=self.f_small, foreground="#374151", wraplength=int(520 * self.sc), justify="left")
        self.tr_note.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        right.rowconfigure(1, weight=0)
        self._draw_cm(None)

    def _on_train_dataset(self):
        _, data, out = DATASETS[self.tr_dataset.get()]
        self.tr_data.set(str(data))
        self.tr_out.set(str(out))

    def _browse(self, var, types):
        p = filedialog.askopenfilename(title="Chọn tệp dữ liệu", initialdir=BASE / "data", filetypes=types)
        if p:
            var.set(p)

    def _browse_save(self):
        p = filedialog.asksaveasfilename(title="Lưu mô hình", initialdir=BASE / "models", defaultextension=".pkl", filetypes=[("Mô hình", "*.pkl")])
        if p:
            self.tr_out.set(p)

    def _log(self, line):
        self.logbox.configure(state="normal")
        self.logbox.insert("end", line + "\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def _busy(self, on):
        for b in (self.tr_btn, self.dl_btn):
            b.state(["disabled"] if on else ["!disabled"])
        if on:
            self.use_btn.state(["disabled"])
            self.pbar.start(12)
        else:
            self.pbar.stop()

    def _download_data(self):
        import download_data

        class W:                      # chuyển print() của download_data vào hàng đợi
            def __init__(self, q): self.q = q
            def write(self, t):
                if t.strip():
                    self.q.put(("log", t.rstrip()))
            def flush(self): pass

        self.q = queue.Queue()
        self._busy(True)
        self.logbox.configure(state="normal"); self.logbox.delete("1.0", "end"); self.logbox.configure(state="disabled")
        self._log("Đang tải dữ liệu (cần Internet)…")

        def work(q=self.q):
            try:
                with contextlib.redirect_stdout(W(q)):
                    download_data.main()
                q.put(("dl_done", None))
            except Exception as e:
                q.put(("error", f"Không tải được dữ liệu: {type(e).__name__}: {e}"))
        threading.Thread(target=work, daemon=True).start()
        self.after(100, self._poll_train)

    def _start_training(self):
        try:
            alpha, min_df, test, seed = float(self.tr_alpha.get().replace(",", ".")), int(self.tr_mindf.get()), float(self.tr_test.get().replace(",", ".")), int(self.tr_seed.get())
            assert alpha > 0 and min_df >= 1 and 0.05 <= test <= 0.5
        except Exception:
            messagebox.showerror("Tham số không hợp lệ", "Kiểm tra lại: α > 0, min_df ≥ 1 (số nguyên), tỉ lệ kiểm tra từ 0,05 đến 0,5, seed là số nguyên.")
            return
        data, out = Path(self.tr_data.get()), Path(self.tr_out.get())
        if not data.exists():
            messagebox.showerror("Không thấy dữ liệu", f"Không tìm thấy tệp:\n{data}\n\nBấm “Tải dữ liệu mẫu từ Internet” hoặc chọn tệp bằng nút “Duyệt…”.")
            return
        if out.exists() and not messagebox.askyesno("Ghi đè mô hình?", f"Tệp đã tồn tại:\n{out}\n\nGhi đè bằng mô hình mới?"):
            return
        self.trained = None
        self._busy(True)
        self.logbox.configure(state="normal"); self.logbox.delete("1.0", "end"); self.logbox.configure(state="disabled")
        self._draw_cm(None)
        for w in self.metric_box.winfo_children():
            w.destroy()
        self.tr_banner.configure(text="Đang huấn luyện…", bg="#F59E0B")
        self.tr_note.configure(text="")
        kw = dict(dataset=self.tr_dataset.get(), data=str(data), out=str(out), model=self.tr_model.get(), alpha=alpha, min_df=min_df,
                  test_size=test, seed=seed, stopwords=self.tr_stop.get(), dedup=self.tr_dedup.get(), full=self.tr_full.get())
        self.q = queue.Queue()

        def work(q=self.q):
            try:
                q.put(("done", train.run_training(**kw, log=lambda t: q.put(("log", t)))))
            except Exception as e:
                q.put(("error", f"{type(e).__name__}: {e}"))
        threading.Thread(target=work, daemon=True).start()
        self.after(100, self._poll_train)

    def _poll_train(self):
        finished = False
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self._log(val)
                elif kind == "done":
                    self._on_trained(val)
                    finished = True
                elif kind == "dl_done":
                    self._log("Đã tải xong dữ liệu.")
                    self._busy(False)
                    finished = True
                elif kind == "error":
                    self._log("LỖI: " + val)
                    self._busy(False)
                    self.tr_banner.configure(text="Có lỗi khi chạy", bg="#B91C1C")
                    messagebox.showerror("Lỗi", val)
                    finished = True
        except queue.Empty:
            pass
        if not finished:
            self.after(100, self._poll_train)

    def _on_trained(self, res):
        self._busy(False)
        self.trained = res
        self.trained["dataset"] = self.tr_dataset.get()
        m = res["metrics"]
        self.tr_banner.configure(text=f"Huấn luyện xong · F1 = {vn(m['f1'] * 100)}% · Accuracy = {vn(m['accuracy'] * 100)}%", bg=OK_GREEN)
        rows = [("Accuracy", f"{vn(m['accuracy'] * 100)}%"), ("Precision", f"{vn(m['precision'] * 100)}%"), ("Recall", f"{vn(m['recall'] * 100)}%"),
                ("F1", f"{vn(m['f1'] * 100)}%"), ("ROC-AUC", vn(m["roc_auc"], 4)), ("Thời gian", f"{vn(res['t_fit'])} s (huấn luyện)")]
        for i, (k, v) in enumerate(rows):
            tk.Label(self.metric_box, text=k, bg="#F3F4F6", fg=GREY, font=self.f_small, anchor="w", width=10).grid(row=i, column=0, sticky="w")
            tk.Label(self.metric_box, text=v, bg="#F3F4F6", fg="#111827", font=self.f_bold, anchor="w").grid(row=i, column=1, sticky="w")
        self._draw_cm(m)
        self.tr_note.configure(text=f"{res['n_train']:,} thư huấn luyện · {res['n_test']:,} thư kiểm tra · từ điển {res['vocab']:,} từ · đã lưu {Path(res['out']).name} ({res['size_kb']:.0f} KB)."
                               .replace(",", ".") + " Chỉ số đo trên tập kiểm tra, không dùng khi huấn luyện.")
        self.use_btn.state(["!disabled"])

    def _draw_cm(self, m):
        c, k = self.cm, self.sc
        c.delete("all")
        W, H = int(260 * k), int(170 * k)
        x0, y0, cw, ch = 62 * k, 32 * k, 96 * k, 60 * k
        c.create_text(x0 + cw, 8 * k, text="Dự đoán", font=self.f_small, fill=GREY)
        for j, t in enumerate(("HAM", "SPAM")):
            c.create_text(x0 + cw * (j + 0.5), y0 - 8 * k, text=t, font=self.f_small, fill="#374151")
        for i, t in enumerate(("HAM", "SPAM")):
            c.create_text(x0 - 6 * k, y0 + ch * (i + 0.5), text="Thực tế\n" + t, font=self.f_small, fill="#374151", anchor="e", justify="right")
        cells = [("TN", "#DDEFE3", "tn", 0, 0), ("FP", "#FBD5C2", "fp", 0, 1), ("FN", "#FBD5C2", "fn", 1, 0), ("TP", "#DDEFE3", "tp", 1, 1)]
        for name, col, key, r, cc in cells:
            xa, ya = x0 + cw * cc, y0 + ch * r
            c.create_rectangle(xa + 2, ya + 2, xa + cw - 2, ya + ch - 2, fill=col, outline="")
            c.create_text(xa + cw / 2, ya + ch * 0.4, text=(f"{m[key]:,}".replace(",", ".") if m else "–"), font=self.f_bold, fill="#111827")
            c.create_text(xa + cw / 2, ya + ch * 0.76, text=name, font=self.f_small, fill=GREY)

    def _use_trained(self):
        if not self.trained:
            return
        key = self.trained["dataset"]
        self.clfs[key] = self.trained["clf"]
        MODELS[key] = (MODELS[key][0], Path(self.trained["out"]))
        self.model_key.set(key)
        self._load_model(silent=False)
        self._reset_result()
        self.nb.select(0)
        messagebox.showinfo("Đã nạp mô hình", "Mô hình vừa huấn luyện đã được nạp vào tab “Phân loại”. Hãy chọn một mẫu hoặc dán nội dung để thử.")

    def _build_history(self, parent):
        f = ttk.LabelFrame(parent, text=" Lịch sử phân loại (nhấp đúp để nạp lại nội dung) ", padding=6)
        f.pack(side="bottom", fill="x", padx=4, pady=(0, 8))
        cols = ("time", "model", "res", "p", "text")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=5)
        for c, t, w, a in (("time", "Giờ", 80, "center"), ("model", "Mô hình", 160, "w"), ("res", "Kết quả", 80, "center"),
                           ("p", "P(spam)", 95, "center"), ("text", "Nội dung", 300, "w")):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=int(w * self.sc), anchor=a, stretch=(c == "text"))
        self.tree.tag_configure("SPAM", foreground=SPAM_C)
        self.tree.tag_configure("HAM", foreground=OK_GREEN)
        sb = ttk.Scrollbar(f, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", self._reload_from_history)
        self.hist_text = {}

    # ------------------------------------------------------------------ mô hình
    def _load_model(self, silent=True):
        key = self.model_key.get()
        path = MODELS[key][1]
        if key not in self.clfs:
            try:
                self.clfs[key] = SpamFilter.load(path)
            except Exception as e:
                if not silent:
                    messagebox.showerror("Không nạp được mô hình",
                                         f"Không mở được:\n{path}\n\nLý do: {e}\n\nHãy huấn luyện lại bằng lệnh:\n"
                                         f"python train.py --dataset {key} --data data/... --out models/{path.name}")
                self.model_info.configure(text=f"⚠ Chưa nạp được mô hình: {path.name}", foreground="#B91C1C")
                return None
        clf = self.clfs[key]
        self.model_info.configure(
            text=f"Đã nạp {path.name} · mô hình {clf.model_name} · α = {vn(clf.alpha, 1)} · từ điển {clf.vocabulary_size:,} từ".replace(",", "."),
            foreground=GREY)
        return clf

    def _on_model_change(self):
        self._load_model(silent=False)
        self._reset_result()

    def _pick_model(self):
        p = filedialog.askopenfilename(title="Chọn tệp mô hình (.pkl)", initialdir=BASE / "models", filetypes=[("Mô hình", "*.pkl")])
        if not p:
            return
        try:
            clf = SpamFilter.load(p)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không nạp được mô hình:\n{e}")
            return
        key = self.model_key.get()
        self.clfs[key] = clf
        MODELS[key] = (MODELS[key][0], Path(p))
        self._load_model()
        self._reset_result()

    # ------------------------------------------------------------------ thao tác
    def _on_thr(self, _=None):
        self.thr_label.configure(text=vn(round(self.threshold.get(), 2)))
        if self.last_prob is not None:
            self._show_verdict(self.last_prob)

    def _on_sample(self, _=None):
        i = self.sample_box.current()
        if i <= 0:
            return
        _, txt, key = SAMPLES[i]
        if key and key != self.model_key.get():
            self.model_key.set(key)
            self._load_model(silent=False)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", txt)
        self.classify()

    def _clear(self):
        self.text.delete("1.0", "end")
        self.sample_box.current(0)
        self._reset_result()

    def _open_file(self):
        p = filedialog.askopenfilename(title="Mở tệp văn bản", initialdir=BASE / "samples", filetypes=[("Văn bản", "*.txt"), ("Tất cả", "*.*")])
        if not p:
            return
        self.text.delete("1.0", "end")
        self.text.insert("1.0", Path(p).read_text(encoding="utf-8", errors="replace"))
        self._reset_result()

    def _reload_from_history(self, _):
        sel = self.tree.selection()
        if sel and sel[0] in self.hist_text:
            self.text.delete("1.0", "end")
            self.text.insert("1.0", self.hist_text[sel[0]])
            self._reset_result()

    # ------------------------------------------------------------------ phân loại
    def classify(self):
        txt = self.text.get("1.0", "end").strip()
        if not txt:
            messagebox.showinfo("Chưa có nội dung", "Hãy nhập hoặc dán nội dung thư/tin nhắn cần phân loại.")
            return
        clf = self._load_model(silent=False)
        if clf is None:
            return
        try:
            res = clf.predict_one(txt, 0.5)
            ex = clf.explain(txt, top_k=10 ** 6)
        except Exception as e:
            messagebox.showerror("Lỗi khi phân loại", str(e))
            return
        self.last_prob, self.last_ex = res["spam_prob"], ex
        self.last_ntok = res["n_tokens"]
        self._show_verdict(res["spam_prob"])
        self._highlight(txt, ex)
        self._draw_chart()
        self._show_details(txt, ex)
        self._add_history(txt, res["spam_prob"])

    def _show_verdict(self, p):
        thr = round(self.threshold.get(), 2)
        is_spam = p >= thr
        self.verdict.configure(text="SPAM – THƯ RÁC" if is_spam else "HAM – THƯ TỐT", bg=SPAM_C if is_spam else OK_GREEN)
        self._draw_gauge()

    def _reset_result(self):
        self.last_prob = self.last_ex = None
        self.verdict.configure(text="Chưa phân loại", bg="#9CA3AF")
        self.detail.configure(text="Chọn mô hình, nhập nội dung rồi bấm “Phân loại”.")
        self.warn.configure(text="")
        self.formula.configure(text="")
        self.text.tag_remove("spamw", "1.0", "end")
        self.text.tag_remove("hamw", "1.0", "end")
        self._draw_gauge()
        self._draw_chart()

    def _show_details(self, txt, ex):
        thr = round(self.threshold.get(), 2)
        self.detail.configure(
            text=f"P(spam) = {vn(ex['spam_prob'] * 100, 2)}%   ·   ngưỡng {vn(thr)}   ·   log-odds = {vn(ex['log_odds'], 2, True)}\n"
                 f"Số từ trong nội dung: {self.last_ntok}   ·   nằm trong từ điển: {ex['n_known']} loại   ·   ngoài từ điển (bị bỏ qua): {ex['n_unknown']} loại")
        self.formula.configure(
            text=f"log-odds = tiên nghiệm ({vn(ex['prior_log_odds'], 2, True)}) + tổng đóng góp của các từ "
                 f"({vn(ex['log_odds'] - ex['prior_log_odds'] - ex['bias'], 2, True)})"
                 + (f" + hằng số ({vn(ex['bias'], 2, True)})" if ex["bias"] else "")
                 + ".  Đóng góp dương → đẩy về SPAM, âm → kéo về HAM.")
        warn = ""
        has_vn = bool(re.search(r"[À-ỹ]", txt))
        if ex["n_known"] == 0:
            warn = "⚠ Không có từ nào nằm trong từ điển của mô hình → kết quả không đáng tin (mô hình chỉ học tiếng Anh)."
        elif has_vn:
            warn = "⚠ Nội dung có chữ tiếng Việt có dấu: mô hình chỉ học từ vựng tiếng Anh nên kết quả có thể sai."
        elif ex["n_unknown"] > ex["n_known"] and ex["n_known"] + ex["n_unknown"] >= 6:
            warn = (f"⚠ Phần lớn từ nằm ngoài từ điển ({ex['n_unknown']}/{ex['n_known'] + ex['n_unknown']} loại) → có thể nội dung "
                    "không phải tiếng Anh; kết quả có thể sai.")
        elif ex["n_known"] < 3:
            warn = "⚠ Rất ít từ nằm trong từ điển → bằng chứng yếu, kết quả có thể chưa đáng tin."
        elif self.model_key.get() == "sms" and len(txt) > 600:
            warn = "ℹ Nội dung khá dài; mô hình SMS được huấn luyện trên tin nhắn ngắn. Với email, hãy chọn mô hình Email."
        self.warn.configure(text=warn)

    def _highlight(self, txt, ex):
        self.text.tag_remove("spamw", "1.0", "end")
        self.text.tag_remove("hamw", "1.0", "end")
        sign = {}
        for tok, _, c in ex["top_spam"] + ex["top_ham"]:
            if abs(c) >= 0.5:
                sign[tok] = c
        raw = self.text.get("1.0", "end-1c")
        for m in re.finditer(r"[A-Za-z][A-Za-z'’]*", raw):
            tok = m.group(0).lower().replace("'", "").replace("’", "")
            c = sign.get(tok)
            if c is not None:
                self.text.tag_add("spamw" if c > 0 else "hamw", f"1.0+{m.start()}c", f"1.0+{m.end()}c")

    def _add_history(self, txt, p):
        thr = round(self.threshold.get(), 2)
        name = "SPAM" if p >= thr else "HAM"
        one = " ".join(txt.split())
        iid = self.tree.insert("", 0, values=(datetime.now().strftime("%H:%M:%S"), MODELS[self.model_key.get()][0], name,
                                              vn(p * 100, 2) + "%", one[:140] + ("…" if len(one) > 140 else "")), tags=(name,))
        self.hist_text[iid] = txt

    # ------------------------------------------------------------------ vẽ
    def _draw_gauge(self):
        c, k = self.gauge, self.sc
        c.delete("all")
        w = max(c.winfo_width(), 200)
        x0, x1, y0, y1 = 10 * k, w - 10 * k, 8 * k, 26 * k
        c.create_rectangle(x0, y0, x1, y1, fill="#E5E7EB", outline="")
        thr = round(self.threshold.get(), 2)
        if self.last_prob is not None:
            p = self.last_prob
            col = SPAM_C if p >= thr else OK_GREEN
            c.create_rectangle(x0, y0, x0 + (x1 - x0) * p, y1, fill=col, outline="")
        tx = x0 + (x1 - x0) * thr
        c.create_line(tx, y0 - 5 * k, tx, y1 + 5 * k, fill=NAVY, width=2)
        ty = y1 + 9 * k
        c.create_text(tx, ty, anchor="n", text=f"ngưỡng {vn(thr)}", font=self.f_small, fill=NAVY)
        c.create_text(x0, ty, anchor="nw", text="0%", font=self.f_small, fill=GREY)
        c.create_text(x1, ty, anchor="ne", text="100% (chắc chắn SPAM)", font=self.f_small, fill=GREY)

    def _draw_chart(self):
        c, k = self.chart, self.sc
        c.delete("all")
        w, h = max(c.winfo_width(), 300), max(c.winfo_height(), 160)
        ex = self.last_ex
        if ex is None:
            c.create_text(w / 2, h / 2, text="Biểu đồ sẽ hiện sau khi phân loại", fill=GREY, font=self.f_small)
            return
        sp, hm = ex["top_spam"][:6], ex["top_ham"][:6]
        row_min = 22 * k                                   # mỗi dòng cần đủ cao để chữ không chồng nhau
        top = 34 * k
        max_rows = max(4, int((h - top - 6 * k) / row_min))
        while len(sp) + len(hm) > max_rows:
            (sp if len(sp) >= len(hm) else hm).pop()
        rows = sp + hm
        if not rows:
            c.create_text(w / 2, h / 2, text="Không có từ nào đóng góp đáng kể", fill=GREY, font=self.f_small)
            return
        left, right = 165 * k, w - 62 * k
        x0 = (left + right) / 2
        mx = max(abs(r[2]) for r in rows) or 1
        scale = (right - left) / 2 / mx
        rh = min(30 * k, (h - top - 6 * k) / len(rows))
        c.create_text(left, 14 * k, anchor="w", text="← kéo về HAM", fill=HAM_C, font=self.f_small)
        c.create_text(right, 14 * k, anchor="e", text="đẩy về SPAM →", fill=SPAM_C, font=self.f_small)
        c.create_line(x0, top - 4 * k, x0, top + rh * len(rows), fill="#9CA3AF")
        for i, (tok, n, v) in enumerate(rows):
            y = top + i * rh
            name = SPECIAL.get(tok, tok) + (f" ×{n}" if n > 1 else "")
            cy = y + rh / 2
            c.create_text(8 * k, cy, anchor="w", text=name, font=self.f_mono, fill="#111827")
            x = x0 + v * scale
            col = SPAM_C if v > 0 else HAM_C
            c.create_rectangle(min(x0, x), y + 3 * k, max(x0, x), y + rh - 3 * k, fill=col, outline="")
            c.create_text(x + (6 * k if v > 0 else -6 * k), cy, anchor="w" if v > 0 else "e", text=vn(v, 2, True), font=self.f_small, fill=col)


def main():
    try:  # Windows: chữ nét hơn trên màn hình độ phân giải cao
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()


if __name__ == "__main__":
    main()
