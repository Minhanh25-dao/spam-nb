# Bộ lọc thư rác Naïve Bayes (Đề tài 42 – Nhóm 9, môn Trí tuệ nhân tạo)

Chương trình phân loại thư điện tử / tin nhắn SMS thành **HAM** hoặc **SPAM** bằng Naïve Bayes
(Multinomial và Bernoulli) do nhóm **tự cài đặt** bằng NumPy/SciPy (scikit-learn chỉ dùng để đối chứng).

## Cài đặt
```bash
pip install -r requirements.txt
python download_data.py          # tải Enron-Spam và SMS Spam Collection vào thư mục data/
```

## Giao diện đồ họa
```bash
python gui.py          # Windows: bấm đúp chay_giao_dien.bat
```
Có hai tab:
- **Phân loại:** chọn mô hình (SMS hoặc Email) → dán nội dung → **Phân loại** (Ctrl+Enter). Hiện nhãn SPAM/HAM, P(spam), thanh
  ngưỡng, biểu đồ các từ ảnh hưởng nhất, tô màu các từ trong nội dung và lưu lịch sử.
- **Huấn luyện:** chọn bộ dữ liệu (hoặc bấm "Tải dữ liệu mẫu từ Internet" nếu chưa có), chỉnh tham số (α, min_df, tỉ lệ tập kiểm
  tra, seed...), bấm "Bắt đầu huấn luyện". Chạy nền nên cửa sổ không bị treo; log và ma trận nhầm lẫn hiện trực tiếp. Bấm
  "Dùng mô hình này để phân loại →" để chuyển ngay mô hình vừa huấn luyện sang tab Phân loại.

Dùng Tkinter (có sẵn trong Python), không cần cài thêm.

## Huấn luyện và đánh giá
```bash
python train.py --dataset enron --data data/enron_spam_data.csv --out models/enron_mnb.pkl
python train.py --dataset sms   --data data/sms.tsv             --out models/sms_mnb.pkl
```

## Phân loại thư mới
```bash
python predict.py --model models/sms_mnb.pkl --text "Congratulations! You won 1000 pounds. Call now" --explain
python predict.py --model models/enron_mnb.pkl --file samples/spam_email_1.txt samples/ham_email_1.txt --explain
python predict.py --model models/sms_mnb.pkl --lines samples/sms_samples.txt
python predict.py --model models/sms_mnb.pkl --text "..." --threshold 0.99    # nâng ngưỡng để giảm chặn nhầm
python predict.py --model models/sms_mnb.pkl                                  # chế độ tương tác
```

## Kiểm thử và tái lập kết quả báo cáo
```bash
python -m unittest discover -s tests -v   # 23 kiểm thử đơn vị
python experiments.py                     # toàn bộ thí nghiệm -> results/results.json
python experiments.py --stages protocol   # đánh giá theo chi phí: CV trên tập huấn luyện, ngưỡng theo chi phí, đo một lần, kiểm tra gần-trùng
python make_figures.py                    # vẽ lại hình (các hình có sẵn trong figures/, bước này không bắt buộc)
```

## Cấu trúc
- `spamfilter/`: gói thư viện (tiền xử lý, túi từ, Naïve Bayes, chỉ số đánh giá, dữ liệu, pipeline)
- `tests/`, `samples/`, `models/`, `results/`, `figures/`
- `report/`: mã dựng báo cáo Word (python-docx); cần tệp mẫu bìa `TTNT.docx` (sửa đường dẫn TEMPLATE trong `report/build_report.py`)

## Hạn chế chính
Chỉ hỗ trợ tiếng Anh; mô hình huấn luyện ở miền dữ liệu này giảm mạnh khi áp dụng sang miền khác (xem mục 2.9.13 của báo cáo).

## Lưu ý khi chạy `make_figures.py` (tuỳ chọn)
- Các sơ đồ khối cần **Graphviz** (lệnh `dot`). Windows: `winget install Graphviz.Graphviz`, rồi mở lại terminal. Nếu chưa cài, chương trình chỉ báo "BỎ QUA" và giữ nguyên hình có sẵn.
- Phông chữ được chọn tự động theo máy (Liberation Serif / Times New Roman / DejaVu Serif), không còn cảnh báo `findfont`. Hình vẽ trên Windows có thể khác nhẹ về kiểu chữ so với hình trong báo cáo.
