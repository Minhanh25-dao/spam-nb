"""Tải hai bộ dữ liệu công khai dùng trong đề tài về thư mục data/.

  * SMS Spam Collection  (Almeida et al., 2011) - 5.574 tin nhắn
  * Enron-Spam           (Metsis et al., 2006)  - 33.716 thư điện tử (bản CSV gộp
    của MWiechmann/enron_spam_data trên GitHub)

Cách dùng:  python download_data.py
"""
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SOURCES = {
    "sms.tsv": "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    "enron_spam_data.zip": "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip",
}


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in SOURCES.items():
        dest = DATA_DIR / name
        if dest.exists():
            print(f"[bỏ qua] {name} đã tồn tại")
            continue
        print(f"[tải] {url}")
        urllib.request.urlretrieve(url, dest)
        if name.endswith(".zip"):
            with zipfile.ZipFile(dest) as z:
                z.extractall(DATA_DIR)
    print("Hoàn tất. Các file trong", DATA_DIR)


if __name__ == "__main__":
    main()
