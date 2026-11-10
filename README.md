# Tiền xử lý bình luận Facebook tiếng Việt

Python 3.10 trở lên. Không gán nhãn, huấn luyện, trích xuất đặc trưng hoặc chia Train/Test.

```powershell
python -m pip install -r requirements.txt
python -X utf8 preprocess.py
python -X utf8 -m unittest -v test_preprocess.py
```

Trong workspace hiện tại, thư viện bổ sung được cài ở `.python_deps`; chương trình tự nạp thư mục này. Với máy mới chỉ cần cài requirements. Có thể cài cục bộ bằng `python -m pip install --target .python_deps -r requirements.txt`.

Đầu vào mặc định là `labeling/facebook_comments_labeled.csv`: 24.491 dòng, đã đối chiếu toàn bộ ID/nội dung với `mua_do_comments_dedup.csv`. Các nhãn và metadata có sẵn được giữ nguyên. Để dùng bản gốc hoặc file CSV/XLSX/JSON khác:

```powershell
python -X utf8 preprocess.py --input mua_do_comments_dedup.csv
python -X utf8 preprocess.py --input du_lieu.xlsx --text-column comment --output-dir ket_qua
python -X utf8 preprocess.py --no-stopwords --output-dir without_stopwords
```

JSON hỗ trợ mảng bản ghi hoặc đối tượng chứa khóa `data`/`comments`. CSV thử UTF-8-SIG, UTF-8 rồi CP1258 và báo encoding; không bỏ dòng lỗi. XLSX đọc sheet đầu tiên. ID được lấy từ `id` hoặc `id_comment`; nếu không có thì tạo chỉ số dòng bắt đầu từ 1, báo cáo đầu vào sẽ cho biết cấu trúc. Nhãn `label` giữ nguyên nếu có; tên nhãn khác vẫn giữ nguyên trong metadata.

Pipeline: NFC → HTML/entities/URL → chữ thường và ký tự vô hình/khoảng trắng → teencode → Underthesea → stopwords có bảo vệ. Không đọc nhãn khi biến đổi văn bản. `k`, `bt`, `vs` được giữ nguyên vì đa nghĩa. Emoji và dấu câu được giữ; lỗi Underthesea làm dừng chương trình kèm nội dung gây lỗi.

Đầu ra UTF-8-SIG: `facebook_comments_preprocessed.csv`, `preprocessing_comparison.csv`. Báo cáo `preprocessing_report.md` gồm số liệu thật và 20 ví dụ. CSV chính giữ toàn bộ cột gốc và thêm các cột từng bước. Chạy lại sẽ thay đầu ra trước đó; không ghi đè đầu vào.

`teencode_dictionary.json` quản lý thay thế rõ nghĩa. `vietnamese_stopwords.txt` là tập con 12 mục được chọn thủ công từ [Vietnamese stopwords](https://github.com/stopwords/vietnamese-stopwords/blob/master/vietnamese-stopwords.txt), lưu bản nguồn tại `stopwords_source.txt` và thông tin giấy phép MIT trong `STOPWORDS_SOURCE_README.md`. Danh sách nhỏ ưu tiên giữ sắc thái; từ phủ định, đại từ, từ công kích được bảo vệ kể cả trong token ghép. Có thể tắt lọc để so sánh sau này.

Kiểm thử dùng các chuỗi kiểm tra yêu cầu và dữ liệu thật để kiểm tra NFC, HTML, URL, teencode, từ được bảo vệ, emoji, tách từ, rỗng, định dạng đầu vào và toàn vẹn từng ô gốc. Chuỗi kiểm thử không được dùng làm dataset đầu ra.
