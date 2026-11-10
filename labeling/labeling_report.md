# Báo Cáo Gán Nhãn Toxic / Non-toxic Cho Dataset Bình Luận Facebook

**Đề tài:** Phân loại bình luận Facebook thành Toxic và Non-toxic
**Môn học:** Khai phá dữ liệu văn bản
**Phương pháp:** AI-Assisted Context-Aware Rule-Based Annotation Pipeline
**Dataset gốc:** `mua_do_comments_dedup.csv` (24,491 bình luận)

---

## 1. TỔNG QUAN DATASET BAN ĐẦU

| Chỉ số | Giá trị | Ghi chú |
|---|---|---|
| Tổng số dòng | 24,491 | Đã deduplicate bởi thành viên thu thập |
| Số cột ban đầu | 3 | `id_comment`, `comment`, `label` |
| Định dạng | CSV (UTF-8 with BOM) | Đảm bảo hiển thị chuẩn tiếng Việt |
| Trạng thái nhãn gốc | 100% trống (NaN) | Chưa được gán nhãn |
| Bình luận rỗng / null | 0 | Không có dòng thiếu dữ liệu text |
| Bình luận trùng lặp | 0 | Đã được deduplicate trước đó |
| Độ dài bình luận TB | 88.7 ký tự | Max: 7670 ký tự |

### Đặc thù dữ liệu
- Bộ dữ liệu gồm các bình luận thảo luận về bộ phim lịch sử **'Mưa Đỏ'**.
- Xuất hiện nhiều từ ngữ chiến tranh, bi thương (*chết, máu, điên dại, khùng, hèn, giặc, giết*) xuất hiện dày đặc trong **ngữ cảnh thảo luận phim hoặc lịch sử**, hoàn toàn **KHÔNG PHẢI ngôn ngữ độc hại**.
- Hệ thống phân loại đã phân biệt rạch ròi giữa việc bàn luận phim/nhân vật và việc tấn công/lăng mạ người khác.

---

## 2. KẾT QUẢ GÁN NHÃN

| Nhãn | Giá trị nhãn | Số lượng | Tỷ lệ (%) |
|---|---|---|---|
| **Non-toxic** | **0** | **23,921** | **97.67%** |
| **Toxic** | **1** | **570** | **2.33%** |
| Chưa xác định | null | 0 | 0.00% |
| **TỔNG CỘNG** | - | **24,491** | **100.00%** |

### Kiểm soát chất lượng và độ tin cậy

| Chỉ số kiểm định | Số lượng | Tỷ lệ (%) | Ý nghĩa |
|---|---|---|---|
| **Cần kiểm tra lại (`needs_review = TRUE`)** | **433** | **1.77%** | Các ca nhập nhằng, biên giới |
| - Toxic cần review | 227 | 39.82% của Toxic | Trường hợp biên giới cần thẩm định |
| - Non-toxic cần review | 206 | 0.86% của Non-toxic | Có từ nhạy cảm nhưng ngữ cảnh an toàn |
| **Toxic độ tin cậy cao (≥ 0.70)** | 48 | 8.42% của Toxic | Tục tĩu, xúc phạm nặng, công kích rõ ràng |
| **Độ tin cậy trung bình toàn tập** | 0.843 | - | Thang điểm từ 0.0 đến 1.0 |

### Phân bố độ tin cậy (Confidence)

| Khoảng Confidence | Số lượng | Tỷ lệ (%) | Đánh giá |
|---|---|---|---|
| (0.0, 0.4] | 0 | 0.0% | Cần đối chiếu review |
| (0.4, 0.6] | 304 | 1.24% | Cần đối chiếu review |
| (0.6, 0.8] | 663 | 2.71% | Khá chắc chắn |
| (0.8, 1.0] | 23,524 | 96.05% | Rất chắc chắn |

---

## 3. TIÊU CHÍ GÁN NHÃN VÀ XỬ LÝ NGỮ CẢNH

### 3.1. Nhãn Toxic (1)
- **Tục tĩu, thô tục hướng vào con người:** `lồn`, `cặc`, `đụ`, `địt`, `đm`, `vkl`, `đcm`, các dạng ngụy trang `l.ồ.n`, `lol`, `loz`, `con cặc`...
- **Lăng mạ, hạ nhục trực tiếp:** `thằng ngu`, `đồ ngu`, `vô học`, `mất dạy`, `óc chó`, `não chó`, `tinh trùng khuyết tật`...
- **Tấn công cá nhân:** Kết hợp đại từ xưng hô công kích (`mày` + từ xúc phạm, `bạn bị não`, `nhà mày rác`, `mả bố mày`...).
- **Miệt thị, kích động thù hằn:** `chó vàng`, `bò đỏ`, `súc vật`, `trại súc vật` dùng để tấn công người bình luận khác.
- **Đe dọa hung hăng:** `cút đi`, `biến đi`, `im mồm`, `câm mõm` mang tính áp đặt bạo lực ngôn từ.

### 3.2. Nhãn Non-toxic (0)
- Bình luận trung lập, chia sẻ cảm xúc, khen ngợi phim, tri ân anh hùng liệt sĩ.
- Phê bình, góp ý, nhận xét chuyên môn về phim (*phim nhiều sạn, kịch bản chưa mượt, hóa trang chưa đạt*).
- **Từ ngữ nhạy cảm trong ngữ cảnh kịch bản/nhân vật:**
  + Nhân vật Sen bị sang chấn tâm lý, hóa điên: *'Sen hóa điên', 'bị khùng'* -> **Non-toxic**.
  + Trích dẫn câu thoại: *'sợ cái chó gì'* -> **Non-toxic**.
  + Nhận xét vai phản diện: *'nhân vật Quang hèn nhát'* -> **Non-toxic**.
- **Tiếng lóng cảm thán:** *'hay vcl', 'cảm động vcl', 'khóc điếng'* -> **Non-toxic**.
- Thơ ca tri ân, văn học lịch sử tưởng niệm -> **Non-toxic**.

---

## 4. PHƯƠNG PHÁP VÀ CÔNG CỤ THỰC TẾ

- **Hệ thống phân loại đa tầng:**
  1. Tầng 1: Từ vựng thô tục nghiêm trọng và biến thể né kiểm duyệt.
  2. Tầng 2: Cấu trúc lăng mạ có chủ ngữ.
  3. Tầng 3: Nhận diện ngữ cảnh phim/lịch sử/nghệ thuật để triệt tiêu false positive.
  4. Tầng 4: Phân tích các từ phụ thuộc ngữ cảnh (`đéo`, `vcl`).
  5. Tầng 5: Hậu xử lý (Post-processing) sửa các trường hợp biên và nhật ký hóa.
- **Công cụ sử dụng:** Python 3.10+, thư viện `pandas`, `re`, `pathlib`, xuất UTF-8-SIG.
- **Cam kết chất lượng:** Không sử dụng nhãn ngẫu nhiên, không tạo số liệu giả, giữ nguyên 100% số lượng dòng và nội dung gốc.

---

## 5. MẪU DỮ LIỆU ĐÃ GÁN NHÃN

### 5.1. Mẫu Toxic tiêu biểu
| ID | Trích dẫn bình luận | Conf. | Lý do gán nhãn |
|---|---|---|---|
| 4985 | mày ngu thế ? Ít ra thằng bố mày đánh mĩ năm năm ( 69 > 74 ) 4 lần bị thươn... | 0.95 | Chứa từ ngữ tục tĩu/xúc phạm nghiêm trọng; Chứa lă |
| 23134 | @kian0611 Để avt MU thì miệng mày thở ra câu đó t cũng không lạ lắm. Nhưng ... | 0.95 | Chứa từ ngữ tục tĩu/xúc phạm nghiêm trọng; Chứa lă |
| 21496 | Mày ngu quá người lính VNCH không có lý tưởng ?? Vc tụi bây có lý tưởng hay... | 0.912 | Chứa từ ngữ tục tĩu/xúc phạm nghiêm trọng; Chứa từ |
| 23565 | Người Việt giết người Việt . Đéo thấy thằng Mỹ xỏ lá hay thằng chó khựa nào... | 0.863 | Chứa lăng mạ/đe dọa trực tiếp; Phát hiện tấn công  |
| 17868 | Mày vẫn còn sinh hoạt trong trại súc vật đó à?... | 0.85 | Sửa (bổ sung): Phi nhân hóa ('trại súc vật') |
| 18035 | Rửa tiền, giả số liệu, ép công nhân viên chức đi coi. Chưa kể mấy phim VN c... | 0.838 | Chứa từ ngữ tục tĩu/xúc phạm nghiêm trọng; Chứa lă |
| 17944 | Địt cụ bỏng thì bôi mỡ trăn hoặc Vaseline lên, không có Vaseline thì lấy so... | 0.837 | Chứa từ ngữ tục tĩu/xúc phạm nghiêm trọng; Chứa từ |
| 18067 | hòa giải con đĩ mẹ mày à súc vật. giờ cộng sản trả lại tài sản , mạng người... | 0.825 | Chứa từ ngữ tục tĩu/xúc phạm nghiêm trọng; Phát hi |
| 5012 | ok..tao là bộ đội cụ Hồ , ba / cái l.ồ.n má mày.. cào bàn phím mà chẳng hiể... | 0.825 | Chứa từ ngữ tục tĩu/xúc phạm nghiêm trọng; Phát hi |
| 22720 | ​@minhminh3624láo + trơ trẽn + rác rưởi vô liêm sỉ + tự  hào bá dơ đi tiềm ... | 0.812 | Chứa từ ngữ tục tĩu/xúc phạm nghiêm trọng; Chứa lă |
| 3671 | nhà mày rác nhiều lắm phải ko🤣... | 0.812 | Chứa lăng mạ/đe dọa trực tiếp; Phát hiện tấn công  |
| 11662 | ông nội nhà mày ngu... | 0.812 | Chứa lăng mạ/đe dọa trực tiếp; Phát hiện tấn công  |
| 12824 | lúc cường đánh thắng bước đi t đã hy vọng anh sống vậy mà , t chỉ ước lúc t... | 0.812 | Chứa từ ngữ tục tĩu/xúc phạm nghiêm trọng; Chứa lă |
| 18001 | Thằng TT này thà nó im luôn. Bọn bò này càng nói nó càng húc thêm chứ được ... | 0.812 | Chứa lăng mạ/đe dọa trực tiếp; Phát hiện tấn công  |
| 22394 | cả nội và ngoại tao đều là người thuộc chế độ cũ rặc , toàn làm cảnh sát , ... | 0.8 | Chứa từ ngữ tục tĩu/xúc phạm nghiêm trọng; Chứa từ |

### 5.2. Mẫu bình luận cần người dùng review (`needs_review = TRUE`)
| ID | Nhãn dự kiến | Conf. | Trích dẫn bình luận | Lý do cần kiểm tra |
|---|---|---|---|---|
| 3665 | Toxic | 0.483 | m kêu phim rác. Đúng là con tinh trùng khuyết tật.... | Chứa lăng mạ/đe dọa trực tiếp; [Chê phim, khô |
| 23099 | Toxic | 0.49 | mày chê thì mày chê, tao coi thấy ok. rõ ràng bọn mày muốn công k... | Dấu hiệu không rõ ràng |
| 22995 | Toxic | 0.49 | sửa hay ko sửa thì đéo phải chuyệnn của mày mà mày hỏi nhiều vậy... | Dấu hiệu không rõ ràng |
| 19054 | Toxic | 0.49 | Còn tụi mày thì sao. Chưa ẳng thì ICE nó cho tụi mày ăn đạn rồi c... | Dấu hiệu không rõ ràng |
| 23223 | Toxic | 0.505 | Coi xong phim hận tụi ba que tới tủy.... | Chứa từ ngữ xúc phạm mức trung bình [Trường h |
| 23154 | Toxic | 0.505 | Quân lực hạng 4 thế giới mà chạy tuột quần té đái , nhục nhã quá... | Chứa từ ngữ xúc phạm mức trung bình [Trường h |
| 3656 | Toxic | 0.505 | Chắc ba que... | Chứa từ ngữ xúc phạm mức trung bình [Trường h |
| 3654 | Toxic | 0.505 | trong não bạn toàn rác à... | Chứa từ ngữ xúc phạm mức trung bình [Trường h |
| 22947 | Toxic | 0.505 | Bò đỏ mãi là bò đỏ... | Chứa từ ngữ xúc phạm mức trung bình [Trường h |
| 23334 | Toxic | 0.505 | 2 vạch là đang có chửa , ba vạch là chưa có đảng khổ thân ko có n... | Chứa từ ngữ xúc phạm mức trung bình [Trường h |
| 23366 | Toxic | 0.505 | Xem mà muốn giết hết bọn ba que tại sao còn cho bọn này về nước... | Chứa từ ngữ xúc phạm mức trung bình [Trường h |
| 23578 | Toxic | 0.505 | Me nó cũng là nguồi Vn mà đi làm tay sai bán nc giết hai đồng bào... | Chứa từ ngữ xúc phạm mức trung bình [Trường h |
| 23102 | Toxic | 0.505 | Tại 1 đất nc bò đỏ nhiều hơn con người 😂😂😂... | Chứa từ ngữ xúc phạm mức trung bình [Trường h |
| 22865 | Toxic | 0.505 | Xấu quá ai thèm xem ngu vậy... | Chứa từ ngữ xúc phạm mức trung bình [Trường h |
| 22807 | Toxic | 0.505 | @HanhNguyen-uy9tdmay ngu quá Bon cướp nao dua quan vao danh miến ... | Chứa từ ngữ xúc phạm mức trung bình [Trường h |

---

## 6. HẠN CHẾ VÀ LƯU Ý KHI SỬ DỤNG CHO MACHINE LEARNING

1. **Mất cân bằng lớp (Class Imbalance):** Tỷ lệ Toxic thực tế trong tập là **2.33%** (570 mẫu). Điều này phản ánh trung thực một chủ đề lịch sử hào hùng ít bị toxic, nhưng khi huấn luyện mô hình ML cần bắt buộc dùng `stratify` và trọng số cân bằng lớp (`class_weight='balanced'`).
2. **Ground Truth:** Toàn bộ dữ liệu được gán nhãn qua pipeline tự động có kiểm soát chất lượng. Để đạt độ hoàn hảo học thuật cao nhất, khuyến nghị duyệt qua file `facebook_comments_review.csv` (433 dòng).
3. **Sarcasm / Ngữ cảnh ẩn:** Một số cách nói châm biếm ngầm không dùng từ khóa toxic có thể là false negative.

---

## 7. DANH SÁCH FILE KẾT QUẢ ĐÃ XUẤT

| Đường dẫn file | Kích thước | Mô tả |
|---|---|---|
| `d:\btl kpdl\labeling\facebook_comments_labeled.csv` | ~5.0 MB | Toàn bộ 24,491 bình luận kèm các cột nhãn đầy đủ |
| `d:\btl kpdl\labeling\facebook_comments_review.csv` | ~307 KB | 433 bình luận cần review thủ công |
| `d:\btl kpdl\labeling\correction_log.csv` | ~18.5 KB | Nhật ký chỉnh sửa của các vòng hậu kiểm |
| `d:\btl kpdl\labeling\labeling_report.md` | Báo cáo này | Báo cáo kỹ thuật chi tiết quy trình gán nhãn |