<!-- Bản dịch tiếng Việt; giữ nguyên thuật ngữ kỹ thuật, code, công thức, lệnh và URL. -->

# Nhật ký thay đổi

Có gì mới trong chương trình giảng dạy. Gần đây nhất đầu tiên.

Định dạng theo [Keep a Changelog](https://keepachangelog.com/) một cách lỏng lẻo. Mỗi mục đặt tên cho giai đoạn, bài học và những gì đã thay đổi, vì vậy người học có thể chuyển thẳng đến delta.

## [Chưa phát hành]

### Đã thêm
- `scripts/scaffold-lesson.sh` - giàn giáo tạo ra `phases/NN-phase/NN-lesson/` với cấu trúc thư mục đầy đủ và khung xương `docs/en.md` được lấp đầy sẵn từ `LESSON_TEMPLATE.md`.
- `.github/PULL_REQUEST_TEMPLATE.md` — danh sách kiểm tra của người đóng góp (chạy mã, không có nhận xét mã, xây dựng từ đầu, nguyên tử mỗi bài commit, hàng ROADMAP liên kết đánh dấu).
- `.github/ISSUE_TEMPLATE/bug_report.md` và `new_lesson_proposal.md` - tiếp nhận có cấu trúc cho các báo cáo lỗi và bài học.
- Đây `CHANGELOG.md`.

## 2026-04 - Giai đoạn 4: Hoàn thành Thị giác máy tính

### Đã thêm
- Tất cả 28 bài học Giai đoạn 4, bao gồm các nguyên tắc cơ bản về hình ảnh thông qua tầm nhìn đa phương thức (VLMs, 3D, video, tự giám sát).
- Các hàng giai đoạn 4 trong `ROADMAP.md` được liên kết dưới dạng đánh dấu đến các thư mục bài học, vì vậy trang web hiển thị chúng.

### Cố định
- Giai đoạn 4 precision vượt qua 15+ bài học:
  - `phase-4/02`: Máy tính hình dạng chỉ định xử lý RF/stride cho bể bơi thích ứng, phẳng và tuyến tính.
  - `phase-4/03`: mô tả bộ chọn đường trục liệt kê tất cả các gia đình được bảo hiểm; hướng dẫn đầu được thêm vào cho OCR, y tế, công nghiệp.
  - `phase-4/04`: chẩn đoán phân loại sử dụng ngưỡng định lượng cho mỗi chế độ lỗi; `n/a` khai báo cho các chỉ số không xác định; bảo vệ dưới 3 classes.
  - `phase-4/06`: trình đọc số liệu phát hiện sử dụng `AP@0.5` (không phải `mAP@0.5`); mỗi class recall được tuyên bố là không bắt buộc; Anchor Designer làm rõ việc cắt bớt sải chân và đường dẫn một neo trên mỗi mức.
  - `phase-4/10`: bộ chọn mẫu khai báo `unet_forward_ms` là đầu vào; Người bảo vệ ControlNet được thăng cấp lên quy tắc 0.
  - `phase-4/14`: Thanh tra viên ViT phù hợp với quy tắc từ chối — các nỗ lực chuyển đổi được kiểm toán, không được xác nhận.
  - `phase-4/24`: bộ chọn stack mở từ vựng có mức độ ưu tiên quy tắc rõ ràng và ngữ nghĩa bộ lọc giấy phép; Nhà thiết kế ý tưởng giải quyết step-5/rule-80 xung đột.
  - `phase-4/25`: VLM tài liệu `_merge` nêu lên `ValueError` mô tả về sự không khớp của trình giữ chỗ; CMER bình thường hóa nội bộ.
  - `phase-4/27`: `synthetic_frames` kẹp các hộp GT vào khung H/W.
  - `phase-4/28`: `rope_3d` xác thực phân tách mờ; bỏ `F` import không sử dụng khỏi ví dụ khối DiT.

## 2026-Q1 trở về trước

### Đã thêm
- Giai đoạn 0 (Thiết lập & Dụng cụ): tất cả 12 bài học.
- Giai đoạn 1 (Nền tảng Toán học): tất cả 22 bài học.
- Giai đoạn 2 (ML Nguyên tắc cơ bản): tất cả 18 bài học.
- Giai đoạn 3 (Deep Learning Core): các bài học cốt lõi thông qua perceptron, backprop, optimizers.
- Mã Claude tích hợp skills: `find-your-level` (bài kiểm tra vị trí) và `check-understanding` (bài kiểm tra mỗi giai đoạn).
- Trang web tại `aiengineeringfromscratch.com`: danh mục, trang mỗi bài học, lộ trình, bảng thuật ngữ 277 thuật ngữ.
- Giàn giáo ban đầu cho tất cả 20 giai đoạn (`phases/00-*` đến `phases/19-*`).
- `LESSON_TEMPLATE.md`, `CONTRIBUTING.md`, `ROADMAP.md`, `README.md`.

[Chưa phát hành]: https://github.com/rohitg00/ai-engineering-from-scratch/compare/HEAD...HEAD
