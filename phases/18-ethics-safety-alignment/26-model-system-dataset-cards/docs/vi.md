# Thẻ Model, Hệ thống và Dataset

> Ba định dạng tài liệu cấu trúc AI tính minh bạch. Thẻ Model (Mitchell et al. 2019) — nhãn dinh dưỡng cho models: dữ liệu training, phân tích phân tách định lượng, cân nhắc đạo đức, cảnh báo; chỉ 0.3% thẻ Hugging Face model ghi lại những cân nhắc về đạo đức (Oreamuno et al. 2023). Bảng dữ liệu cho Datasets (Gebru et al. 2018, CACM) - động lực, thành phần, process thu thập, ghi nhãn, phân phối, bảo trì; tương tự bảng dữ liệu điện tử. Thẻ dữ liệu (Pushkarna và cộng sự, Google 2022) — chi tiết phân lớp mô-đun (kính thiên văn, kính tiềm vọng, kính hiển vi) làm đối tượng ranh giới cho người đọc đa dạng. Phát triển 2024-2025: tạo tự động qua LLMs (CardGen, Liu et al. 2024); Chi tiết thẻ model tương quan với mức tăng tải xuống lên đến 29% trên HF (Liang et al. 2024); chứng thực có thể xác minh (Laminator, Duddu et al. 2024); bổ sung báo cáo bền vững cho carbon/water (Jouneaux et al. Tháng 7 năm 2025); EU/ISO thẻ quy định xuất hiện. Thẻ hệ thống (Sidhpurwala 2024; Minh bạch cấp hệ thống meta; "Blueprints of Trust" arXiv:2509.20394) — tài liệu hệ thống AI đầu cuối bao gồm các khả năng bảo mật, bảo vệ tiêm prompt, phát hiện lấy cắp dữ liệu alignment với các giá trị con người.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, model-card + datasheet + system-card generator)
**Kiến thức tiên quyết:** Giai đoạn 18 · 18 (frameworks an toàn), Giai đoạn 18 · 24 (quy định)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Mô tả thẻ model Mitchell et al. 2019 ban đầu và bảng dữ liệu Gebru et al. 2018.
- Mô tả phân lớp telescopic/periscopic/microscopic của Thẻ dữ liệu.
- Mô tả Thẻ hệ thống và phạm vi phủ sóng đầu cuối của chúng.
- Nêu ba phát triển 2024-2025 (tạo tự động, chứng thực có thể xác minh, báo cáo bền vững).

## Vấn đề

frameworks quy định (Bài 24) và policies an toàn phòng thí nghiệm (Bài 18) đều yêu cầu tài liệu. Các định dạng tài liệu đã phát triển từ model cụ thể (thẻ model) sang dataset cụ thể (bảng dữ liệu) đến hệ thống cụ thể (thẻ hệ thống). Mỗi đề cập đến một phạm vi minh bạch khác nhau. Công việc tự động hóa và chứng thực có thể xác minh 2024-2025 giải quyết vấn đề áp dụng lâu dài.

## Khái niệm

### Thẻ Model (Mitchell et al. 2019)

Các phần:
- Model chi tiết.
- Mục đích sử dụng.
- Các yếu tố (các yếu tố nhân khẩu học hoặc môi trường có liên quan để đánh giá).
- Số liệu.
- Dữ liệu đánh giá.
- Training dữ liệu.
- Phân tích định lượng (phân tách theo các yếu tố).
- Cân nhắc đạo đức.
- Cảnh báo và khuyến nghị.

Vấn đề áp dụng: Oreamuno et al. Kiểm tra thẻ Hugging Face model năm 2023 cho thấy chỉ có 0,3% tài liệu cân nhắc về đạo đức.

### Bảng dữ liệu cho Datasets (Gebru et al. 2018)

Tương tự bảng dữ liệu điện tử. Các phần:
- Động lực (tại sao dataset được tạo ra).
- Thành phần (có gì trong đó).
- Bộ sưu tập process (nó được lắp ráp như thế nào).
- Ghi nhãn (nếu có).
- Sử dụng (dự định, bị cấm, rủi ro).
- Phân phối.
- Bảo trì.

Được xuất bản trong CACM 2021. Biểu dữ liệu là tài liệu ngược dòng; Thẻ model phụ thuộc vào độ chính xác của biểu dữ liệu.

### Thẻ dữ liệu (Pushkarna và cộng sự, Google 2022)

Chi tiết phân lớp mô-đun. Ba mức thu phóng:
- **Kính thiên văn.** Tóm tắt cấp cao cho những người không phải là chuyên gia.
- **Periscopic.** Tổng quan cấp trung cho các học viên ML.
- **Kính hiển vi.** Tài liệu chi tiết cấp feature dành cho kiểm toán viên.

Khung đối tượng ranh giới: các độc giả khác nhau trích xuất thông tin khác nhau từ cùng một tài liệu.

### Thẻ hệ thống

Phạm vi: hệ thống AI end-to-end bao gồm model + an toàn stack + bối cảnh triển khai. Các phần thường bao gồm:
- Khả năng bảo mật.
- Bảo vệ Prompt tiêm.
- Phát hiện lấy cắp dữ liệu.
- Alignment với các giá trị nhân văn đã nêu.
- Ứng phó sự cố.

Sidhpurwala 2024 và minh bạch cấp hệ thống Meta hoạt động. "Blueprints of Trust" (arXiv:2509.20394) chính thức hóa System Card như là lớp triển khai bổ sung cho Model Cards.

### Diễn biến 2024-2025

- **CardGen (Liu et al. 2024).** Tạo thẻ model tự động qua LLMs; báo cáo tính khách quan cao hơn nhiều thẻ do con người tạo ra trên các trường Mitchell 2019 được tiêu chuẩn hóa.
- **Tương quan tải xuống (Liang et al. 2024).** Thẻ model chi tiết tương quan với tỷ lệ tải xuống cao hơn tới 29% trên HF — áp lực áp dụng hiện do thị trường định hướng, không chỉ dựa trên tuân thủ.
- **Laminator (Duddu et al. 2024).** Chứng thực có thể xác minh thông qua TEE phần cứng / chữ ký mật mã - cho phép thẻ model mang bằng chứng yêu cầu bồi thường, không chỉ là yêu cầu bồi thường.
- **Tính bền vững (Jouneaux et al. Tháng 7 năm 2025).** Bổ sung cho dấu chân carbon, nước và năng lượng điện toán; các tiêu chuẩn ISO mới nổi.
- **Thẻ quy định.** Đạo luật AI EU (Bài 24) Chương Minh bạch về Quy tắc Thực hành GPAI yêu cầu thẻ model làm artifact tuân thủ.

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài 24-25 là các lớp quy định và CVE. Bài 26 là lớp tài liệu. Bài 27 là quản trị dữ liệu training, là ngược dòng của biểu dữ liệu. Bài 28 là hệ sinh thái nghiên cứu tạo ra các đánh giá được tham chiếu trong thẻ.

## Ứng dụng

`code/main.py` tạo ra một thẻ model, bảng dữ liệu và thẻ hệ thống tối thiểu để triển khai đồ chơi. Mỗi loại tuân theo cấu trúc phần chuẩn. Bạn có thể kiểm tra định dạng và so sánh ba phạm vi.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-card-audit.md`. Với thẻ model, bảng dữ liệu hoặc thẻ hệ thống, nó sẽ kiểm tra phạm vi của phần, phân tách số và liệu có chứng thực có thể xác minh hay không.

## Bài tập

1. Chạy `code/main.py`. Kiểm tra các thẻ đã tạo. Xác định các phần yếu (chỉ giữ chỗ) và chỉ rõ bằng chứng nào sẽ củng cố chúng.

2. Mở rộng thẻ model với phân tích phân tách định lượng trên hai nhóm nhân khẩu học (Bài 20).

3. Đọc Oreamuno et al. 2023 về tỷ lệ chấp nhận 0.3%. Đề xuất một thay đổi cấu trúc đối với đặc điểm kỹ thuật thẻ model sẽ tăng cường việc áp dụng cân nhắc đạo đức.

4. Laminator (Duddu et al. 2024) sử dụng TEE để chứng thực có thể xác minh. Thiết kế một trường thẻ model mang chứng thực mật mã của kết quả đánh giá và mô tả vai trò của người xác minh.

5. Viết Thẻ hệ thống (Thẻ hệ thống, không phải Thẻ Model) cho một trong các dự án trước đây của bạn hoặc triển khai giả định. Xác định phần có giá trị cao nhất cho kiểm toán viên bên thứ ba.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Thẻ Model | "lá bài Mitchell" | Mitchell et al. Tài liệu tiêu chuẩn năm 2019 cho ML models |
| Bảng dữ liệu | "bảng dữ liệu Gebru" | Gebru et al. Tài liệu tiêu chuẩn năm 2018 cho datasets |
| Thẻ dữ liệu | "lá bài Pushkarna" | Tài liệu về dữ liệu phân lớp mô-đun của Google 2022 |
| Thẻ hệ thống | "Thẻ triển khai" | Tài liệu hệ thống AI đầu cuối bao gồm stack an toàn |
| Đối tượng ranh giới | "độc giả khác nhau, một tài liệu" | Đóng khung thẻ dữ liệu: cùng một tài liệu phục vụ nhiều đối tượng khác nhau |
| Chứng thực có thể xác minh | "Chứng thực máy cán màng" | Bằng chứng mật mã hoặc TEE đính kèm với yêu cầu tài liệu |
| Lĩnh vực bền vững | "Dấu chân carbon / nước" | Bổ sung năm 2025 mới nổi cho kế toán môi trường |

## Đọc thêm

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — thẻ model chính tắc
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — giấy bảng dữ liệu
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — tài liệu dữ liệu phân lớp
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) - Chính thức hóa thẻ hệ thống
