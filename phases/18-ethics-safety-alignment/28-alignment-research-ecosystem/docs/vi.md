# Hệ sinh thái nghiên cứu Alignment - MATS, Redwood, Apollo, METR

> Năm tổ chức xác định lớp nghiên cứu alignment phi phòng thí nghiệm năm 2026. MATS (Học giả ML Alignment & Lý thuyết): 527+ nhà nghiên cứu kể từ cuối năm 2021, 180+ bài báo, 10K + trích dẫn, h-index 47; Nhóm thuần tập mùa hè năm 2024 được thành lập thành 501 (c) (3) với ~90 học giả và 40 cố vấn; 80% cựu sinh viên trước năm 2025 làm việc trên safety/security với 200+ tại Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo. Redwood Research: ứng dụng alignment phòng thí nghiệm do Buck Shlegeris thành lập; giới thiệu AI Kiểm soát (Bài 10); hợp tác với AISI của Vương quốc Anh về các trường hợp an toàn kiểm soát. Apollo Research: đánh giá kế hoạch trước khi triển khai cho các phòng thí nghiệm biên giới; tác giả Kế hoạch trong ngữ cảnh (Bài 8) và Hướng tới các trường hợp an toàn cho AI kế hoạch. METR (Model Đánh giá và Nghiên cứu Mối đe dọa): đánh giá khả năng dựa trên nhiệm vụ, nghiên cứu thời gian nhiệm vụ tự chủ; "Các yếu tố chung của Policies an toàn AI biên giới" so sánh frameworks phòng thí nghiệm. Nghiên cứu Eleos AI: Đánh giá trước khi triển khai phúc lợi model (Bài 19); tiến hành đánh giá phúc lợi Claude Opus 4.

**Loại:** Học
**Ngôn ngữ:** none
**Kiến thức tiên quyết:** Giai đoạn 18 · 01-27 (prior Giai đoạn 18 bài học)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Xác định năm tổ chức của hệ sinh thái nghiên cứu alignment phi phòng thí nghiệm và đầu ra cốt lõi của họ.
- Mô tả thang điểm của MATS (học giả, bài báo, chỉ số h) và vai trò của nó như một pipeline tài năng.
- Mô tả chương trình nghị sự Kiểm soát AI của Redwood và quan hệ đối tác với AISI của Vương quốc Anh.
- Mô tả phương pháp đánh giá dựa trên nhiệm vụ của METR.

## Vấn đề

Các phòng thí nghiệm biên giới (Bài 18) đưa ra các đánh giá an toàn trong nội bộ và công bố các kết quả chọn lọc. Hệ sinh thái bên ngoài phòng thí nghiệm là nơi các đánh giá được xác nhận, nơi các chế độ thất bại mới được phát hiện lần đầu tiên và nơi huấn luyện tài năng. Hiểu hệ sinh thái giúp giải thích kết quả nghiên cứu nào được ai tin tưởng.

## Khái niệm

### MATS (Học giả ML Alignment & Lý thuyết)

Bắt đầu vào cuối năm 2021. Chương trình cố vấn nghiên cứu; Các học giả dành 10-12 tuần với một nhà nghiên cứu cao cấp về một vấn đề alignment cụ thể.

Quy mô (2026):
- 527+ nhà nghiên cứu kể từ khi thành lập.
- 180+ bài báo được xuất bản.
- 10K + trích dẫn.
- Chỉ số H 47.
- Mùa hè 2024: 90 học giả + 40 cố vấn; được thành lập như 501 (c) (3).

Kết quả nghề nghiệp: ~80% cựu sinh viên trước năm 2025 đang làm việc trên safety/security. 200+ tại Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo.

### Nghiên cứu Redwood

Áp dụng alignment phòng thí nghiệm. Được thành lập bởi Buck Shlegeris. Giới thiệu chương trình kiểm soát AI (Bài 10). Hợp tác với AISI của Vương quốc Anh về các trường hợp an toàn kiểm soát. Tư vấn cho DeepMind và Anthropic về thiết kế đánh giá.

Các bài báo kinh điển: Greenblatt, Shlegeris và cộng sự, "Kiểm soát AI" (arXiv: 2312.06942, ICML 2024); Alignment Giả mạo (Greenblatt, Denison, Wright và cộng sự, arXiv: 2412.14093, kết hợp với Anthropic).

Phong cách: models mối đe dọa cụ thể, đối thủ trong trường hợp xấu nhất, các giao thức cụ thể có thể được kiểm tra căng thẳng.

### Nghiên cứu Apollo

Đánh giá kế hoạch trước khi triển khai cho các phòng thí nghiệm biên giới. Tác giả Kế hoạch trong ngữ cảnh (Bài 8, arXiv: 2412.04984). Đối tác vào năm 2025 OpenAI hợp tác chống âm mưu training. Sản xuất hướng tới các trường hợp an toàn cho AI kế hoạch (2024).

Phong cách: đánh giá agentic thiết lập nơi lừa dối có thể xuất hiện; phân hủy ba trụ cột (sai lệch, định hướng mục tiêu, nhận thức tình huống).

### METR (Đánh giá Model và Nghiên cứu Mối đe dọa)

Đánh giá năng lực dựa trên nhiệm vụ. Nghiên cứu thời gian hoàn thành nhiệm vụ tự chủ. "Các yếu tố chung của Policies an toàn AI biên giới" (metr.org/common-elements, 2025) so sánh frameworks phòng thí nghiệm.

Đồng tác giả của bản phác thảo trường hợp an toàn AI Scheming với Apollo.

Phong cách: đánh giá nhiệm vụ dài hạn, đo lường khả năng thực nghiệm framework tổng hợp.

### Nghiên cứu AI Eleos

Đánh giá trước khi triển khai phúc lợi Model. Tiến hành đánh giá phúc lợi Claude Opus 4 được ghi lại trong phần 5.3 của thẻ hệ thống. Cung cấp kiểm tra phương pháp bên ngoài cho các yêu cầu liên quan đến phúc lợi của Bài 19.

### Dòng chảy

MATS huấn luyện các nhà nghiên cứu. Sinh viên tốt nghiệp đến Anthropic, DeepMind, OpenAI (nhóm an toàn phòng thí nghiệm) hoặc Redwood, Apollo, METR, Eleos (đánh giá bên ngoài). Các nhà đánh giá bên ngoài hợp tác với các phòng thí nghiệm và với UK AISI / CAISI. Các ấn phẩm cung cấp hệ sinh thái trở lại MATS cho nhóm tiếp theo.

### Tại sao lớp này lại quan trọng

Đánh giá một nguồn không đáng tin cậy: các phòng thí nghiệm đánh giá models của chính họ có xung đột lợi ích về cấu trúc. Các nhà đánh giá bên ngoài có thể nâng cao và xác nhận các chế độ thất bại mà phòng thí nghiệm có thể báo cáo thấp. Bài báo Sleeper Agents năm 2024 (Bài 7) là Anthropic + Redwood; Alignment Faking là Anthropic + Redwood; Âm mưu trong bối cảnh là Apollo; Anti-Scheming là Apollo + OpenAI. Cấu trúc đa tổ chức là kiểm soát chất lượng.

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài 7-11 tham khảo công việc của Redwood và Apollo; Bài 18 tham khảo so sánh framework của METR; Bài 19 tham khảo Eleos. Bài 28 là bản đồ tổ chức rõ ràng cho hệ sinh thái mà rest của Giai đoạn dựa vào.

## Ứng dụng

Không có mã. Đọc "Các yếu tố chung của Policies an toàn AI biên giới" của METR như một ví dụ về cách tổng hợp bên ngoài làm tăng giá trị cho công việc policy bên trong phòng thí nghiệm.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-ecosystem-map.md`. Đưa ra một tuyên bố hoặc đánh giá alignment, nó xác định tổ chức, địa điểm xuất bản và phong cách phương pháp luận, đồng thời kiểm tra chéo các tổ chức đối tác đã biết.

## Bài tập

1. Chọn một bài từ Bài học 7-15 và xác định các tổ chức liên quan. Kiểm tra chéo các tác giả với cựu sinh viên MATS và các liên kết hệ sinh thái hiện tại.

2. Đọc "Các yếu tố chung của Policies an toàn AI biên giới" của METR. Xác định ba sự hội tụ giữa các phòng thí nghiệm mà họ nhấn mạnh và hai sự phân kỳ lớn nhất.

3. Kết quả nghề nghiệp của MATS là ~80% safety/security. Tranh luận xem áp lực lựa chọn này là thích ứng (huấn luyện lĩnh vực này) hay thiên vị (lọc ra các vị trí không đồng nhất).

4. Redwood và Apollo đều làm control/scheming tác phẩm nhưng với phong cách khác nhau. Chọn một chế độ thất bại và mô tả cách mỗi chế độ sẽ điều tra nó.

5. Eleos AI là tổ chức phúc lợi model thuần túy duy nhất. Thiết kế một tổ chức thứ hai giả định tập trung vào một câu hỏi khác liên quan đến phúc lợi (tự do nhận thức, hiện thân robot, v.v.) và trình bày rõ ràng phương pháp luận của nó.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| THẢM | "Chương trình cố vấn" | Học giả ML Alignment & Lý thuyết; 527+ nhà nghiên cứu kể từ năm 2021 |
| Nghiên cứu Redwood | "Phòng thí nghiệm điều khiển" | Áp dụng alignment; AI Kiểm soát tác giả; Đối tác AISI của Vương quốc Anh |
| Nghiên cứu Apollo | "Những võ sĩ mưu mô" | Đánh giá sơ đồ trước khi triển khai cho các phòng thí nghiệm biên giới |
| ĐO LƯỜNG | "Chân trời nhiệm vụ" | Đánh giá năng lực dựa trên nhiệm vụ; framework tổng hợp |
| Eleos AI | "Phòng thí nghiệm phúc lợi" | Đánh giá trước khi triển khai phúc lợi Model |
| Tài năng pipeline | "MATS -> phòng thí nghiệm" | Sinh viên tốt nghiệp MATS chuyển sang Anthropic, DM, OpenAI, Redwood, Apollo, METR |
| Đánh giá bên ngoài | "Kiểm tra không trong phòng thí nghiệm" | Đánh giá không được thực hiện bởi nhà sản xuất model; Thêm độ tin cậy |

## Đọc thêm

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) - chương trình cố vấn
- [Redwood Research](https://www.redwoodresearch.org/) — Giấy kiểm soát AI
- [Apollo Research](https://www.apolloresearch.ai/) - đánh giá âm mưu
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — so sánh framework
- [Eleos AI Research](https://www.eleosai.org/research) - model phương pháp phúc lợi
