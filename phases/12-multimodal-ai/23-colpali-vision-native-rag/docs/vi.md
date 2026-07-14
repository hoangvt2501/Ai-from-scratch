# ColPali và Vision-Native Document RAG

> RAG truyền thống phân tích cú pháp PDF thành văn bản, chia thành các khối, nhúng các khối, lưu trữ vectors. Mỗi bước đều mất tín hiệu: OCR bỏ dữ liệu biểu đồ, chia nhỏ các hàng bảng, văn bản embeddings bỏ qua các số liệu. ColPali (Faysse và cộng sự, tháng 7 năm 2024) đã đặt câu hỏi đơn giản hơn: tại sao lại trích xuất văn bản? Nhúng hình ảnh trang trực tiếp qua PaliGemma, sử dụng tương tác muộn kiểu ColBERT để truy xuất và giữ tất cả bố cục, số liệu, phông chữ và tín hiệu định dạng mà tài liệu mang theo. Xuất bản benchmarks: accuracy đầu cuối tốt hơn 20-40% so với RAG văn bản trên các tài liệu giàu hình ảnh. ColQwen2, ColSmol và VisRAG đã mở rộng mô hình. Bài học này đọc luận án RAG gốc tầm nhìn và xây dựng một trình lập chỉ mục nhỏ giống như ColPali.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Kiến thức tiên quyết:** Giai đoạn 11 (Kỹ thuật LLM - RAG kiến thức cơ bản), Giai đoạn 12 · 05 (LLaVA)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Giải thích sự khác biệt giữa truy xuất hai encoder (một vector cho mỗi tài liệu) và truy xuất tương tác muộn (nhiều vectors cho mỗi tài liệu).
- Mô tả hoạt động MaxSim của ColBERT và cách ColPali khái quát hóa nó từ tokens văn bản sang các bản vá hình ảnh.
- Xây dựng một trình lập chỉ mục nhỏ giống như ColPali: trang → bản vá embeddings → MaxSim trên các trang embeddings → top-k thuật ngữ truy vấn.
- So sánh trình tạo ColPali + Qwen2.5-VL so với RAG văn bản + GPT-4 trên trường hợp sử dụng hóa đơn / báo cáo tài chính.

## Vấn đề

Text-RAG trên PDF vứt bỏ hầu hết tài liệu. Tăng trưởng doanh thu quý 3 của báo cáo tài chính thường nằm trong biểu đồ; những phát hiện của báo cáo y tế được ghi chú bằng hình ảnh; Khối chữ ký của hợp đồng pháp lý là một sự kiện bố cục, không phải là một sự kiện văn bản.

Văn bản-RAG pipeline:

1. PDF → văn bản qua OCR / pdftotext.
2. Nhắn tin → 300-500 token đoạn.
3. Chunk → bi-encoder embedding (một vector).
4. Truy vấn của người dùng → embedding → sự tương đồng cosin → top-k các khối.
5. Chunks + → LLM truy vấn.

Năm bước mất mát. Biểu đồ không được chụp. Các bảng bị chia thành từng khối. Bố cục nhiều cột phẳng. Chú thích hình biến mất.

Cách khắc phục của ColPali: bỏ qua OCR, nhúng trực tiếp hình ảnh trang. Sử dụng tương tác muộn kiểu ColBERT để truy xuất để model có thể tham gia vào các bản vá chi tiết tại thời điểm truy vấn.

## Khái niệm

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832) là một phương pháp truy xuất văn bản. Thay vì một vector cho mỗi tài liệu, nó tạo ra một vector mỗi token. Tại thời điểm truy vấn:

- Truy vấn tokens có embeddings riêng (N_q vectors).
- Tài liệu tokens nhận được embeddings (N_d vectors, thường được lưu trong bộ nhớ đệm).
- Điểm = tổng trên truy vấn tokens của tối đa trên tài liệu tokens của sự tương đồng cosin: Σ_i max_j cos(q_i, d_j).

Đây là hoạt động MaxSim. Mỗi truy vấn token "chọn" token tài liệu phù hợp nhất của nó. Điểm cuối cùng là tổng.

Ưu điểm: recall mạnh, xử lý ngữ nghĩa cấp thuật ngữ. Nhược điểm: N_d vectors cho mỗi tài liệu, lưu trữ đắt tiền.

### ColPali

ColPali (Faysse et al., arXiv:2407.01449) áp dụng mẫu ColBERT cho hình ảnh.

- Mỗi trang được mã hóa bởi PaliGemma (ViT + ngôn ngữ) thành bản vá embeddings: N_p vectors mỗi trang.
- Mỗi truy vấn của người dùng (văn bản) được mã hóa thành truy vấn token embeddings: N_q vectors.
- Điểm = Σ_i max_j cos(q_i, p_j), tức là MaxSim qua các bản vá truy vấn-text-tokens và page-image-patches.
- Truy xuất top-k trang theo tổng điểm.

Tại thời điểm nhập tài liệu: nhúng mọi trang với PaliGemma, lưu trữ tất cả các bản vá embeddings. Tại thời điểm truy vấn: nhúng tokens truy vấn, tính toán MaxSim so với tất cả các embeddings trang được lưu trữ, trả về top-k trang.

Ưu điểm: end-to-end đánh bại RAG văn bản từ 20-40% trên các tài liệu có hình ảnh phong phú. Mỗi bản vá vector ghi lại bố cục và nội dung cục bộ.

Nhược điểm: Các bản vá N_p × nổi 4 byte × vectors D-dim trên mỗi trang = dung lượng lưu trữ tăng nhanh. Giảm thiểu bằng quantization PQ / OPQ.

### ColQwen2 và ColSmol

ColQwen2 (illuin-tech, 2024-2025) hoán đổi PaliGemma lấy Qwen2-VL. encoder cơ sở tốt hơn, truy xuất tốt hơn.

ColSmol là biến thể quy mô nhỏ hơn để sử dụng cục bộ / biên. Một săn ColSmol ở ~1 tỷ tham số chạy trên GPU tiêu dùng.

### VisRAG

VisRAG (Yu et al., arXiv:2410.10594) là một biến thể khác: thay vì MaxSim trên các bản vá, hãy gộp mỗi trang thành một vector duy nhất với một VLM sau đó truy xuất hai encoder. Lập chỉ mục nhanh hơn + dung lượng lưu trữ nhỏ hơn, recall yếu hơn.

Sự đánh đổi giữa chất lượng và chi phí: ColPali về chất lượng, VisRAG về quy mô.

### M3DocRAG

M3DocRAG (Cho et al., arXiv:2411.04952) mở rộng truy xuất đa phương thức sang suy luận nhiều tài liệu nhiều trang. Truy xuất các trang trên các tài liệu, soạn ngữ cảnh nhiều trang cho VLM.

### ViDoRe — the benchmark

Người bạn đồng hành của ColPali benchmark. Đánh giá truy xuất tài liệu trực quan. Nhiệm vụ bao gồm báo cáo tài chính, tài liệu khoa học, tài liệu hành chính, hồ sơ y tế, sổ tay hướng dẫn. Số liệu: nDCG@5.

ColPali-v1 đạt điểm ~80% nDCG@5 trên ViDoRe; RAG văn bản trên cùng một tài liệu đạt điểm ~50-60%.

### RAG pipeline đầu cuối

Đối với RAG gốc tầm nhìn:

1. Nhập: Hình ảnh trang → PDF → mã hóa PaliGemma → lưu trữ tất cả các embeddings bản vá.
2. Truy vấn: văn bản người dùng → MaxSim token embeddings → truy vấn đối với tất cả các trang → top-k trang được lập chỉ mục.
3. Tạo: top-k hình ảnh trang + → VLM truy vấn (Qwen2.5-VL hoặc Claude) → câu trả lời.

Không OCR ở bất cứ đâu. Số liệu, biểu đồ, phông chữ, bố cục đều chảy vào câu trả lời.

### Toán học lưu trữ

Báo cáo tài chính dài 50 trang với 729 bản vá mỗi trang và 128 embeddings mờ:

- ColPali: 50 * 729 * 128 * 4 byte = ~18 MB thô, ~4 MB sau PQ.
- Text-RAG: 50 đoạn * 768-mờ * 4 byte = ~150 kB.

ColPali có dung lượng lưu trữ nhiều hơn ~30 lần cho mỗi tài liệu. Ở quy mô lớn, OPQ / PQ giảm xuống ~5-10 lần, thường có thể chấp nhận được.

### Khi RAG văn bản vẫn thắng

- Tài liệu văn bản thuần túy không có tín hiệu bố cục (bài viết wiki, nhật ký trò chuyện). Text-RAG đơn giản hơn và lưu trữ rẻ hơn.
- Kho lưu trữ hàng triệu trang trong đó dung lượng lưu trữ chiếm ưu thế về chi phí.
- Các yêu cầu quy định nghiêm ngặt yêu cầu văn bản OCR có thể trích xuất cùng với việc truy xuất.

Đối với mọi thứ khác vào năm 2026 - báo cáo tài chính, bài báo khoa học, hợp đồng pháp lý, hồ sơ y tế, tài liệu UX - RAG gốc tầm nhìn sẽ chiến thắng.

## Ứng dụng

`code/main.py`:

- Bản vá đồ chơi encoder: ánh xạ một "trang" (lưới nhỏ gồm feature vectors) với một mảng embeddings bản vá.
- MaxSim scorer: tính điểm kiểu ColBERT giữa một truy vấn token embedding bộ và một bộ bản vá trang.
- Lập chỉ mục 5 trang đồ chơi, chạy 3 truy vấn, trả về top-k với điểm số.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-vision-rag-designer.md`. Cho một dự án RAG tài liệu, hãy chọn ColPali / ColQwen2 / VisRAG / text-RAG và kích thước lưu trữ.

## Bài tập

1. Báo cáo thường niên dài 200 trang với 729 bản vá mỗi trang, emb 128 mờ, thả nổi 4 byte. Tính toán lưu trữ thô và lưu trữ nén PQ (8x).

2. MaxSim là Σ_i max_j cos(q_i, p_j). Tổng này nắm bắt được điều gì mà một sự tương đồng trung bình đơn giản không có?

3. ColPali lập chỉ mục các trang dưới dạng bộ bản vá. Điều gì sẽ thay đổi nếu thay vào đó chúng ta lập chỉ mục ở cấp độ từ (như ColBERT làm)? Đánh đổi?

4. Thiết kế pipeline đầu cuối cho kho dữ liệu 1 triệu trang với ngân sách độ trễ là 500 mili giây cho mỗi truy vấn. Chọn ColQwen2 / VisRAG và biện minh.

5. Đọc M3DocRAG (arXiv: 2411.04952). Mô tả mẫu attention nhiều trang và sự khác biệt của nó so với truy xuất ColPali một trang.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Tương tác muộn | "Phong cách ColBERT" | Truy xuất bằng cách sử dụng embeddings mỗi token hoặc mỗi bản vá + MaxSim, không phải một tài liệu duy nhất vector |
| Tối đa Sim | "Max-over-patches" | Đối với mỗi token truy vấn, hãy chọn tài liệu có độ tương tự cao nhất token; tổng trên truy vấn |
| Bi-encoder | "Single-vector" | Một vector cho mỗi tài liệu; nhanh hơn nhưng mất độ chi tiết |
| Đa vector | "Nhiều vectors cho mỗi bác sĩ" | Lưu trữ N_p vectors trên mỗi tài liệu / trang; Chi phí lưu trữ tăng nhưng recall cải thiện |
| Bản vá embedding | "Trang feature" | Một vector cho mỗi bản vá hình ảnh từ một VLM encoder, được lưu vào bộ nhớ đệm trên mỗi trang |
| ViDoRe | "Băng ghế tài liệu tầm nhìn" | Bộ benchmark của ColPali để truy xuất tài liệu trực quan |
| PQ quantization | "quantization sản phẩm" | Nén duy trì sự tương đồng vector trong khi thu nhỏ dung lượng lưu trữ ~8x |

## Đọc thêm

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
