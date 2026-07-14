# Batch APIs - Giảm giá 50% theo tiêu chuẩn công nghiệp

> Mọi nhà cung cấp lớn đều ships một batch API không đồng bộ với chiết khấu 50% và quay vòng ~24 giờ. OpenAI, Anthropic, Google và hầu hết các nền tảng inference (Fireworks batch tier, Together batch) triển khai cùng một mẫu. Stack batch với bộ nhớ đệm prompt và pipelines qua đêm giảm xuống ~10% chi phí không lưu bộ nhớ đệm đồng bộ. Quy tắc rất đơn giản: nếu nó không tương tác, nó thuộc về batch. pipelines tạo nội dung, phân loại tài liệu, trích xuất dữ liệu, tạo báo cáo, ghi nhãn hàng loạt, gắn thẻ danh mục — bất kỳ thứ gì chịu được độ trễ 24 giờ đều là tiền còn lại trên bàn cho đến khi chuyển sang batch. Mô hình production 2026 là phân loại mọi khối lượng công việc LLM mới thành ba làn: tương tác (đồng bộ với bộ nhớ đệm), bán tương tác (hàng đợi không đồng bộ với dự phòng), batch (qua đêm, đầu vào được lưu trong bộ nhớ đệm xếp chồng lên nhau). Khối lượng công việc giả vờ tương tác nhưng chịu được nhiều phút lãng phí độ trễ nhất.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy batch-vs-sync cost simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 14 (Prompt & Bộ nhớ đệm ngữ nghĩa)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Kể tên ba nhà cung cấp batch APIs (OpenAI, Anthropic, Google) và giảm giá 50% phổ biến + đảm bảo quay vòng 24 giờ.
- Tính toán chi phí cho việc xếp chồng batch + đầu vào được lưu trong bộ nhớ đệm trên khối lượng công việc phân loại qua đêm và so sánh với đường cơ sở không được lưu trong bộ nhớ đệm đồng bộ.
- Phân loại khối lượng công việc thành tương tác / bán tương tác / batch và biện minh cho làn đường.
- Đặt tên cho hai cái bẫy: tương tác một phần (người dùng mong đợi nhanh hơn 24h) và trôi schema đầu ra (batch định dạng tệp khác nhau tùy theo nhà cung cấp).

## Vấn đề

Nhóm của bạn ships một pipeline tạo báo cáo hàng đêm. 50.000 tài liệu, tóm tắt từng tài liệu, phân nhóm các bản tóm tắt, soạn thảo một bản tóm tắt điều hành. Chạy đồng bộ mất 4 giờ với giá 2.000/night. Bạn nghe nói về batch APIs.

Thuộc tính batch giúp bạn giảm giá 50%. Bạn cũng bật prompt bộ nhớ đệm trên system prompt (được chia sẻ trên tất cả 50 nghìn cuộc gọi). Xếp chồng lên nhau, hóa đơn giảm xuống $180/night - ~9% đường cơ sở. Giống nhau pipeline, ba config thay đổi.

Batch là đòn bẩy rẻ nhất trong bộ công cụ chi phí LLM mà không ai kéo. Lý do chủ yếu là tổ chức: các nhóm suy nghĩ "thời gian thực" trong khi SLA thực sự là "vào buổi sáng". Bài học này là về việc không để lại 90% hóa đơn trên bàn.

## Khái niệm

### Ba batch APIs

**OpenAI Batch API**: JSONL tải lên tệp với danh sách các yêu cầu. Hứa hẹn quay vòng 24 giờ (thường là ~2-8 giờ trong thực tế). Giảm giá 50% cho tokens đầu vào và đầu ra. `/v1/batches` endpoint. Các đầu vào đủ điều kiện trong bộ nhớ cache cũng được đặt giá đầu vào trong bộ nhớ đệm ở trên cùng.

**Anthropic Tin nhắn Batches**: JSONL tải lên. Quay vòng 24 giờ. Giảm giá 50%. Hỗ trợ `cache_control` - ghi bộ nhớ cache là rõ ràng, việc đọc diễn ra tự động trong batch.

**Dự đoán AI Batch đỉnh của Google**: Đầu vào BigQuery hoặc GCS. Giảm giá 50% tương tự cho Gemini. Tích hợp với Vertex pipelines.

### Ngữ nghĩa: không đồng bộ, không chậm

Batch là "Tôi hứa sẽ trở lại trong vòng 24 giờ" - không phải "quá trình này sẽ mất 24 giờ". P50 điển hình là 2-6 giờ. Nhà cung cấp lên lịch batch của bạn trong windows thấp điểm khi hàng tồn kho GPU không được sử dụng hết.

### Stack với bộ nhớ đệm

Tóm tắt 50k tài liệu với cùng 4K-token system prompt:

- Đồng bộ không được lưu vào bộ nhớ cache: 50000 × (đầu ra $input × 4000 + $ × 200) ở tốc độ đầy đủ.
- Bộ nhớ đệm đồng bộ: system prompt được lưu vào bộ nhớ đệm sau lần ghi đầu tiên; 49999 còn lại nhận được đầu vào rẻ hơn 10 lần.
- Batch bộ nhớ cache: tất cả những điều trên cộng với giảm giá 50% cho cả đọc và ghi.

stack: batch + bộ nhớ cache = ~10% hóa đơn không được đồng bộ hóa tệ. Bất kỳ khối lượng công việc nào chạy qua đêm và có system prompt dùng chung đều nên sử dụng công việc này.

### Phân loại khối lượng công việc

**Tương tác** — người dùng chờ phản hồi. TTFT rất quan trọng. Cuộc gọi đồng bộ với bộ nhớ đệm prompt. Không thể batch.

**Bán tương tác** — người dùng gửi tác vụ, kiểm tra lại sau vài phút. Hàng đợi không đồng bộ với dự phòng để đồng bộ hóa nếu batch không khả dụng. Hãy nghĩ đến việc lập chỉ mục volume RAG vừa phải.

**Batch** - Người dùng mong đợi kết quả "vào buổi sáng" hoặc "giờ tiếp theo". pipelines nội dung, phân loại trên quy mô lớn, phân tích ngoại tuyến. Luôn batch, luôn stack bộ nhớ đệm.

Sai lầm thường gặp: phân loại mọi thứ là tương tác vì pipeline production. Production không phải là thông số kỹ thuật về độ trễ - SLA là như vậy.

### Bẫy tương tác một phần

Một số features trông tương tác nhưng chịu được 5-10 phút. Ví dụ: báo cáo sức khỏe khách hàng hàng đêm với nút "làm mới". Làm mới nhấp chuột của người dùng; Chờ 10 phút là được. Nhóm ships nó đồng bộ. 50 lần làm mới đồng thời có giá gấp 10 lần chi phí hàng loạt và gửi qua email.

Câu hỏi cần hỏi: "24 giờ có ý nghĩa gì đối với người dùng này?" Nếu câu trả lời là "họ sẽ không nhận ra", hãy batch nó.

### Bẫy schema đầu ra

Batch định dạng tệp khác nhau tùy theo nhà cung cấp:

- OpenAI: JSONL, một yêu cầu mỗi dòng.
- Anthropic: JSONL, một tin nhắn mỗi dòng; định dạng phản hồi được nhúng.
- Đỉnh: Bảng BigQuery hoặc tiền tố GCS với TFRecord.

Viết "một batch khách hàng" giữa các nhà cung cấp có nghĩa là mã bộ điều hợp cho mỗi nhà cung cấp. Gateways quảng cáo batch đa nhà cung cấp (Portkey, LiteLLM một số cấp) vẫn bao bọc mỏng định dạng thô.

### Những con số bạn nên nhớ

- Batch chiết khấu giữa các nhà cung cấp: cố định 50% cho đầu vào + đầu ra.
- SLA quay vòng: Đảm bảo 24 giờ, 2-6 giờ P50 điển hình.
- batch xếp chồng + đầu vào được lưu trong bộ nhớ đệm: ~10% chi phí đồng bộ hóa không được lưu trong bộ nhớ cache.
- Quy tắc phân loại khối lượng công việc: nếu độ trễ 24h chấp nhận được, luôn batch.

## Ứng dụng

`code/main.py` tính toán chi phí trên đồng bộ hóa, đồng bộ + bộ nhớ đệm, batch và batch + bộ nhớ đệm cho khối lượng công việc 50 nghìn tài liệu. Báo cáo tiết kiệm bằng $ và phần trăm.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-batch-triager.md`. Với đặc điểm khối lượng công việc, phân loại thành interactive/semi/batch và ước tính tiết kiệm.

## Bài tập

1. Chạy `code/main.py`. Đối với pipeline 100 nghìn tài liệu với đầu ra 3K token system prompt và 500 token, hãy tính toán mức tiết kiệm của stack đầy đủ (batch + bộ nhớ đệm) so với đường cơ sở đồng bộ hóa.
2. Chọn ba features trong một sản phẩm thực sự mà bạn biết. Phân loại từng interactive/semi/batch.
3. Một người dùng phàn nàn rằng báo cáo của họ mất 3 giờ. Đó là một batch phân loại sai hay một tương tác hợp pháp? Viết tiêu chí quyết định.
4. SLA trả batch API của bạn là 24 giờ nhưng P99 là 20 giờ. Làm thế nào để bạn truyền đạt điều này cho người dùng - hành vi của hệ thống xuôi dòng trên trường hợp biên là gì?
5. Tính toán hòa vốn: ở độ dài tiền tố dùng chung nào thì batch + bộ nhớ đệm trở nên rẻ hơn so với chạy qua đêm trên GPU dành riêng của bạn?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Batch API | "Giảm giá không đồng bộ" | Giảm giá 50% với thời gian quay vòng 24h |
| JSONL | "Định dạng batch" | Một yêu cầu JSON cho mỗi dòng; Tiêu chuẩn OpenAI/Anthropic |
| Tin nhắn Batches | "Anthropic batch" | Tên sản phẩm batch API của Anthropic |
| Dự đoán Batch | "Đỉnh batch" | Sản phẩm batch API của Vertex AI |
| SLA quay vòng | "Lời hứa 24h" | Đảm bảo, không điển hình; điển hình là 2-6h |
| Phân loại khối lượng công việc | "Quyết định tương tác" | Quyết định định tuyến tương tác / bán / batch |
| Đầu ra schema | "Định dạng phản hồi" | Bố cục JSONL cho mỗi nhà cung cấp; không di động |
| Giảm giá xếp chồng | "batch + bộ nhớ đệm" | ~10% hóa đơn đồng bộ hóa không được lưu trong bộ nhớ cache khi cả hai đều áp dụng |

## Đọc thêm

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — định dạng JSONL và ngữ nghĩa `/v1/batches`.
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch định dạng và tương tác `cache_control`.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini batch ngữ nghĩa.
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
