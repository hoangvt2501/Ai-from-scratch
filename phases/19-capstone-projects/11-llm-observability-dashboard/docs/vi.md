# Capstone 11 — Bảng điều khiển LLM Observability & Eval

> Langfuse đã mở lõi. Arize Phoenix đã xuất bản bản đồ semconv GenAI năm 2026. Helicone và Braintrust đều tăng gấp đôi phân bổ chi phí cho mỗi người dùng. OpenLLMetry của Traceloop đã trở thành thiết bị đo lường SDK trên thực tế. Hình dạng production là ClickHouse cho traces, Postgres cho siêu dữ liệu, Next.js cho giao diện người dùng và một đội quân nhỏ các công việc đánh giá (DeepEval, RAGAS, LLM-judge) chạy trên các traces được lấy mẫu. Xây dựng một bản tự lưu trữ, nhập từ ít nhất bốn gia đình SDK và chứng minh việc bắt hồi quy được tiêm trong vòng chưa đầy năm phút.

**Loại:** Đá đỉnh
**Ngôn ngữ:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Kiến thức tiên quyết:** Giai đoạn 11 (LLM kỹ thuật), Giai đoạn 13 (công cụ), Giai đoạn 17 (cơ sở hạ tầng), Giai đoạn 18 (an toàn)
**Các giai đoạn thực hiện: **P11 · P13 · P17 · Trang 18
**Thời lượng:** 25 giờ

## Vấn đề

Mỗi đội AI chạy production lưu lượng truy cập vào năm 2026 đều giữ một chiếc máy bay observability bên cạnh model. Phân bổ chi phí. Phát hiện ảo giác. Giám sát trôi dạt. Tín hiệu bẻ khóa. Bảng điều khiển SLO. Cảnh báo rò rỉ PII. Các tham chiếu mã nguồn mở - Langfuse, Phoenix, OpenLLMetry - hội tụ trên các quy ước ngữ nghĩa OpenTelemetry GenAI như schema nhập vào. Giờ đây, bạn có thể đo lường OpenAI, Anthropic, Google, LangChain, LlamaIndex và vLLM với một SDK và spans tương thích ship.

Bạn sẽ xây dựng một bảng thông tin tự lưu trữ thu nạp từ ít nhất bốn họ SDK, chạy một tập hợp nhỏ các tác vụ đánh giá trên traces được lấy mẫu, phát hiện trôi dạt và cảnh báo. Thanh đo lường: với một hồi quy được cố ý tiêm (một prompt bắt đầu tạo PII), bảng điều khiển sẽ bắt nó và kích hoạt cảnh báo trong vòng chưa đầy năm phút.

## Khái niệm

Ingest là OTLP HTTP. SDK sản xuất GenAI-semconv spans: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`. Spans đến ClickHouse để phân tích cột; siêu dữ liệu (người dùng, sessions, ứng dụng) đến Postgres.

Đánh giá chạy dưới dạng batch công việc trên traces được lấy mẫu. DeepEval chấm điểm độ trung thực, độc hại và mức độ liên quan của câu trả lời. RAGAS chấm điểm các chỉ số truy xuất khi trace mang ngữ cảnh truy xuất. Giám khảo LLM tùy chỉnh chạy kiểm tra tên miền cụ thể (rò rỉ PII, phản hồi ngoài policy). Eval chạy viết lại đến cùng một ClickHouse như eval spans liên kết với trace mẹ.

Phát hiện trôi theo dõi embedding phân bố không gian theo thời gian (phân kỳ PSI hoặc KL trên prompt embeddings) cộng với xu hướng điểm đánh giá. Nguồn cấp dữ liệu cảnh báo Prometheus Alertmanager và sau đó là Slack / PagerDuty. Giao diện người dùng là Next.js 15 với Recharts.

## Kiến trúc

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## Stack

- Nhập: Quy ước ngữ nghĩa OpenTelemetry SDKs + GenAI; OTLP HTTP transport
- Bộ sưu tập: OpenTelemetry Collector với bộ xử lý sampling đuôi (để kiểm soát chi phí)
- Lưu trữ: ClickHouse cho spans, Postgres cho siêu dữ liệu, S3 cho lưu trữ sự kiện thô
- Đánh giá: DeepEval, RAGAS 0.2, gói đánh giá Arize Phoenix, thẩm phán LLM tùy chỉnh
- Trôi dạt: PSI / KL trên prompt embeddings gộp (câu-transformers) hàng tuần
- Cảnh báo: Prometheus Alertmanager -> Slack / PagerDuty
- Giao diện người dùng: Next.js 15 Bộ định tuyến ứng dụng + Biểu đồ lại + server hành động
- SDKs hỗ trợ ngay lập tức: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

## Tự xây dựng

1. **Collector config.** OpenTelemetry Collector với bộ thu HTTP OTLP, một bộ lấy mẫu đuôi giữ 100% traces lỗi và 10% thành công, và các nhà xuất sang ClickHouse và S3.

2. **ClickHouse schema.** Bảng `spans` với các cột phản ánh GenAI semconv: `gen_ai_system`, `gen_ai_request_model`, `input_tokens`, `output_tokens`, `latency_ms`, `prompt_hash`, `trace_id`, `parent_span_id`, cộng với túi JSON để payloads lâu. Thêm chỉ mục phụ theo user_id và app_id.

3. **SDK kiểm tra độ bao phủ.** Viết một ứng dụng khách nhỏ bằng cách sử dụng từng SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) với công cụ tự động OpenLLMetry. Xác minh mỗi sản phẩm GenAI chuẩn spans hạ cánh trong ClickHouse.

4. **Công việc đánh giá.** Một công việc đã lên lịch đọc các traces được lấy mẫu trong 15 phút cuối cùng và chạy độ trung thực, độc hại và mức độ liên quan của câu trả lời của DeepEval. Đầu ra được đánh giá spans liên kết với trace mẹ.

5. **Thẩm phán LLM tùy chỉnh.** Thẩm phán rò rỉ PII: khi nhận được phản hồi, hãy gọi LLM bảo vệ để chấm điểm likelihood rò rỉ PII. Câu trả lời có điểm cao sẽ nằm trong hàng đợi phân loại.

6. **Phát hiện trôi dạt.** Công việc hàng tuần tính toán PSI giữa prompt embeddings gộp của tuần này và đường cơ sở 4 tuần sau đó. Nếu PSI vượt quá ngưỡng, hãy cảnh báo.

7. **Bảng điều khiển.** Next.js 15 với các trang: tổng quan (spans/sec, cost/user, độ trễ p95), traces (tìm kiếm + thác nước), đánh giá (xu hướng trung thực, độc hại), trôi dạt (PSI theo thời gian), cảnh báo.

8. **Chuỗi cảnh báo.** Trình xuất Prometheus đọc tổng hợp điểm đánh giá và phần trăm độ trễ; Alertmanager định tuyến đến Slack để nhận cảnh báo và PagerDuty cho các vi phạm nghiêm trọng.

9. **Đầu dò hồi quy.** Chèn lỗi: chatbot được đánh giá bắt đầu rò rỉ SSN giả mạo 1% thời gian. Đo MTTR: từ lỗi được triển khai đến cảnh báo Slack.

## Ứng dụng

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Sản phẩm bàn giao

`outputs/skill-llm-observability.md` là sản phẩm được giao. Với một ứng dụng LLM, bảng điều khiển sẽ nhập traces của nó, chạy đánh giá, cảnh báo khi trôi dạt và bề mặt cost/user sự cố trong Next.js.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | Bảo hiểm Trace-schema | Số SDK họ sản xuất GenAI spans chuẩn (mục tiêu: 6+) |
| 20 | Tính đúng đắn của Eval | Điểm DeepEval / RAGAS so với bộ dán nhãn bằng tay |
| 20 | UX bảng điều khiển | MTTR về hồi quy tiêm (mục tiêu dưới 5 phút) |
| 20 | Chi phí / quy mô | Nhập liên tục ở 1k spans/sec mà không tồn đọng |
| 15 | Cảnh báo + phát hiện trôi dạt | Prometheus/Alertmanager chuỗi được thực hiện từ đầu đến cuối |
| **100** |||

## Bài tập

1. Thêm thiết bị tùy chỉnh cho Haystack framework. Xác minh spans chính tắc trong ClickHouse bằng các thuộc tính `gen_ai.*` trung thực.

2. Hoán đổi DeepEval cho các công cụ đánh giá Phoenix trên cùng một traces. Đo độ lệch điểm giữa hai công cụ đánh giá.

3. Làm sắc nét trình phát hiện trôi: tính toán PSI trên mỗi app-id thay vì trên toàn cầu. Hiển thị các vệt trôi trên mỗi ứng dụng.

4. Thêm trang "tác động đến người dùng": chi phí trên mỗi người dùng và tỷ lệ thất bại trên mỗi người dùng bằng đường viền lấp lánh.

5. Xây dựng một sampling policy đuôi giữ 100% traces có độc tính > 0,5 cộng với 10% mẫu phân tầng của rest. Biện pháp sampling bias giới thiệu.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| GenAI semconv | "Thuộc tính LLM OTel" | Thông số kỹ thuật OpenTelemetry 2025 cho các thuộc tính LLM span (hệ thống, model, tokens) |
| Đuôi sampling | "Mẫu sau trace" | Collector quyết định giữ hoặc thả một trace sau khi nó hoàn thành (có thể nhìn trộm lỗi) |
| PSI | "Chỉ số ổn định dân số" | Số liệu trôi so sánh hai phân phối; > 0,2 thường báo hiệu sự trôi dạt có ý nghĩa |
| LLM thẩm phán | "Đánh giá như model" | Một LLM chấm điểm đầu ra của một LLM khác trên một bảng đánh giá (độ trung thực, độc hại, PII) |
| Đuôi-sampling policy | "Giữ quy tắc" | Quy tắc quyết định traces nào nên kiên trì so với bỏ cuộc; lỗi + tốc độ lấy mẫu |
| Đánh giá span | "trace đánh giá được liên kết" | Trẻ em span mang theo điểm đánh giá được liên kết với span cuộc gọi LLM ban đầu |
| Chi phí cho mỗi người dùng | "Kinh tế đơn vị" | Chi phí đô la do user_id qua một cửa sổ; Chỉ số sản phẩm chính |

## Đọc thêm

- [Langfuse](https://github.com/langfuse/langfuse) — nền tảng observability lõi mở tham chiếu
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — tham chiếu thay thế với hỗ trợ drift mạnh mẽ
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — thiết bị đo lường tự động SDK họ
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - schema nhập
- [Helicone](https://www.helicone.ai) — observability lưu trữ thay thế
- [Braintrust](https://www.braintrust.dev) — nền tảng thay thế ưu tiên đánh giá
- [ClickHouse documentation](https://clickhouse.com/docs) — cửa hàng span cột
- [DeepEval](https://github.com/confident-ai/deepeval) — thư viện đánh giá
