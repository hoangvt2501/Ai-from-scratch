# OpenTelemetry GenAI — Công cụ theo dõi gọi từ đầu đến cuối

> Một agent gọi năm công cụ, ba MCP servers và hai agents phụ. Bạn cần một trace trên tất cả nó. Các quy ước ngữ nghĩa OpenTelemetry GenAI (thuộc tính ổn định trong v1.37 trở lên) là tiêu chuẩn năm 2026, được hỗ trợ nguyên bản bởi Datadog, Langfuse, Arize Phoenix, OpenLLMetry và AgentOps. Bài học này đặt tên cho các thuộc tính bắt buộc, đi bộ phân cấp span (công cụ agent → LLM →) và ships một bộ phát span stdlib mà bạn có thể cắm vào bất kỳ trình xuất OTel nào.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, OTel span emitter)
**Kiến thức tiên quyết:** Giai đoạn 13 · 07 (MCP server), Giai đoạn 13 · 08 (MCP khách hàng)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Đặt tên cho các thuộc tính OTel GenAI cần thiết cho LLM span và span thực thi công cụ.
- Xây dựng hệ thống phân cấp trace bao gồm vòng lặp agent, cuộc gọi LLM, lệnh gọi công cụ và MCP gửi máy khách.
- Quyết định nội dung nào sẽ ghi lại (chọn tham gia) so với biên tập (mặc định).
- Phát spans đến bộ sưu tập cục bộ (Jaeger, Langfuse) mà không cần viết lại mã công cụ.

## Vấn đề

Một bản gỡ lỗi từ tháng 2 năm 2026: người dùng báo cáo "agent của tôi đôi khi mất 30 giây để phản hồi; những lần khác là 3 giây." Không traces. Nhật ký hiển thị cuộc gọi LLM, nhưng không hiển thị công cụ điều động, không phải chuyến đi khứ hồi MCP server, không phải agent phụ. Bạn đoán xem. Cuối cùng bạn sẽ thấy: một MCP server thỉnh thoảng bị treo khi khởi động nguội.

Nếu không có tính năng theo dõi từ đầu đến cuối, bạn không thể tìm thấy điều này. OTel GenAI khắc phục nó.

Các quy ước được giải quyết vào năm 2025-2026 theo nhóm quy ước ngữ nghĩa OpenTelemetry. Họ xác định tên thuộc tính ổn định để Datadog, Langfuse, Phoenix, OpenLLMetry và AgentOps đều phân tích cú pháp giống nhau spans. Dụng cụ một lần; ship đến bất kỳ phần phụ trợ nào.

## Khái niệm

### Span phân cấp

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

Toàn bộ sự việc lồng vào một trace id. Span id liên kết mối quan hệ cha-con.

### Thuộc tính bắt buộc

Theo semconv 2025-2026:

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model` — chuỗi model yêu cầu (ví dụ: `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model` - model thực sự phục vụ.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — ID phản hồi của nhà cung cấp cho mối tương quan.

Đối với spans dụng cụ:

- `gen_ai.tool.name` — mã định danh công cụ.
- `gen_ai.tool.call.id` — id cuộc gọi cụ thể.
- `gen_ai.tool.description` - mô tả công cụ (tùy chọn).

Đối với agent spans:

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### Span loại

- `SpanKind.CLIENT` cho các cuộc gọi vượt qua ranh giới process (nhà cung cấp LLM, MCP server).
- `SpanKind.INTERNAL` cho các bước vòng lặp và thực thi công cụ của riêng agent.

### Chụp nội dung chọn tham gia

Theo mặc định, spans mang các chỉ số và thời gian — không phải prompts hoặc hoàn thành. payloads lớn và PII bị tắt theo mặc định. Đặt `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` và các biến đổi ghi lại nội dung cụ thể để bao gồm nội dung. Review cẩn thận trước khi bật trong sản phẩm.

### Sự kiện trên spans

Bạn có thể thêm sự kiện cấp Token dưới dạng span sự kiện:

- `gen_ai.content.prompt` - tin nhắn đầu vào.
- `gen_ai.content.completion` - tin nhắn đầu ra.
- `gen_ai.content.tool_call` — lệnh gọi công cụ như đã ghi âm.

Thứ tự thời gian sự kiện trong một span để phát lại chi tiết.

### Nhà xuất khẩu

Xuất khẩu spans OTel sang:

- **Jaeger / Tempo.** OSS, tại chỗ.
- **Langfuse.** LLM-observability-cụ thể; trực quan hóa cách sử dụng token.
- **Arize Phoenix.** Evals + truy tìm kết hợp.
- **Datadog.** Thương mại; phân tích cú pháp nguyên bản `gen_ai.*` thuộc tính.
- **Tổ ong.** Định hướng cột; thân thiện với truy vấn.

Tất cả đều nói OTLP, định dạng dây. Mã của bạn không quan tâm.

### Tuyên truyền trên khắp MCP

Khi một ứng dụng MCP gọi một server, hãy chèn tiêu đề theo dõi W3C vào yêu cầu. HTTP có thể phát trực tuyến hỗ trợ tiêu đề tiêu chuẩn. Stdio không mang tiêu đề HTTP nguyên bản; Lộ trình năm 2026 của thông số kỹ thuật thảo luận về việc thêm trường `_meta.traceparent` trên các lệnh gọi JSON-RPC.

Cho đến khi ships đó: bao gồm traceparent trong `_meta` của mọi yêu cầu theo cách thủ công. Server ghi lại id trace.

### Số liệu

Cùng với spans, GenAI semconv xác định các chỉ số:

- `gen_ai.client.token.usage` — biểu đồ.
- `gen_ai.client.operation.duration` — biểu đồ.
- `gen_ai.tool.execution.duration` — biểu đồ.

Sử dụng những thông tin này cho bảng thông tin không cần chi tiết cho mỗi cuộc gọi.

### Lớp AgentOps

AgentOps (thành lập năm 2024) chuyên về GenAI observability. Nó bao bọc các frameworks phổ biến (LangGraph, Pydantic AI, CrewAI) để tự động phát ra spans OTel. Hữu ích nếu stack của bạn sử dụng framework được hỗ trợ; sử dụng thiết bị thủ công nếu không.

## Ứng dụng

`code/main.py` phát ra spans hình OTel thành stdout (ở định dạng giống như OTLP-JSON) cho một agent gọi một LLM, gửi hai công cụ và thực hiện một MCP khứ hồi. Không có nhà xuất khẩu thực sự - bài học tập trung vào span hình dạng và bộ thuộc tính. Dán đầu ra vào trình xem tương thích với OTLP hoặc chỉ cần đọc nó.

Những gì cần xem:

- ID Trace được chia sẻ trên tất cả các spans.
- Liên kết mẹ-con được mã hóa thông qua `parentSpanId`.
- Các thuộc tính `gen_ai.*` bắt buộc được điền.
- Chụp nội dung bị tắt theo mặc định; Một kịch bản bật nó thông qua env var.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-otel-genai-instrumentation.md`. Với một cơ sở mã agent, skill tạo ra một kế hoạch đo lường: nơi thêm spans, thuộc tính nào cần điền và trình xuất nào sẽ nhắm mục tiêu.

## Bài tập

1. Chạy `code/main.py`. Đếm spans và xác định đâu là KHÁCH HÀNG và NỘI BỘ.

2. Bật tính năng chụp nội dung (env var) và xác nhận `gen_ai.content.prompt` và `gen_ai.content.completion` sự kiện xuất hiện. Lưu ý ý nghĩa đối với PII.

3. Thêm `gen_ai.tool.execution.duration` chỉ số thực thi công cụ và phát ra nó dưới dạng mẫu biểu đồ cho mỗi lệnh gọi.

4. Truyền bá traceparent từ agent span mẹ vào trường `_meta.traceparent` của yêu cầu MCP. Xác minh MCP server sẽ thấy cùng một id trace.

5. Đọc thông số kỹ thuật semconv OTel GenAI. Xác định một thuộc tính được liệt kê trong semconv mà mã của bài học này KHÔNG phát ra. Thêm nó.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| OTel | "Đo từ xa mở" | Tiêu chuẩn mở cho traces, chỉ số, nhật ký |
| GenAI semconv | "Quy ước ngữ nghĩa GenAI" | Tên thuộc tính ổn định cho LLM / công cụ / agent spans |
| `gen_ai.*` | "Không gian tên thuộc tính" | Tất cả các thuộc tính GenAI đều có chung tiền tố này |
| Span | "Hoạt động theo thời gian" | Một đơn vị công việc có bắt đầu, kết thúc và thuộc tính |
| Trace | "Tổ tiên span chéo" | Tree of spans chia sẻ id trace |
| SpanKind | "KHÁCH HÀNG / SERVER / NỘI BỘ" | Gợi ý về hướng span |
| OTLP | "Giao thức đường truyền OpenTelemetry" | Định dạng dây cho nhà xuất khẩu |
| Nội dung chọn tham gia | "Prompt / chụp hoàn thành" | Tắt theo mặc định; env var để bật |
| cha mẹ theo dõi | "Tiêu đề W3C" | Truyền bá ngữ cảnh trace trên các dịch vụ |
| Nhà xuất khẩu | "Người gửi hàng dành riêng cho phụ trợ" | Thành phần gửi spans đến Jaeger / Datadog / v.v. |

## Đọc thêm

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — quy ước chuẩn cho spans, số liệu và sự kiện của GenAI
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM và thực thi công cụ span danh sách thuộc tính
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) - `invoke_agent` span cấp agent
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — Nguồn tin cậy được lưu trữ GitHub
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — Hướng dẫn tích hợp production
