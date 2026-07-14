# Quy ước ngữ nghĩa OpenTelemetry GenAI

> GenAI SIG của OpenTelemetry (ra mắt tháng 4 năm 2024) xác định schema tiêu chuẩn cho agent telemetry. Span tên, thuộc tính và quy tắc thu thập nội dung hội tụ giữa các nhà cung cấp nên agent traces có cùng ý nghĩa trong Datadog, Grafana, Jaeger và Honeycomb.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 13 (LangGraph), Giai đoạn 14 · 24 (Observability nền tảng)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Đặt tên cho GenAI span danh mục: model/client, agent, công cụ.
- Phân biệt spans `invoke_agent` CLIENT và INTERNAL và khi nào áp dụng.
- Liệt kê các thuộc tính GenAI cấp cao nhất: tên nhà cung cấp, model yêu cầu, ID nguồn dữ liệu.
- Giải thích hợp đồng thu thập nội dung: chọn tham gia, `OTEL_SEMCONV_STABILITY_OPT_IN`, đề xuất tham chiếu bên ngoài.

## Vấn đề

Mỗi nhà cung cấp đều phát minh ra tên span của riêng họ. Các nhóm vận hành cuối cùng sẽ xây dựng bảng điều khiển cho mỗi framework. GenAI SIG của OpenTelemetry khắc phục điều này bằng cách xác định một tiêu chuẩn mà toàn bộ hệ sinh thái nhắm mục tiêu.

## Khái niệm

### Span danh mục

1. **Model / spans khách hàng.** Bao gồm các cuộc gọi LLM thô. Được phát ra bởi nhà cung cấp SDKs (Anthropic, OpenAI, Bedrock) và bộ điều hợp framework model.
2. **Agent spans.** `create_agent` (khi agent được xây dựng) và `invoke_agent` (khi nó chạy).
3. **Công cụ spans.** Một công cụ cho mỗi lần gọi công cụ; kết nối với agent span bằng mối quan hệ cha mẹ - con cái.

### Agent span đặt tên

- Span tên: `invoke_agent {gen_ai.agent.name}` nếu được đặt tên; dự phòng về `invoke_agent`.
- Span loại:
  - **CLIENT** — dành cho các dịch vụ agent từ xa (OpenAI Assistants API, Bedrock Agents).
  - **INTERNAL** — dành cho in-process agent frameworks (LangChain, CrewAI, local ReAct).

### Các thuộc tính chính

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model` — ID model.
- `gen_ai.response.model` — model đã giải quyết (có thể khác với yêu cầu do định tuyến).
- `gen_ai.agent.name` — mã định danh agent.
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id` — cho RAG: kho dữ liệu hoặc cửa hàng nào đã được tham khảo.

Các quy ước dành riêng cho công nghệ tồn tại cho Anthropic, Azure AI Inference, AWS Bedrock, OpenAI.

### Chụp nội dung

Quy tắc mặc định: thiết bị đo lường KHÔNG NÊN ghi lại inputs/outputs theo mặc định. Capture được chọn tham gia qua:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

Mẫu production được đề xuất: lưu trữ nội dung bên ngoài (S3, kho nhật ký của bạn), ghi lại tài liệu tham khảo trên spans (ID con trỏ, không phải văn xuôi). Đây là biện pháp phòng thủ đầu độc nội dung của Bài học 27 được kết nối với observability.

### Tính ổn định

Hầu hết các hội nghị đều được thử nghiệm kể từ tháng 3 năm 2026. Chọn tham gia bản xem trước ổn định với:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ ánh xạ các thuộc tính GenAI nguyên bản vào LLM Observability schema của nó. Các phần phụ trợ khác (Grafana, Honeycomb, Jaeger) hỗ trợ các thuộc tính thô.

### Mô hình này sai ở đâu

- **Nắm bắt toàn bộ prompts trong spans.** PII, bí mật, dữ liệu khách hàng trong traces mà các hoạt động có thể đọc. Lưu trữ bên ngoài.
- **Không `gen_ai.provider.name`.** Bảng điều khiển nhiều nhà cung cấp bị hỏng khi thiếu phân bổ.
- **Spans không có liên kết mẹ.** Công cụ mồ côi spans. Luôn truyền bá ngữ cảnh.
- **Không đặt chọn tham gia ổn định.** Các thuộc tính của bạn có thể được đổi tên khi nâng cấp phụ trợ.

## Tự xây dựng

`code/main.py` triển khai bộ phát span stdlib phù hợp với các quy ước GenAI:

- `Span` với thuộc tính GenAI schema.
- `Tracer` với các ngữ cảnh lồng nhau `start_span`.
- Chạy agent theo kịch bản phát ra: `create_agent`, `invoke_agent` (NỘI BỘ), spans trên mỗi công cụ `chat` spans cho các lệnh gọi LLM.
- Chế độ chụp nội dung lưu trữ prompts bên ngoài và ghi lại ID trên spans.

Chạy nó:

```
python3 code/main.py
```

Đầu ra: cây span với tất cả các thuộc tính GenAI bắt buộc và "cửa hàng bên ngoài" hiển thị các tham chiếu nội dung chọn tham gia.

## Ứng dụng

- **Datadog LLM Observability** (v1.37+) ánh xạ các thuộc tính nguyên bản.
- **Langfuse / Phoenix / Opik** (Bài 24) — tự động thiết lập hệ sinh thái.
- **Jaeger / Honeycomb / Grafana Tempo **- traces OTel thô; xây dựng bảng điều khiển từ các thuộc tính GenAI.
- **Tự lưu trữ** — chạy OTel Collector với bộ xử lý GenAI.

## Sản phẩm bàn giao

`outputs/skill-otel-genai.md` kết nối OTel GenAI spans vào một agent hiện có với các mặc định chụp nội dung và lưu trữ tham chiếu bên ngoài.

## Bài tập

1. Thiết bị vòng lặp ReAct Bài học 01 của bạn với `invoke_agent` (NỘI BỘ) + spans cho mỗi công cụ. Gửi đến một phiên bản Jaeger.
2. Thêm tính năng thu thập nội dung ở chế độ "chỉ tham chiếu": prompts vào SQLite span các thuộc tính chỉ mang ID hàng.
3. Đọc thông số kỹ thuật để biết `gen_ai.data_source.id`. Nối nó vào tìm kiếm Bài học 09 Mem0 của bạn.
4. Đặt `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` và xác minh các thuộc tính của bạn không bị bộ sưu tập đổi tên.
5. Xây dựng bảng điều khiển: "lỗi công cụ nào tương quan với models nào" chỉ từ các thuộc tính GenAI.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| GenAI SIG | "Nhóm OpenTelemetry GenAI" | Nhóm làm việc OTel xác định schema |
| invoke_agent | "Agent span" | Tên của span đại diện cho một lần chạy agent |
| span KHÁCH HÀNG | "Cuộc gọi từ xa" | Span gọi đến dịch vụ agent từ xa |
| span NỘI BỘ | "Trong process" | Span để chạy trong process agent |
| gen_ai.provider.name | "Nhà cung cấp" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "Nguồn RAG" | Cái nào corpus/store một lần truy xuất |
| Chụp nội dung | "Prompt ghi nhật ký" | Chọn tham gia ghi lại tin nhắn; Lưu trữ bên ngoài trong sản phẩm |
| Chọn tham gia ổn định | "Chế độ xem trước" | Env var để ghim các quy ước thực nghiệm |

## Đọc thêm

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — thông số kỹ thuật
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) - GenAI spans theo mặc định
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — OTel spans tích hợp sẵn
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace truyền ngữ cảnh
