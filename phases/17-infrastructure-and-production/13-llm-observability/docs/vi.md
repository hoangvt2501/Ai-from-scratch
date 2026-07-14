# LLM Observability Stack Lựa chọn

> Năm 2026 observability Thị trường được chia thành hai loại. Nền tảng phát triển (LangSmith, Langfuse, Comet Opik) giám sát gói với các đánh giá, prompt quản lý, session phát lại. Gateway/instrumentation các công cụ (Helicone, SigNoz, OpenLLMetry, Phoenix) tập trung vào telemetry. Langfuse là lõi được MIT cấp phép với cân bằng OSS mạnh mẽ (50K events/month Miễn phí cloud). Phoenix là OpenTelemetry theo Giấy phép Elastic 2.0 - tuyệt vời cho drift/RAG trực quan, không phải là một sự bền bỉ production Phần phụ trợ. Arize AX sử dụng zero-copy Iceberg/Parquet Tích hợp tuyên bố rẻ hơn 100 lần so với nguyên khối observability. LangSmith dẫn đầu cho LangChain/LangGraph, $39/user/mo, chỉ tự lưu trữ trong Enterprise. Helicone là proxy-dựa trên thiết lập 15-30 phút, 100K req/mo miễn phí, nhưng ít chiều sâu hơn agent traces. Phổ biến production Mẫu: Gateway (Helicone/Portkey) + Nền tảng đánh giá (Phoenix/TruLens) được dán bởi OpenTelemetry.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy trace-sampling simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 08 (Inference Metrics), Giai đoạn 14 (Agent Engineering)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Phân biệt các nền tảng phát triển (đi kèm: đánh giá + prompts + sessions) với các công cụ gateway/telemetry (chỉ traces + chỉ số).
- Ánh xạ sáu công cụ chính (Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik) với các trường hợp sử dụng cấp phép, giá cả và điểm ngọt ngào của chúng.
- Giải thích mẫu keo OpenTelemetry cho phép bạn kết hợp một công cụ gateway với một nền tảng đánh giá riêng biệt.
- Đặt tên cho bộ phân biệt chi phí năm 2026 (phương pháp tiếp cận không sao chép của Arize AX so với nhập nguyên khối) và nêu hệ số nhân 100 lần thô.

## Vấn đề

Bạn shipped một LLM feature. Nó hoạt động. Bạn không có khả năng hiển thị các lỗi prompt, vòng lặp công cụ, hồi quy độ trễ, chi phí tăng đột biến hoặc tỷ lệ truy cập bộ nhớ đệm prompt. Bạn "LLM observability" trên Google và nhận được tám công cụ, tất cả đều tuyên bố rằng chúng giải quyết cùng một vấn đề ở ba mức giá khác nhau.

Họ không giải quyết cùng một vấn đề. LangSmith trả lời "tại sao LangGraph này lại thất bại?" Phoenix trả lời "RAG pipeline của tôi có trôi không?" Helicone trả lời "ứng dụng nào đang ghi tokens?" Langfuse trả lời "tôi có thể tự tổ chức toàn bộ không?" Các công cụ khác nhau, đối tượng khác nhau.

Chọn bao gồm bốn trục: stack (LangChain? raw SDK? multi-vendor?), dung sai giấy phép (chỉ MIT? Đàn hồi OK? Phạt thương mại?), Ngân sách (bậc miễn phí? $100/mo? $ 1000/mo?), và tự lưu trữ (phải? tốt để có? không bao giờ?).

## Khái niệm

### Hai loại

**Nền tảng phát triển** kết hợp observability với đánh giá, quản lý prompt, dataset phiên bản session phát lại. Bạn chạy thử nghiệm, xem prompt nào hiệu quả, hồi quy dataset một prompt mới so với những người chiến thắng cũ. LangSmith, Langfuse, Sao chổi Opik.

**Gateway/telemetry công cụ** công cụ inference cuộc gọi — prompt, phản hồi, tokens, độ trễ, model, chi phí. Helicone, SigNoz, OpenLLMetry, Phượng hoàng. Tối giản. Có thể được kết hợp với một công cụ đánh giá riêng biệt thông qua OpenTelemetry.

### Langfuse - Cân OSS

- Core Apache / MIT được cấp phép; tự lưu trữ qua Docker.
- Cloud bậc miễn phí: 50K events/month. Thanh toán: $29/mo cho đội.
- Đánh giá, quản lý prompt, traces, datasets. Phạm vi hợp lý của cả bốn features nền tảng phát triển.
- Điểm ngọt ngào: bạn muốn LangSmith-class features nhưng phải tự lưu trữ hoặc tiếp tục sử dụng giấy phép OSS.

### Phoenix (Arize) — telemetry đầu tiên, OpenTelemetry gốc

- Giấy phép đàn hồi 2.0; tự lưu trữ tầm thường.
- Xuất sắc trong việc RAG và trực quan hóa trôi dạt. Các biểu đồ phân tán không gian Embedding shipped là class đầu tiên.
- Không được thiết kế như backend production liên tục - chủ yếu là observability thời gian phát triển.
- Điểm ngọt ngào: phát triển RAG pipeline, gỡ lỗi trôi dạt, ghép nối với một gateway riêng cho production.

### Arize AX - vở kịch quy mô

- Thương mại. Tích hợp hồ dữ liệu không sao chép qua Iceberg/Parquet.
- Tuyên bố rẻ hơn ~100 lần so với observability nguyên khối (Datadog-class) trên quy mô lớn. Phép toán: bạn lưu trữ traces trong Parquet của riêng mình trên S3; Arize đọc trực tiếp.
- Điểm hấp dẫn: >10 triệu traces/day, hồ dữ liệu hiện có, muốn có bảng điều khiển dành riêng cho LLM mà không cần định giá Datadog.

### LangSmith - LangChain/LangGraph đầu tiên

- Thương mại, $39/user/month. Chỉ tự lưu trữ trên Enterprise.
- Tốt nhất trong class cho LangChain và LangGraph stacks. Nếu bạn không tham gia, nó sẽ kém hấp dẫn hơn.
- Điểm ngọt ngào: đội ngũ cam kết với LangChain, sẵn sàng trả tiền.

### Helicone — khả thi tối thiểu dựa trên proxy

- Thiết lập 15-30 phút bằng cách hoán đổi `OPENAI_API_BASE` của bạn sang proxy Helicone.
- MIT được cấp phép; 100 nghìn req/mo miễn phí, trả phí $20/mo+.
- Bao gồm failover, bộ nhớ đệm, rate limits - cũng hoạt động như một gateway.
- Ít độ sâu hơn trên traces agent / nhiều bước.
- Điểm ngọt ngào: khởi động nhanh, ứng dụng stack đơn, cần gateway + observability trong một.

### Opik (Comet) — Nền tảng phát triển OSS

- Apache 2.0, hoàn toàn OSS.
- Tương tự feature đặt cho Langfuse với di sản sao chổi.
- Điểm ngọt ngào: ML đội đã có mặt trên Sao chổi, muốn LLM observability trong cùng một ngăn.

### SigNoz — APM đầy đủ đầu tiên của OpenTelemetry

- Apache 2.0. Xử lý APM chung cộng với LLM thông qua OpenTelemetry.
- Điểm ngọt ngào: observability thống nhất giữa các dịch vụ và cuộc gọi LLM.

### Chất keo: Quy ước ngữ nghĩa OpenTelemetry + GenAI

OpenTelemetry đã xuất bản các quy ước ngữ nghĩa GenAI vào cuối năm 2025 (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`). Các công cụ tiêu thụ OTel có thể tương tác. Mô hình production nổi lên:

1. Phát ra OTel với các quy ước GenAI từ mỗi cuộc gọi LLM.
2. Tuyến đường đến gateway (Helicone / Portkey) hàng ngày.
3. Dual-ship để đánh giá nền tảng (Phoenix / Langfuse) để hồi quy.
4. Lưu trữ trong hồ dữ liệu (Iceberg) để phân tích dài hạn thông qua Arize AX hoặc DuckDB.

### Bẫy: đo lường sai lớp

Thiết bị đo lường bên trong agent framework của bạn (ví dụ: thêm LangSmith traces) kết hợp bạn với framework đó. Thiết bị đo lường ở lớp HTTP/OpenAI-SDK (thông qua OpenLLMetry hoặc gateway của bạn) có thể di động.

### Sampling - bạn không thể giữ mọi thứ

Với requests/day 1 triệu >, chi phí giữ chân toàn trace cao hơn so với các cuộc gọi LLM. Mẫu theo quy tắc: 100% lỗi, 100% chi phí cao, 5% thành công. Giữ cốt liệu luôn luôn; giữ sống cho đuôi dài.

### Những con số bạn nên nhớ

- Langfuse miễn phí cloud: 50K events/month.
- LangSmith: $39/user/month.
- Trực thăng miễn phí: 100K req/month.
- Arize AX tuyên bố: rẻ hơn ~100 lần so với nguyên khối trên quy mô lớn.
- Các quy ước OpenTelemetry GenAI: shipping 2025, 2026 được áp dụng rộng rãi.

## Ứng dụng

`code/main.py` mô phỏng một ngày từ 1 triệu trace qua các chiến lược lưu giữ (100% nhập, sampling, sampling + lỗi). Báo cáo chi phí lưu trữ và những gì bị mất trong mỗi tài khoản.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-observability-stack.md`. Cho stack, quy mô, ngân sách, tư thế giấy phép, chọn (các) công cụ.

## Bài tập

1. Nhóm của bạn trên LangChain muốn observability tự lưu trữ OSS. Chọn Langfuse hoặc Opik và biện minh.
2. Ở 5M traces/day với báo giá Datadog $150K/month, tính toán hòa vốn cho Arize AX.
3. Thiết kế thuộc tính OpenTelemetry GenAI đặt hướng dẫn của tổ chức của bạn nên bắt buộc trên mỗi cuộc gọi LLM.
4. Tranh luận liệu chỉ riêng Phượng hoàng có đủ cho production hay không. Khi nào thì không đủ?
5. Helicone cách proxy 20ms trên cao. Ở P99 TTFT 300 ms, điều đó có chấp nhận được không? Điều gì sẽ xảy ra nếu SLA là 100 ms?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| OpenLLMetry | "OTel cho LLMs" | Thiết bị đo lường OpenTelemetry mã nguồn mở cho LLMs |
| Quy ước GenAI | "Thuộc tính OTel" | Tên thuộc tính OTel chuẩn cho lệnh gọi LLM |
| Thợ rèn LangSmith | "observability LangChain" | Nền tảng thương mại đi kèm với hệ sinh thái LangChain |
| Cầu chì | "OSS LangSmith" | MIT OSS với bộ feature tương tự |
| Phượng hoàng | "Công cụ phát triển Arize" | Nền tảng dev/eval gốc OpenTelemetry |
| Arize AX | "Tỷ lệ observability" | Iceberg/Parquet observability không sao chép thương mại |
| Trực thăng | "proxy observability" | HTTP proxy Thu thập LLM telemetry + gateway features |
| Thuốc lá | "Sao chổi LLM" | Nền tảng phát triển Apache 2.0 OSS từ Comet |
| Session phát lại | "trace chạy lại" | Phát lại toàn bộ agent session với các lệnh gọi công cụ |
| Đánh giá | "Kiểm tra ngoại tuyến" | Chạy ứng cử viên model/prompt quá dataset được dán nhãn |

## Đọc thêm

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
