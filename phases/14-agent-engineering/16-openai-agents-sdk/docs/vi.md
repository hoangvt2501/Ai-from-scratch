# OpenAI Agents SDK: Bàn giao, Guardrails, Truy tìm

> OpenAI Agents SDK là đa agent framework nhẹ được xây dựng trên API Phản hồi. Năm primitives: Agent, Handoff, Guardrail, Session, Tracing. Handoffs là công cụ có tên `transfer_to_<agent>`. Guardrails chuyến đi trên đầu vào hoặc đầu ra. Theo mặc định, tính năng theo dõi được bật.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Vòng lặp Agent), Giai đoạn 14 · 06 (Sử dụng công cụ)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Kể tên năm primitives của OpenAI Agents SDK.
- Giải thích bàn giao: tại sao chúng được mô hình hóa như công cụ, tên định hình mà model nhìn thấy và cách chuyển ngữ cảnh.
- Phân biệt guardrails đầu vào, guardrails đầu ra và guardrails dụng cụ; Giải thích chế độ `run_in_parallel` và chặn.
- Triển khai runtime stdlib với chuyển giao + guardrails + theo dõi kiểu span.

## Vấn đề

Agents không thể ủy thác một cách sạch sẽ cuối cùng sẽ nhồi nhét mọi thứ vào một prompt. Agents mà không guardrails ship PII, đầu ra vi phạm policy hoặc lặp lại mãi mãi. OpenAI's SDK hệ thống hóa ba primitives làm cho công việc đa agent trở nên dễ dàng.

## Khái niệm

### Năm primitives

1. **Agent.** LLM + hướng dẫn + công cụ + bàn giao.
2. **Bàn giao.** Phái đoàn đến một agent khác. Được thể hiện cho model dưới dạng một công cụ có tên là `transfer_to_<agent_name>`.
3. **Guardrail.** Xác thực trên đầu vào (chỉ agent đầu tiên), đầu ra (chỉ agent cuối cùng) hoặc gọi công cụ (mỗi công cụ chức năng).
4. **Session.** Lịch sử hội thoại tự động qua các lượt.
5. **Truy kìm.** spans tích hợp cho các thế hệ LLM, gọi công cụ, bàn giao guardrails.

### Handoffs làm công cụ

model thấy `transfer_to_billing_agent` trong danh sách công cụ của mình. Gọi nó báo hiệu runtime:

1. Sao chép ngữ cảnh cuộc trò chuyện (hoặc thu gọn nó thông qua `nest_handoff_history` beta).
2. Khởi tạo agent mục tiêu theo hướng dẫn của nó.
3. Tiếp tục chạy với mục tiêu agent.

Đây là mô hình giám sát (Bài 13 / Bài 28) được sản xuất.

### Guardrails

Ba hương vị:

- **Đầu vào guardrails.** Chạy trên đầu vào của agent đầu tiên. Từ chối các yêu cầu không an toàn hoặc nằm ngoài phạm vi trước bất kỳ cuộc gọi LLM nào.
- **Đầu ra guardrails.** Chạy trên đầu ra của agent cuối cùng. Phát hiện rò rỉ PII, vi phạm policy, phản hồi không đúng định dạng.
- **Công cụ guardrails.** Chạy công cụ theo chức năng. Xác thực đối số, kiểm tra quyền, kiểm tra việc thực hiện.

Chế độ:

- **Song song** (mặc định). Guardrail LLM chạy dọc theo LLM chính. Độ trễ đuôi thấp hơn. Nếu vấp ngã, công việc của LLM chính sẽ bị loại bỏ (token chất thải).
- **Chặn** (`run_in_parallel=False`). Guardrail LLM chạy trước. Nếu bị vấp ngã, không tokens lãng phí cuộc gọi chính.

Tripwires nâng `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`.

### Truy tìm

Bật theo mặc định. Mỗi thế hệ LLM, gọi công cụ, chuyển giao và guardrail phát ra một span. `OPENAI_AGENTS_DISABLE_TRACING=1` chọn không tham gia. `add_trace_processor(processor)` người hâm mộ spans đến phần phụ trợ của riêng bạn cùng với OpenAI.

### Sessions

`Session` lưu trữ lịch sử cuộc trò chuyện trong phần phụ trợ (SQLite, Redis, tùy chỉnh). `Runner.run(agent, input, session=session)` tự động tải và thêm vào.

### Mô hình này sai ở đâu

- **Trôi dạt bàn giao.** Agent A chuyển tay cho Agent B, giao lại cho Agent A. Thêm bộ đếm nhảy.
- **Guardrail bỏ qua.** Công cụ chỉ guardrails kích hoạt trên các công cụ chức năng; Các công cụ tích hợp (trình đọc tệp, tìm nạp web) cần policy riêng.
- **Truy tìm quá mức.** Nội dung nhạy cảm trong spans. Kết hợp với các quy tắc chụp nội dung OTel GenAI (Bài 23) — lưu trữ bên ngoài, tham khảo theo ID.

## Tự xây dựng

`code/main.py` triển khai hình dạng SDK trong stdlib:

- `Agent`, `FunctionTool`, `Handoff` (như một công cụ chức năng với ngữ nghĩa truyền).
- `Runner` với input/output/tool guardrails, công văn bàn giao và bộ đếm nhảy.
- Một bộ phát span đơn giản để hiển thị hình dạng trace.
- Một agent phân loại chuyển giao cho việc thanh toán hoặc hỗ trợ dựa trên truy vấn của người dùng; guardrail chuyến đi trên một đầu vào.

Chạy nó:

```
python3 code/main.py
```

trace cho thấy hai lần chuyển giao thành công, một lần đầu vào guardrail chuyến đi và một cây span phản ánh những gì SDK thực phát ra.

## Ứng dụng

- **OpenAI Agents SDK** đối với các sản phẩm ưu tiên OpenAI.
- **Claude Agent SDK** (Bài 17) đối với sản phẩm ưu tiên Claude.
- **LangGraph** (Bài 13) khi bạn muốn trạng thái rõ ràng và sơ yếu lý lịch lâu dài.
- **Tùy chỉnh** khi bạn cần kiểm soát chính xác (triển khai thoại, nhiều nhà cung cấp, liên kết).

## Sản phẩm bàn giao

`outputs/skill-agents-sdk-scaffold.md` giàn giáo một ứng dụng Agents SDK với agent phân loại, bàn giao, input/output/tool guardrails, cửa hàng session và bộ xử lý trace.

## Bài tập

1. Thêm bộ đếm chuyển giao: từ chối sau khi N chuyển. Trace hành vi.
2. Triển khai `nest_handoff_history` như một tùy chọn — thu gọn prior thư thành một bản tóm tắt trước khi chuyển.
3. Viết một guardrail đầu ra chặn. So sánh độ trễ trên prompts sẽ vấp ngã với những người vượt qua.
4. Nối dây `add_trace_processor` với bộ ghi JSON. Nó phát ra hình dạng nào mỗi span?
5. Đọc tài liệu SDK. Chuyển đồ chơi stdlib của bạn vào `openai-agents-python`. Bạn đã model sai điều gì?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Agent | "LLM + hướng dẫn" | Agent loại trong SDK; sở hữu các công cụ và bàn giao |
| Bàn giao | "Chuyển khoản" | Công cụ model gọi để ủy quyền cho một agent khác |
| Guardrail | "Kiểm tra Policy" | Xác thực khi gọi đầu vào / đầu ra / công cụ |
| Dây ngắt | "Chuyến đi Guardrail" | Ngoại lệ được đưa ra khi guardrail từ chối |
| Session | "Cửa hàng lịch sử" | Bộ nhớ hội thoại vẫn tồn tại giữa các lần chạy |
| Truy tìm | "Spans" | Tích hợp observability qua LLM + công cụ + bàn giao + guardrail |
| Chặn guardrail | "Kiểm tra tuần tự" | Guardrail chạy trước; Không token lãng phí trong chuyến đi |
| guardrail song song | "Kiểm tra đồng thời" | Guardrail chạy cùng nhau; Độ trễ thấp hơn, lãng phí tokens trong chuyến đi |

## Đọc thêm

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — primitives, bàn giao, guardrails, truy tìm
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) - đối tác hương vị Claude
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - khi nào cần tiếp cận bàn giao
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — bản đồ Agents SDK spans tiêu chuẩn đến
