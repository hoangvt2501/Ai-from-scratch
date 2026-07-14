# AutoGen v0.4: Actor Model và Agent Framework

> AutoGen v0.4 (Nghiên cứu của Microsoft, tháng 1 năm 2025) đã thiết kế lại agent orchestration xung quanh diễn viên model. Trao đổi tin nhắn không đồng bộ, agents theo sự kiện, cách ly lỗi, đồng thời tự nhiên. framework hiện đang ở chế độ bảo trì trong khi Microsoft Agent Framework (bản xem trước công khai tháng 10 năm 2025) trở thành người kế nhiệm.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Vòng lặp Agent), Giai đoạn 14 · 12 (Mẫu quy trình làm việc)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Mô tả tác nhân model: agents là tác nhân, tin nhắn là IPC duy nhất, cách ly lỗi cho mỗi tác nhân.
- Đặt tên cho ba lớp API của AutoGen v0.4 — Core, AgentChat, Extensions — và mỗi lớp dùng để làm gì.
- Giải thích lý do tại sao việc tách việc phân phối tin nhắn khỏi việc xử lý mang lại sự cô lập lỗi và đồng thời tự nhiên.
- Triển khai một diễn viên stdlib runtime trong Python và chuyển quy trình xem xét mã hai agent vào đó.

## Vấn đề

Hầu hết các agent frameworks đều đồng bộ: một agent sản xuất, một agent tiêu thụ, trong một cuộc gọi stack. Thất bại làm sụp đổ stack. Tính đồng thời được bắt vít. Phân phối yêu cầu viết lại.

Câu trả lời của AutoGen v0.4: diễn viên model. Mỗi agent là một diễn viên có hộp thư đến riêng. Tin nhắn là tương tác duy nhất. runtime tách giao hàng khỏi việc xử lý. Thất bại cô lập ở một tác nhân. Tính đồng thời là gốc. Phân phối chỉ khác transport.

## Khái niệm

### Diễn viên

Một diễn viên có:

- Một nhà nước tư nhân (không bao giờ chạm trực tiếp từ bên ngoài).
- Hộp thư đến (hàng đợi tin nhắn).
- Trình xử lý: `receive(message) -> effects` nơi các hiệu ứng có thể là "trả lời", "gửi cho diễn viên khác", "sinh ra diễn viên mới", "cập nhật trạng thái", "dừng bản thân".

Hai diễn viên không thể chia sẻ ký ức. Họ chỉ có thể gửi tin nhắn.

### Ba lớp API trong AutoGen v0.4

1. **Cốt lõi.** Diễn viên cấp thấp framework. `AgentRuntime`, `Agent`, `Message`, `Topic`. Trao đổi tin nhắn không đồng bộ, theo hướng sự kiện.
2. **AgentChat.** API cấp cao theo hướng tác vụ (thay thế cho ConversableAgent của v0.2). `AssistantAgent`, `UserProxyAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`.
3. **Tiện ích mở rộng.** Tích hợp — OpenAI, Anthropic, Azure, công cụ, bộ nhớ.

### Tại sao việc tách rời lại quan trọng

Trong model v0.2, việc gọi `agent_a.chat(agent_b)` sẽ chặn đồng bộ agent_a cho đến khi agent_b trả về. Trong phiên bản 0.4, `send(agent_b, msg)` đưa thư vào hộp thư đến của agent_b và trả về. runtime cung cấp sau. Ba hậu quả:

- **Cách ly lỗi.** Agent Sự cố B không gặp sự cố Agent A - runtime bắt lỗi trong trình xử lý của B và quyết định phải làm gì (nhật ký, thử lại, thư chết).
- **Đồng thời tự nhiên.** Nhiều tin nhắn trên chuyến bay cùng một lúc; các diễn viên process hộp thư đến của họ đồng thời.
- **Sẵn sàng phân phối.** Inbox + transport là cùng một trừu tượng cho dù actor đang process hay trên một máy chủ khác.

### Cấu trúc liên kết

- **RoundRobinGroupChat.** Agents thay phiên nhau xoay vòng cố định.
- **SelectorGroupChat.** Bộ chọn agent chọn người tiếp theo dựa trên ngữ cảnh cuộc trò chuyện.
- **Magentic-One.** Tham khảo nhóm đa agent để duyệt web, thực thi mã, xử lý tệp. Được xây dựng trên AgentChat.

### Observability

Hỗ trợ OpenTelemetry được tích hợp sẵn. Mỗi thông điệp phát ra một span; các lệnh gọi công cụ mang các thuộc tính `gen_ai.*` theo quy ước ngữ nghĩa OTel GenAI năm 2026 (Bài 23).

### Tình trạng: chế độ bảo trì

Đầu năm 2026: AutoGen v0.7.x ổn định để nghiên cứu và tạo mẫu. Microsoft đã chuyển phát triển tích cực sang Microsoft Agent Framework (bản xem trước công khai ngày 1 tháng 10 năm 2025; 1.0 GA dự kiến vào cuối quý 1 năm 2026). Các mẫu AutoGen chuyển về phía trước một cách sạch sẽ - tác nhân model là ý tưởng bền bỉ.

## Tự xây dựng

`code/main.py` triển khai một runtime actor stdlib:

- `Message` — payload được gõ với `sender`, `recipient`, `topic`, `body`.
- `Actor` — trừu tượng với `receive(message, runtime)`.
- `Runtime` — vòng lặp sự kiện với hàng đợi, phân phối, cách ly thất bại được chia sẻ.
- Bản demo hai tác nhân: `ReviewerAgent` đánh giá mã `ChecklistAgent` chạy danh sách kiểm tra; họ trao đổi tin nhắn cho đến khi có sự đồng thuận.

Chạy nó:

```
python3 code/main.py
```

trace cho thấy việc gửi thông điệp, một lỗi mô phỏng ở một tác nhân không làm hỏng tác nhân kia và hội tụ trên một phán quyết được chia sẻ.

## Ứng dụng

- **AutoGen v0.4/v0.7** (bảo trì) — ổn định cho nghiên cứu, tạo mẫu, các mẫu đa agent.
- **Microsoft Agent Framework** (bản xem trước công khai) — đường dẫn chuyển tiếp; cùng một ý tưởng model diễn viên trong một API mới mẻ.
- **Cấu trúc liên kết LangGraph swarm** (Bài 13) — mẫu tương tự thông qua chuyển giao công cụ dùng chung.
- **Actor tùy chỉnh runtime** — khi bạn cần transport cụ thể (NATS, RabbitMQ, gRPC).

## Sản phẩm bàn giao

`outputs/skill-actor-runtime.md` tạo ra một runtime tác nhân tối thiểu cộng với một mẫu nhóm (RoundRobin hoặc Selector) cho một nhiệm vụ nhiều agent nhất định.

## Bài tập

1. Thêm hàng đợi thư chết: khi trình xử lý tăng lên, hãy đỗ thông báo không thành công để con người kiểm tra. DLQ thường bị đánh trong đồ chơi của bạn như thế nào?
2. Triển khai `SelectorGroupChat`: một tác nhân chọn chọn ai processes tin nhắn tiếp theo dựa trên trạng thái cuộc trò chuyện.
3. Thêm transport phân tán: hoán đổi hàng đợi trong process cho JSON trên HTTP server để các tác nhân có thể chạy trong các processes riêng biệt.
4. Nối span OTel cho mỗi tin nhắn (hoặc dự phòng không hoạt động). Phát ra `gen_ai.agent.name`, `gen_ai.operation.name` mỗi Bài 23.
5. Đọc bài đăng kiến trúc của AutoGen v0.4. Chuyển đồ chơi của bạn vào `autogen_core` API thật. Bạn đã bỏ qua điều gì quan trọng trong production?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Diễn viên | "Agent" | Trạng thái riêng + hộp thư đến + trình xử lý; Không có bộ nhớ dùng chung |
| Thông điệp | "Sự kiện" | Đánh máy payload; Cách duy nhất các tác nhân tương tác |
| Hộp thư đến | "Hộp thư" | Hàng đợi tin nhắn đang chờ xử lý cho mỗi tác nhân |
| Runtime | "Agent chủ nhà" | Vòng lặp sự kiện định tuyến tin nhắn và cô lập lỗi |
| Chủ đề | "Kênh" | Định tuyến xuất bản-đăng ký được đặt tên giữa các tác nhân |
| Cách ly lỗi | "Hãy để nó sụp đổ" | Một diễn viên thất bại không làm sụp đổ những người khác |
| RoundRobinGroupTrò chuyện | "Đội luân chuyển cố định" | Agents thay phiên nhau theo thứ tự |
| Bộ chọnGroupChat | "Nhóm định tuyến theo ngữ cảnh" | Bộ chọn chọn người đi tiếp theo |
| Magentic-Một | "Nhóm tham khảo" | Đội hình đa agent cho web + mã + tệp |

## Đọc thêm

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) - bài đăng thiết kế lại
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — thay thế hình đồ thị
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - spans AutoGen phát ra theo mặc định
