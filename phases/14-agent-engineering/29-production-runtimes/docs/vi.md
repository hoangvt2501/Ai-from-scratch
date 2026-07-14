# Production Runtimes: Hàng đợi, Sự kiện, Cron

> Production agents chạy trên sáu hình dạng runtime: phản hồi yêu cầu, streaming, thực thi lâu dài, nền dựa trên hàng đợi, theo hướng sự kiện và theo lịch trình. Chọn hình dạng trước khi bạn chọn framework. Observability chịu tải ở mọi hình dạng.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 13 (LangGraph), Giai đoạn 14 · 22 (Giọng nói)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Đặt tên cho sáu hình dạng production runtime và khớp từng hình với một mẫu framework / sản phẩm.
- Giải thích lý do tại sao thực thi bền bỉ (LangGraph) lại quan trọng đối với các tác vụ dài hạn.
- Mô tả runtime theo sự kiện và thời điểm Claude Managed Agents phù hợp.
- Giải thích tuyên bố observability chịu lực cho agents nhiều bước.

## Vấn đề

Production agents lỗi theo cách mà máy tính xách tay Jupyter không xuất hiện: timeouts mạng ở bước 37, người dùng ngắt cuộc gọi thoại, công việc cron chết khi khởi động lại máy, worker nền hết bộ nhớ. Hình dạng runtime xác định lỗi nào có thể sống sót.

## Khái niệm

### Yêu cầu-phản hồi

- HTTP đồng bộ. Người dùng chờ hoàn tất.
- Chỉ khả thi cho các tác vụ ngắn (<30 giây).
- Stacks: Agno (Python + FastAPI), Mastra (TypeScript + Express/Hono/Fastify/Koa).
- Observability: nhật ký truy cập HTTP tiêu chuẩn + spans OTel.

### Streaming

- SSE hoặc WebSocket cho đầu ra lũy tiến.
- LiveKit mở rộng điều này cho WebRTC cho voice/video (Bài 22).
- Stacks: bất kỳ framework nào có hỗ trợ streaming + giao diện người dùng xử lý SSE/WS.
- Observability: thời gian trên mỗi khối, độ trễ token đầu tiên, độ trễ đuôi.

### Thực thi bền bỉ

- Trạm kiểm soát nhà nước sau mỗi bước; tự động tiếp tục khi thất bại.
- AutoGen v0.4 actor model cô lập các lỗi thành một agent (Bài 14).
- Bộ vi phân cốt lõi của LangGraph (Bài 13).
- Cần thiết khi không xác định số bước và chi phí phục hồi cao.

### Dựa trên hàng đợi / nền

- Công việc vào hàng đợi, workers nhấc máy, kết quả quay trở lại qua webhooks hoặc pub/sub.
- Cần thiết cho agents đường chân trời dài (hàng chục đến hàng trăm bước cho mỗi tác vụ, theo thông báo sử dụng máy tính của Anthropic).
- Stacks: Cần tây (Python), BullMQ (Node), SQS + Lambda (AWS), tùy chỉnh.
- Observability: độ sâu hàng đợi, phân phối độ trễ cho mỗi tác vụ, kích thước DLQ.

### Định hướng sự kiện

- Agents đăng ký triggers: email mới, PR mở, cron fire.
- Claude Managed Agents đề cập đến điều này ngay lập tức (Bài 17).
- CrewAI Flows (Bài 15) cấu trúc quy trình làm việc xác định theo sự kiện.
- Observability: nguồn trigger, độ trễ từ sự kiện đến khi bắt đầu agent độ trễ.

### Đã lên lịch

- agents hình Cron chạy định kỳ.
- Kết hợp với khả năng thực hiện bền bỉ để một lần chạy hàng đêm thất bại sẽ tiếp tục tick tiếp theo.
- Stacks: Kubernetes CronJob + framework bền; hosted (Render cron, Vercel cron).

### Mô hình triển khai năm 2026

- **CrewAI Flows** dành cho production theo sự kiện.
- **Agno** FastAPI không trạng thái cho Python vi dịch vụ.
- **Mastra** server bộ điều hợp (Express, Hono, Fastify, Koa) cho embedding.
- **Pipecat Cloud / LiveKit Cloud** cho giọng nói được quản lý (Bài 22).
- **Claude Agents được quản lý** để lưu trữ không đồng bộ chạy trong thời gian dài.

### Observability chịu tải

Nếu không có OpenTelemetry GenAI spans (Bài 23) cộng với phần phụ trợ Langfuse/Phoenix/Opik (Bài 24), bạn không thể gỡ lỗi agent nhiều bước không thành công ở bước 40. Điều này không phải là tùy chọn đối với production. Đó là sự khác biệt giữa "chúng tôi gỡ lỗi nhanh" và "chúng tôi phát lại từ đầu với nhiều ghi nhật ký hơn".

### Nơi production runtimes thất bại

- **Lựa chọn hình dạng sai.** Chọn yêu cầu-phản hồi cho một nhiệm vụ kéo dài 5 phút. Người dùng gác máy; workers chất đống; thử lại hợp chất.
- **Không có DLQ.** Hàng đợi workers mà không có thư chết. Những công việc thất bại biến mất.
- **Công việc nền mờ đục.** agent nền chạy mà không cần xuất trace. Lỗi sẽ không hiển thị cho đến khi người dùng báo cáo chúng.
- **Bỏ qua trạng thái bền bỉ.** Bất kỳ lần chạy nào > 30 giây mà bạn không đủ khả năng khởi động lại đều cần thực thi bền bỉ.

## Tự xây dựng

`code/main.py` là một bản demo đa hình dạng stdlib:

- endpoint yêu cầu-phản hồi (hàm thuần túy).
- Streaming xử lý (máy phát điện).
- worker dựa trên hàng đợi với DLQ.
- Sự kiện trigger registry.
- Bộ lập lịch hình Cron.

Chạy nó:

```bash
python3 code/main.py
```

Đầu ra: năm traces hiển thị hành vi của mỗi hình dạng trên cùng một nhiệm vụ. Cùng logic agent, vỏ ngoài khác nhau. Thực thi bền bỉ (hình dạng thứ sáu) được đề cập có chủ ý trong Bài 13 với điểm kiểm tra LangGraph.

## Ứng dụng

- **Yêu cầu-phản hồi** cho UX kiểu trò chuyện.
- **Streaming** cho các phản hồi tiến bộ.
- **Bền bỉ** cho các tác vụ đường chân trời dài.
- **Hàng đợi** cho batch / không đồng bộ / chạy dài.
- **Sự kiện** để phản ứng agent.
- **Cron** để dọn dẹp (hợp nhất bộ nhớ, đánh giá, báo cáo chi phí).

## Sản phẩm bàn giao

`outputs/skill-runtime-shape.md` chọn một hình dạng runtime cho một nhiệm vụ và đưa ra các yêu cầu observability.

## Bài tập

1. Chuyển vòng lặp Bài học 01 ReAct của bạn sang tất cả sáu hình dạng trong stack của bạn. Hình dạng nào phù hợp với bề mặt sản phẩm nào?
2. Thêm DLQ vào bản demo dựa trên hàng đợi. Mô phỏng thất bại 10% công việc; kích thước DLQ bề mặt.
3. Viết một agent đánh giá được kích hoạt bởi cron chạy hàng đêm so với 20 traces hàng đầu của bạn trong ngày.
4. Thực hiện streaming với áp suất ngược: nếu máy khách chậm, hãy tạm dừng agent. Điều này tương tác như thế nào với ngân sách lượt?
5. Đọc Claude tài liệu Managed Agents. Khi nào bạn sẽ chuyển một agent chân trời dài tự lưu trữ sang quản lý?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Yêu cầu-phản hồi | "Đồng bộ" | Người dùng chờ đợi; Chỉ nhiệm vụ ngắn |
| Streaming | "SSE / WS" | Đầu ra lũy tiến; UX tốt hơn; độ trễ có thể quan sát được trên mỗi khối |
| Thực thi bền bỉ | "Tiếp tục từ thất bại" | Trạng thái trạm kiểm soát; Khởi động lại ở bước cuối cùng |
| Dựa trên hàng đợi | "Công việc nền" | Nhà sản xuất / Nhóm worker / DLQ |
| Định hướng sự kiện | "Dựa trên Trigger" | Agent phản ứng với các sự kiện bên ngoài |
| DLQ | "Hàng đợi thư chết" | Bãi đậu xe cho các công việc thất bại |
| Claude Agents được quản lý | "Được lưu trữ harness" | Không đồng bộ chạy lâu dài được lưu trữ Anthropic với bộ nhớ đệm + nén |

## Đọc thêm

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — chi tiết khớp lệnh bền bỉ
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — lưu trữ không đồng bộ chạy lâu dài
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "hàng chục đến hàng trăm bước cho mỗi tác vụ"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) - cách ly lỗi model diễn viên
