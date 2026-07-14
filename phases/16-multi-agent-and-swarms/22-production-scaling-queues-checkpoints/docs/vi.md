# Production Mở rộng quy mô — Hàng đợi, Checkpoints, Độ bền

> Thay đổi quy mô hệ thống đa agent lên hàng nghìn lần chạy đồng thời đòi hỏi **thực thi bền bỉ**. runtime của LangGraph viết một checkpoint sau mỗi siêu bước được khóa bởi `thread_id` (Postgres theo mặc định); worker sự cố giải phóng hợp đồng thuê và một worker khác tiếp tục. Agents có thể ngủ vô thời hạn chờ đợi ý kiến của con người. **MegaAgent** (arXiv:2408.09955) chạy hàng đợi nhà sản xuất-người tiêu dùng trên mỗi agent với ba trạng thái (Nhàn rỗi / Xử lý / Phản hồi) và phối hợp hai lớp (trò chuyện trong nhóm + trò chuyện quản trị viên giữa các nhóm). **Fiber/async** đánh bại thread cho mỗi công việc trong LLM streaming: threads ngồi nhàn rỗi 99% thời gian chờ đợi tokens, các sợi quang hợp tác tạo ra trên I/O. Counterpoint: "Scaling Agentic Software" của Ashpreet Bedi lập luận cho **FastAPI + Postgres + không có gì khác** cho đến khi tải chứng minh điều ngược lại - các kiến trúc đơn giản đi xa hơn mong đợi. Bài học này xây dựng nhật ký checkpoint bền vững, hàng đợi công việc trên mỗi agent với chuyển đổi trạng thái, bản demo không đồng bộ so với thread và đưa ra quy tắc "bắt đầu đơn giản" thực dụng.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib, `asyncio`, `sqlite3`)
**Kiến thức tiên quyết:** Giai đoạn 16 · 09 (Mạng Swarm song song), Giai đoạn 16 · 13 (Bộ nhớ chia sẻ)
**Thời lượng:** ~75 phút

## Vấn đề

Một hệ thống đa agent nguyên mẫu hoạt động trên một máy tính xách tay với ba agents trong vòng lặp sự kiện trong bộ nhớ. Bạn chuyển đến production:

- Agents đôi khi chạy hàng giờ (nghiên cứu lâu, con người chờ đợi).
- Worker processes sự cố. Khởi động lại sẽ mất trạng thái.
- Tải trọng cực đại là trung bình gấp 10 lần; Bạn cần mở rộng theo chiều ngang.
- Người dùng trả tiền cho mỗi agent chạy; Bạn cần ngữ nghĩa chính xác một lần để sạc.

Vòng lặp sự kiện trong bộ nhớ không thực hiện bất kỳ điều nào trong số này. Bạn cần một lớp thực thi bền bên dưới. Các tùy chọn chuẩn năm 2026 là:

1. Một công cụ quy trình làm việc với checkpoints (Temporal, LangGraph runtime).
2. Hàng đợi tin nhắn có kho lưu trữ trạng thái (Postgres + SQS/RabbitMQ).
3. Actor-model frameworks (nhà sản xuất-người tiêu dùng của MegaAgent trên mỗi agent).
4. FastAPI + Postgres cuộn tay (lập luận của Bedi).

Bài học này xây dựng một bản thu nhỏ của mỗi loại.

## Khái niệm

### Thực thi bền bỉ, mô hình

Một công cụ thực thi bền bỉ duy trì trạng thái chương trình đầy đủ sau mỗi "bước" (siêu bước, trong ngôn ngữ của LangGraph). Khi gặp sự cố:

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

Yêu cầu để điều này hoạt động:

- **Trạng thái có thể tuần tự.** Tất cả trạng thái agent phải có thể duy trì. Việc đóng hàm với các kết nối cơ sở dữ liệu trực tiếp không tồn tại.
- **Sơ yếu lý lịch xác định.** Cho cùng một trạng thái và cùng đầu vào, agent tạo ra các hành động giống nhau (hoặc trì hoãn một oracle xác định bên ngoài cho các cuộc gọi LLM).
- **Tác dụng phụ idempotent.** Các cuộc gọi bên ngoài (lệnh gọi công cụ, thanh toán) phải là idempotent hoặc sử dụng khóa khử trùng lặp.

LangGraph viết một checkpoint sau mỗi siêu bước; Tạm thời viết sau mỗi hoạt động; Restate sử dụng nhật ký có nguồn sự kiện. Cả ba đều thực hiện cùng một mẫu.

### runtime của LangGraph

Mỗi agent có một `thread_id`; state là một dict được nhập; Mỗi siêu bước ghi một hàng vào bảng checkpoints. Trên tiếp tục, runtime phát lại từ checkpoint trước, không phải từ đầu. Agents có thể `interrupt()` chờ đợi đầu vào của con người; runtime tồn tại và giải phóng worker. Khi đầu vào đến, bất kỳ worker nào cũng có thể tiếp tục.

Đây là thiết kế production tham khảo vào tháng 4 năm 2026.

### Hàng đợi mỗi agent của MegaAgent

arXiv:2408.09955 mô tả một thí nghiệm quy mô: hàng ngàn agents đồng thời trong một cụm. Kiến trúc:

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

Sự phối hợp hai lớp cho phép cuộc trò chuyện trong nhóm diễn ra dày đặc trong khi giữa các nhóm vẫn thưa thớt - mô hình được sử dụng để giữ chi phí tuyến tính trong hàng nghìn agents.

### Không đồng bộ so với thread cho mỗi công việc

LLM cuộc gọi I/O-bound. Một thread chờ đợi token tiếp theo là nhàn rỗi 99% thời gian. Threads có giá ~1MB RAM mỗi trang; với 10.000 cuộc gọi đồng thời, tức là 10GB chỉ dành cho stacks.

Sợi (Python `asyncio`, Go goroutines, Rust `tokio`) hợp tác tạo ra I/O. 10.000 cuộc gọi tương tự phù hợp với process. Ở quy mô LLM-agent, không đồng bộ không phải là một tối ưu hóa - đó là kiến trúc.

Ngoại lệ: xử lý hậu kỳ ràng buộc CPU (embedding, thủ thuật tokenizer) vẫn muốn threads hoặc processes. Tách layer I/O của bạn khỏi layer CPU của bạn.

### Đối lập của Bedi

"Phần mềm Agentic mở rộng quy mô" (Ashpreet Bedi, 2026) lập luận rằng hầu hết các nhóm đều thiết kế quá mức trước khi họ đo tải. Mặc định thực dụng:

- FastAPI + Postgres.
- Mỗi agent chạy là một hàng; trạng thái được cập nhật tại chỗ với tính đồng thời lạc quan.
- Công việc nền tảng thông qua `pg_notify` hoặc worker Celery đơn giản.
- Thử lại policy trong mã ứng dụng.

Đối với tải dưới ~100 lần chạy agent đồng thời trên các tác vụ có thể quản lý, đây thường là tất cả những gì bạn cần. Nâng cấp khi bạn đo lường nó không thành công.

Quy tắc: áp dụng frameworks thực thi bền vững khi bạn gặp phải một vấn đề cụ thể mà các kiến trúc đơn giản không thể giải quyết. Việc nhận con nuôi sớm đốt cháy thời gian cho các nghi lễ không được đền đáp.

### Ngữ nghĩa chính xác một lần

Để chạy agent trả phí, bạn cần "chính xác một lần hiệu quả" (phân phối ít nhất một lần + người tiêu dùng idempotent). Kỹ thuật di chuyển:

- **Phím Dedup mỗi lần chạy.** Bao gồm nó trong mọi cuộc gọi hiệu ứng phụ.
- **Mẫu hộp thư đi.** Các hiệu ứng phụ ghi vào bảng trước, sau đó một process riêng biệt sẽ thực thi chúng. Cả hai bước đều idempotent.
- **Giao dịch bồi thường.** Khi tác dụng phụ thành công nhưng ghi theo dõi không thành công, hãy lên lịch bồi thường.

Đây là các mẫu kỹ thuật cơ sở dữ liệu, không dành riêng cho LLM. Thuế LLM chỉ là các cuộc gọi LLM chậm; mọi thứ khác là các hệ thống phân tán tiêu chuẩn.

### Triển khai cầu vồng

Hệ thống nghiên cứu đa agent của Anthropic sử dụng "triển khai cầu vồng": nhiều phiên bản của agent runtime chạy đồng thời nên agents chạy lâu dài không phải bị giết trên mỗi lần triển khai mã. Canary các phiên bản mới trên một phần giao thông; Nghỉ hưu các phiên bản cũ khi agents của chúng kết thúc.

Đây là tiêu chuẩn cho các hệ thống trạng thái chạy lâu dài; Sự thích ứng vào năm 2026 là agents có thể sống hàng giờ, vì vậy các chu kỳ triển khai phải phù hợp.

### Danh sách kiểm tra production chuẩn

- Trạng thái bền (checkpoints, ảnh chụp nhanh hoặc hộp thư đi + nhật ký có thể phát lại).
- Tác dụng phụ idempotent.
- Lớp I/O không đồng bộ cho các cuộc gọi LLM.
- Giao hàng ít nhất một lần với dedup.
- Rainbow/canary triển khai cho khối lượng công việc có trạng thái.
- Observability: mỗi agent traces, kiểm tra siêu bước, bộ đếm thử lại.

## Tự xây dựng

`code/main.py` thực hiện:

- `CheckpointStore` — Nhật ký checkpoint được hỗ trợ bởi SQLite với các khóa thread-id. Mỗi siêu bước nối thêm một hàng.
- `run_with_checkpoint(agent, thread_id)` - mô phỏng một vụ tai nạn giữa chừng; worker thứ hai tiếp tục từ checkpoint trước.
- `AgentQueue` - máy trạng thái Idle / Processing / Response trên mỗi agent với một hàng đợi công việc nhỏ.
- `demo_async_vs_threads()` - chạy 500 "cuộc gọi LLM" mô phỏng đồng thời thông qua asyncio và qua threads; báo cáo đồng hồ treo tường và bộ nhớ cao nhất (gần đúng).

Chạy:

```
python3 code/main.py
```

Đầu ra dự kiến: checkpoint tiếp tục thành công sau khi mô phỏng sự cố; phiên bản async xử lý 500 cuộc gọi đồng thời trong < 1; thread bản mất vài giây và sử dụng nhiều bộ nhớ hơn cho mỗi đơn vị đồng thời.

## Ứng dụng

`outputs/skill-scaling-advisor.md` tư vấn về lựa chọn thực thi bền bỉ: FastAPI + Postgres, LangGraph runtime, Temporal hoặc custom. Được hiệu chỉnh theo tải, nhu cầu duy trì trạng thái và tần suất triển khai.

## Sản phẩm bàn giao

Làm cứng production chuẩn:

- **Bắt đầu đơn giản (quy tắc của Bedi).** FastAPI + Postgres cho đến khi bạn đo lường nó không thành công.
- **Đo lường mọi thứ trước khi tối ưu hóa.** Biểu đồ độ trễ mỗi lần chạy, thời gian mỗi bước, số lần thử lại, phân loại lỗi.
- **Mẫu hộp thư đi cho các tác dụng phụ.** Đặc biệt là thanh toán và các cuộc gọi API bên ngoài.
- **Rainbow triển khai.** Không bao giờ giết các agent chạy trên máy bay trong quá trình triển khai.
- **Áp dụng các công cụ thực thi bền bỉ (Temporal / LangGraph / Restate) khi** bạn gặp phải các vấn đề cụ thể: chờ đợi con người trong vòng lặp kéo dài hàng giờ, phối hợp giữa các khu vực, retry/compensation policies phức tạp.
- **Async cho lớp I/O.** Threads chỉ dành cho xử lý hậu kỳ ràng buộc CPU.

## Bài tập

1. Chạy `code/main.py`. Xác nhận checkpoint tiếp tục công việc; đo lường sự khác biệt không đồng bộ so với thread đồng thời.
2. Triển khai bảng **hộp thư đi**: mọi lệnh gọi công cụ sẽ ghi vào hộp thư đi trước, sau đó một goroutine/task riêng biệt sẽ thực thi. Xác minh idempotency bằng cách chạy lệnh gọi công cụ hai lần.
3. Mô phỏng **triển khai cầu vồng**: hai phiên bản runtime đồng thời; định tuyến một nửa thread_ids mới đến mỗi nơi; Xác nhận rằng threads trên chuyến bay trên phiên bản cũ không bị gián đoạn.
4. Đọc tài liệu runtime của LangGraph (được liên kết bên dưới). Xác định features nào trong runtime sẽ mất nhiều thời gian nhất để sao chép trong phiên bản FastAPI + Postgres cuộn tay. Đó có phải là lý do để nhận con nuôi, hay bạn có thể trì hoãn?
5. Đọc MegaAgent (arXiv: 2408.09955) Phần 3. Sự phối hợp hai lớp (nội bộ nhóm + trò chuyện quản trị viên giữa các nhóm) là rõ ràng. Phác thảo cách bạn sẽ ánh xạ điều này với hàng đợi tin nhắn có hai họ hàng đợi.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Thực thi bền bỉ | "Duy trì trạng thái chương trình" | Trạng thái ghi công cụ sau mỗi siêu bước; Phục hồi sự cố là xác định. |
| Siêu bước | "Ranh giới giao dịch" | Đơn vị làm việc giữa checkpoints. Thuật ngữ LangGraph. |
| thread_id | "Agent mã định danh chạy" | Khóa liên kết logic checkpoints và tiếp tục. |
| Idempotency | "An toàn để thử lại" | Lặp lại một tác dụng phụ tạo ra kết quả tương tự như một lần thử. |
| Mẫu hộp thư đi | "Tác dụng phụ tách rời" | Viết ý định vào bảng; Một người thực thi riêng biệt thực hiện và đánh dấu xong. |
| Giao hàng ít nhất một lần | "Các bản sao có thể xảy ra" | Ngữ nghĩa hàng đợi tin nhắn; Dedup làm cho người tiêu dùng hiệu quả một lần. |
| Triển khai cầu vồng | "Phiên bản chồng chéo" | Nhiều phiên bản runtime đồng thời trong khối lượng công việc chạy dài. |
| Sợi không đồng bộ | "Lợi nhuận hợp tác" | Đồng thời chế độ người dùng; rẻ so với threads cho tải I/O-bound. |
| Checkpoint | "Ảnh chụp nhanh trạng thái" | Trạng thái tuần tự ở ranh giới siêu bước; chìa khóa cho sơ yếu lý lịch. |

## Đọc thêm

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — Thiết kế runtime LangGraph
- [MegaAgent](https://arxiv.org/abs/2408.09955) — hàng đợi sản xuất-người tiêu dùng trên mỗi agent; phối hợp hai lớp tại hàng nghìn agents đồng thời
- [Matrix](https://arxiv.org/abs/2511.21686) — framework phi tập trung với hàng đợi tin nhắn làm nền điều phối
- [Temporal docs](https://docs.temporal.io/) — công cụ quy trình làm việc tham chiếu để thực thi bền bỉ
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — production bài học bao gồm triển khai cầu vồng
